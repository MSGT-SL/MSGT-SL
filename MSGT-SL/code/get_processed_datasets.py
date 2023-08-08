import pickle

import numpy as np
import pandas as pd
import argparse, sys, json, random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import matplotlib.pyplot as plt

from utils import *
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, \
    f1_score
from model import *


def init_argparse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--type', type=str, default="feature", help="one of these: feature, comparison, test")
    parser.add_argument('--data_source', type=str, default="Jurkat", help="which cell line to train and predict")
    parser.add_argument('--threshold', type=float, default=-3, help="threshold of SL determination")
    parser.add_argument('--specific_graph', type=lambda s: [item for item in s.split("%") if item != ""],
                        default=["SL"], help="lists of cell-specific graphs to use.")
    parser.add_argument('--indep_graph', type=lambda s: [item for item in s.split("%") if item != ""],
                        default=['PPI-genetic', 'PPI-physical', 'co-exp', 'co-ess'],
                        help="lists of cell-independent graphs to use.")
    parser.add_argument('--node_feats', type=str, default="raw_omics", help="gene node features")

    parser.add_argument('--balanced', type=int, default=1,
                        help="whether the negative and positive samples are balanced")
    parser.add_argument('--pos_weight', type=float, default=50, help="weight for positive samples in loss function")
    parser.add_argument('--CCLE', type=int, default=0, help="whether or not include CCLE features into node features")
    parser.add_argument('--CCLE_dim', type=int, default=64,
                        help="dimension of embeddings for each type of CCLE omics data")
    parser.add_argument('--node2vec_feats', type=int, default=0, help="whether or not using node2vec embeddings")

    parser.add_argument('--model', type=str, default="GCN_pool", help="model type")
    parser.add_argument('--pooling', type=str, default="max", help="type of pooling operations")
    parser.add_argument('--LR', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--epochs', type=int, default=2, help="number of maximum training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="batch size")
    parser.add_argument('--out_channels', type=int, default=64, help="dimension of output channels")
    parser.add_argument('--patience', type=int, default=150, help="patience in early stopping")
    parser.add_argument('--training_percent', type=float, default=0.70,
                        help="proportion of the SL data as training set")

    parser.add_argument('--save_results', type=int, default=1, help="whether to save test results into json")
    parser.add_argument('--split_method', type=str, default="novel_gene",
                        help="how to split data into train, val and test")
    parser.add_argument('--predict_novel_genes', type=int, default=0, help="whether to predict on novel out of samples")
    parser.add_argument('--novel_cellline', type=str, default="Jurkat", help="name of novel celllines")

    args = parser.parse_args()

    return args

def train_model(model, optimizer, data, device, train_pos_edge_index, train_neg_edge_index):

    max=0
    model.train()
    optimizer.zero_grad()
    x = data.x.to(device)

    edge_index_list = []
    for edge_index in data.edge_index_list:
        edge_index = edge_index.to(device)
        edge_index_list.append(edge_index)

    edge_index_map_list = []
    for edge_index in data.edge_index_list:
        temp_edge_index=torch.transpose(edge_index, 0, 1)
        edge_index_map={}
        for edge in temp_edge_index:
            if edge[0].item() not in edge_index_map:
                edge_index_map[edge[0].item()]=set()
            edge_index_map[edge[0].item()].add(edge[1].item())
            if edge[1].item() not in edge_index_map:
                edge_index_map[edge[1].item()]=set()
            edge_index_map[edge[1].item()].add(edge[0].item())

        edge_index_map_list.append(edge_index_map)


    # shuffle training edges and labels
    all_edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=-1)
    labels = get_link_labels(train_pos_edge_index, train_neg_edge_index, device)
    num_samples = all_edge_index.shape[1]
    all_idx = list(range(num_samples))
    np.random.shuffle(all_idx)
    all_edge_index = all_edge_index[:, all_idx]
    labels = labels[all_idx]
    start = 0
    loss = 0

    while start < num_samples:
        this_batch_edge_index=all_edge_index[:, start:(start + args.batch_size)]
        this_batch_labels=labels[start:(start + args.batch_size)]
        this_batch_node_index=torch.flatten(this_batch_edge_index)
        this_batch_node_index=torch.unique(this_batch_node_index)#这个batch中用来预测的边所有的节点

        this_batch_edge_index_list=[]



        # this_batch_edge_index_and_appened_neighbor_edge_index = torch.clone(this_batch_edge_index)
        # this_batch_edge_index_and_appened_neighbor_edge_index = torch.transpose(this_batch_edge_index_and_appened_neighbor_edge_index, 0, 1)
        # appened_neighbor_edge_index=[]


        this_batch_edges_and_neighbor_edges=[]
        this_batch_nodes_and_neighbor_nodes=set(this_batch_node_index.tolist())
        for edge_index_map in edge_index_map_list:
            temp_this_batch_edge_index = set()
            for node_index in this_batch_node_index:
                if node_index.item() not in edge_index_map:
                    continue
                for neighbor_node_index in edge_index_map[node_index.item()]:
                    temp_this_batch_edge_index.add(torch.tensor([node_index.item(),neighbor_node_index]))
                    this_batch_nodes_and_neighbor_nodes.add(neighbor_node_index)
            if len(temp_this_batch_edge_index)==0:
                this_batch_edges_and_neighbor_edges.append(None)
            else:
                temp_this_batch_edge_index=torch.transpose(torch.stack(list(temp_this_batch_edge_index)), 0, 1)
                this_batch_edges_and_neighbor_edges.append(temp_this_batch_edge_index)

        this_batch_nodes_and_neighbor_nodes=sorted(this_batch_nodes_and_neighbor_nodes)
        node_dict={}
        for i,index in enumerate(this_batch_nodes_and_neighbor_nodes):
            node_dict[index]=i
        #this_batch_node_index=[node_dict[i] for i in this_batch_node_index]

        for i in range(len(this_batch_edges_and_neighbor_edges)):
            if this_batch_edges_and_neighbor_edges[i]==None:
                continue
            temp=this_batch_edges_and_neighbor_edges[i].tolist()
            temp=[[node_dict[node] for node in temp[0]],[node_dict[node] for node in temp[1]]]
            this_batch_edges_and_neighbor_edges[i]=torch.tensor(temp)

        this_batch_edge_index = this_batch_edge_index.tolist()
        this_batch_edge_index=[[node_dict[node] for node in this_batch_edge_index[0]],[node_dict[node] for node in this_batch_edge_index[1]]]

        temp_z_list = []
        for this_batch_edges_and_neighbor_edge in this_batch_edges_and_neighbor_edges:
            if this_batch_edges_and_neighbor_edge==None:
                this_batch_edges_and_neighbor_edge=torch.tensor([[1,2],[1,2]])
            temp_z = model.encode(x[this_batch_nodes_and_neighbor_nodes], this_batch_edges_and_neighbor_edge)
            temp_z_list.append(temp_z)

        z = torch.cat(temp_z_list, 1)
        # transpose is used to transform the data from (batch, # graphs, # features) into (batch, # features, # graphs)
        # the pooling operation is performed on the third dimension (graphs)
        z = z.unsqueeze(1).reshape(z.shape[0], len(edge_index_list), -1).transpose(1, 2)
        print()

    #z = torch.cat(temp_z_list, 1)
    # transpose is used to transform the data from (batch, # graphs, # features) into (batch, # features, # graphs)
    # the pooling operation is performed on the third dimension (graphs)
    #z = z.unsqueeze(1).reshape(z.shape[0], len(edge_index_list), -1).transpose(1, 2)

        if args.pooling == "max":
            z = F.max_pool2d(z, (1, len(edge_index_list))).squeeze(2)
        elif args.pooling == "mean":
            z = F.avg_pool2d(z, (1, len(edge_index_list))).squeeze(2)

        link_logits = model.decode(z, this_batch_edge_index)
        # link_probs = link_logits.sigmoid()
        link_labels = labels

        if args.balanced:
            pos_weight = torch.tensor(1)
        else:
            pos_weight = torch.tensor(args.pos_weight)

        batch_loss = F.binary_cross_entropy_with_logits(link_logits, this_batch_labels, pos_weight=pos_weight)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        start += args.batch_size

        if len(this_batch_nodes_and_neighbor_nodes) > max:
            max=len(this_batch_nodes_and_neighbor_nodes)

    print("最大长度",max)

    return float(loss)


@torch.no_grad()
def test_model(model, optimizer, data, device, pos_edge_index, neg_edge_index):
    model.eval()
    results = {}
    x = data.x.to(device)

    all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index, device)

    temp_z_list = []
    for i, edge_index in enumerate(data.edge_index_list):
        edge_index = edge_index.to(device)
        temp_z = model.encode(x, edge_index)
        temp_z_list.append(temp_z)

    z = torch.cat(temp_z_list, 1)
    z = z.unsqueeze(1).reshape(z.shape[0], len(data.edge_index_list), -1).transpose(1, 2)

    if args.pooling == "max":
        z = F.max_pool2d(z, (1, len(data.edge_index_list))).squeeze(2)
    elif args.pooling == "mean":
        z = F.avg_pool2d(z, (1, len(data.edge_index_list))).squeeze(2)

    link_logits = model.decode(z, all_edge_index)
    link_probs = link_logits.sigmoid()

    if args.balanced:
        pos_weight = torch.tensor(1)
    else:
        pos_weight = torch.tensor(args.pos_weight)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels, pos_weight=pos_weight)

    results = evaluate_performance(link_labels.cpu().numpy(), link_probs.cpu().numpy())

    return float(loss), results


@torch.no_grad()
def predict_oos(model, optimizer, data, device, pos_edge_index, neg_edge_index):
    model.eval()
    x = data.x.to(device)

    temp_z_list = []
    for i, edge_index in enumerate(data.edge_index_list):
        edge_index = edge_index.to(device)
        if args.model == 'GCN_multi' or args.model == 'GAT_multi':
            i = torch.tensor(i).to(device)
            temp_z = model.encode(x, edge_index, i)
        else:
            temp_z = model.encode(x, edge_index)
        temp_z_list.append(temp_z)

    z = torch.cat(temp_z_list, 1)
    z = z.unsqueeze(1).reshape(z.shape[0], len(data.edge_index_list), -1).transpose(1, 2)

    if args.pooling == "max":
        z = F.max_pool2d(z, (1, len(data.edge_index_list))).squeeze(2)
    elif args.pooling == "mean":
        z = F.avg_pool2d(z, (1, len(data.edge_index_list))).squeeze(2)

    # due to the huge size of the input data, split them into 100 batches
    batch_num = 100
    step_size_neg = int(neg_edge_index.shape[1] / batch_num) + 1
    link_probs = []
    for j in tqdm(range(batch_num)):
        temp_link_logits = model.decode(z, pos_edge_index,
                                        neg_edge_index[:, (j * step_size_neg):((j + 1) * step_size_neg)])
        temp_link_probs = temp_link_logits.sigmoid()
        link_probs.extend(temp_link_probs.cpu().numpy().tolist())

    return link_probs

if __name__ == "__main__":
    args = init_argparse()
    print(args)
    graph_input = args.specific_graph + args.indep_graph
    print("Number of input graphs: {}".format(len(graph_input)))
    if len(graph_input) == 0:
        print("Please specify input graph features...")
        sys.exit(0)
    # load data
    data, SL_data_train, SL_data_val, SL_data_test, SL_data_oos, gene_mapping = generate_torch_geo_data(
        args.data_source, args.CCLE, args.CCLE_dim, args.node2vec_feats,
        args.threshold, graph_input, args.node_feats, args.split_method, args.predict_novel_genes,
        args.training_percent)


    num_features = data.x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    if args.model == "GCN_pool":
        model = GCN_pool(num_features, args.out_channels, len(data.edge_index_list)).to(device)
    elif args.model == 'GCN_conv':
        model = GCN_conv(num_features, args.out_channels, len(data.edge_index_list)).to(device)
    elif args.model == 'GCN_multi':
        model = GCN_multi(num_features, args.out_channels, len(data.edge_index_list)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR)

    # generate SL torch data
    # train_pos_edge_index, train_neg_edge_index = generate_torch_edges(SL_data_train, args.balanced, True, device)
    val_pos_edge_index, val_neg_edge_index = generate_torch_edges(SL_data_val, True, False, device)
    test_pos_edge_index, test_neg_edge_index = generate_torch_edges(SL_data_test, True, False, device)
    if args.predict_novel_genes:
        oos_pos_edge_index, oos_neg_edge_index = generate_torch_edges(SL_data_oos, False, False, device)

    train_losses = []
    valid_losses = []
    # initialize the early_stopping object
    random_key = random.randint(1, 100000000)
    checkpoint_path = "checkpoint/{}.pt".format(str(random_key))
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, reverse=True, path=checkpoint_path)

    train_pos_edge_index, train_neg_edge_index = generate_torch_edges(SL_data_train, args.balanced, True, device)

    raw_data={'data':data ,
              'train_pos_edge_index':train_pos_edge_index,
              'train_neg_edge_index':train_neg_edge_index,
              'val_pos_edge_index':val_pos_edge_index,
              'val_neg_edge_index':val_neg_edge_index,
              'test_pos_edge_index':test_pos_edge_index,
              'test_neg_edge_index':test_neg_edge_index,
              'SL_data_train':SL_data_train
              }

    with open('Jurkat_gene_raw_data.pkl','wb') as f:
        pickle.dump(raw_data,f)

    for epoch in range(1, args.epochs + 1):
        # in each epoch, using different negative samples
        train_pos_edge_index, train_neg_edge_index = generate_torch_edges(SL_data_train, args.balanced, True, device)
        train_loss = train_model(model, optimizer, data, device, train_pos_edge_index, train_neg_edge_index)
        train_losses.append(train_loss)
        val_loss, results = test_model(model, optimizer, data, device, val_pos_edge_index, val_neg_edge_index)
        valid_losses.append(val_loss)
        print(
            'Epoch: {:03d}, loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}, val_loss: {:.4f}, precision@5: {:.4f}, precision@10: {:.4f}'.format(
                epoch,
                train_loss, results['AUC'], results['AUPR'], val_loss, results['precision@5'], results['precision@10']))

        # early_stopping(results['aupr'], model)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping!!!")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path))
    test_loss, results = test_model(model, optimizer, data, device, test_pos_edge_index, test_neg_edge_index)
    print("\ntest result:")
    print('AUC: {:.4f}, AP: {:.4f}, precision@5: {:.4f}, precision@10: {:.4f}'.format(results['AUC'], results['AUPR'],results['precision@5'],results['precision@10']))
    save_dict = {**vars(args), **results}

    if args.save_results:
        with open("../results/MVGCN_{}_{}_{}.json".format(args.data_source, args.split_method, str(random_key)),"w") as f:
            json.dump(save_dict, f)