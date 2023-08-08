
import argparse, sys

import torch
from tqdm import tqdm
from utils import *
from model import *
from collections import Counter
import pickle


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

    parser.add_argument('--model', type=str, default="GCN_transformer_pool", help="model type")
    parser.add_argument('--pooling', type=str, default="max", help="type of pooling operations")
    parser.add_argument('--LR', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--epochs', type=int, default=1, help="number of maximum training epochs")
    parser.add_argument('--batch_size', type=int, default=50, help="batch size")
    parser.add_argument('--out_channels', type=int, default=64, help="dimension of output channels")
    parser.add_argument('--patience', type=int, default=5, help="patience in early stopping")
    parser.add_argument('--training_percent', type=float, default=0.70,
                        help="proportion of the SL data as training set")
    parser.add_argument('--save_results', type=int, default=1, help="whether to save test results into json")
    parser.add_argument('--split_method', type=str, default="novel_pair",
                        help="how to split data into train, val and test")
    parser.add_argument('--predict_novel_genes', type=int, default=0, help="whether to predict on novel out of samples")
    parser.add_argument('--novel_cellline', type=str, default="K562", help="name of novel celllines")
    parser.add_argument('--src_len', default=512, type=int, help='length of source')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help='length of source')
    parser.add_argument('--tgt_len', default=512, type=str, help='length of target')
    parser.add_argument('--d_model', default=512, type=int, help='Embedding Size')
    parser.add_argument('--d_ff', default=2048, type=int, help='FeedForward dimension')
    parser.add_argument('--d_k', default=64, type=int, help='dimension of K(=Q), V')
    parser.add_argument('--n_layers', default=2, type=int, help='number of Encoder of Decoder Layer')
    parser.add_argument('--n_heads', default=4, type=int, help='number of heads in Multi-Head Attention')

    args = parser.parse_args()

    return args


def train_model(model, optimizer, raw_data,random_walk_list, device, train_pos_edge_index, train_neg_edge_index):
    model.train()
    optimizer.zero_grad()
    x = raw_data['data'].x.to(device)
    edge_index_list = []
    for edge_index in raw_data['data'].edge_index_list:
        edge_index = edge_index.to(device)
        edge_index_list.append(edge_index)

    edge_index_list = [edge_index_list[0]]

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
    while start <= num_samples-args.batch_size:
        temp_z_list = []
        for edge_index in edge_index_list:
            temp_z = model.encode(x, edge_index)
            #temp_z=x
            temp_z_list.append(temp_z)

        z = torch.cat(temp_z_list, 1)

        this_batch_edge_index=all_edge_index[:, start:(start + args.batch_size)]
        this_batch_node_index=torch.cat((this_batch_edge_index[0], this_batch_edge_index[1])).unique()


        neighbor_nodes_index=[random_walk_list[node_index.item()] for node_index in this_batch_node_index]

        transformer_input_list=[]
        for temp1 in neighbor_nodes_index:
            for temp2 in temp1:
                for temp3 in random.choice(temp2):
                    transformer_input_list.append(temp3)

        counter = Counter(transformer_input_list)
        # 按频率从大到小排序
        sorted_items = counter.most_common()

        transformer_input_list = [temp[0] for temp in sorted_items[0:500]]

        #随机取数，记得屏蔽掉
        #transformer_input_list = random.sample(range(len(x)),500)

        transformer_input_list = []




        transformer_input_list=list(set(transformer_input_list)-set(this_batch_node_index.tolist()))
        transformer_input_list=this_batch_node_index.tolist()+transformer_input_list

        this_batch_node_index_map={}#用来预测的边的位置进行标记
        for i,node_index in enumerate(this_batch_node_index):
            this_batch_node_index_map[node_index.item()]=i



        transformer_output=model.modified_transformer(z[transformer_input_list])
        #transformer_output = z[transformer_input_list]


        decoder_input=torch.cat((transformer_output[[this_batch_node_index_map[i.item()] for i in this_batch_edge_index[0]]],transformer_output[[this_batch_node_index_map[i.item()] for i in this_batch_edge_index[1]]]),dim=1)
        #this_batch_edge=z[]
        # transpose is used to transform the data from (batch, # graphs, # features) into (batch, # features, # graphs)
        # the pooling operation is performed on the third dimension (graphs)
        #z = z.unsqueeze(1).reshape(z.shape[0], len(edge_index_list), -1).transpose(1, 2)

        link_logits = model.decode(decoder_input)
        # link_probs = link_logits.sigmoid()
        link_labels = labels[start:(start + args.batch_size)]

        if args.balanced:
            pos_weight = torch.tensor(1)
        else:
            pos_weight = torch.tensor(args.pos_weight)

        batch_loss = F.binary_cross_entropy_with_logits(link_logits, link_labels, pos_weight=pos_weight)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        start += args.batch_size

    return float(loss)


@torch.no_grad()
def test_model(model, optimizer, raw_data, device, pos_edge_index, neg_edge_index):
    model.eval()

    x = raw_data['data'].x.to(device)
    edge_index_list = []
    for edge_index in raw_data['data'].edge_index_list:
        edge_index = edge_index.to(device)
        edge_index_list.append(edge_index)

    # shuffle training edges and labels
    all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    labels = get_link_labels(pos_edge_index, neg_edge_index, device)
    num_samples = all_edge_index.shape[1]
    all_idx = list(range(num_samples))
    np.random.shuffle(all_idx)
    all_edge_index = all_edge_index[:, all_idx]
    labels = labels[all_idx]

    start = 0
    loss = 0

    all_link_probs=None
    all_link_labels=None

    while start < num_samples:
        temp_z_list = []
        for edge_index in edge_index_list:
            temp_z = model.encode(x, edge_index)
            temp_z_list.append(temp_z)

        edge_index_list = [edge_index_list[0]]

        z = torch.cat(temp_z_list, 1)

        this_batch_edge_index = all_edge_index[:, start:(start + args.batch_size)]
        this_batch_node_index = torch.cat((this_batch_edge_index[0], this_batch_edge_index[1])).unique()

        neighbor_nodes_index = [random_walk_list[node_index.item()] for node_index in this_batch_node_index]

        transformer_input_list = []
        for temp1 in neighbor_nodes_index:
            for temp2 in temp1:
                for temp3 in random.choice(temp2):
                    transformer_input_list.append(temp3)

        counter = Counter(transformer_input_list)
        # 按频率从大到小排序
        sorted_items = counter.most_common()
        transformer_input_list = [temp[0] for temp in sorted_items[0:500]]

        # 随机取数，记得屏蔽掉
        #transformer_input_list = random.sample(range(len(x)), 500)

        transformer_input_list = []


        transformer_input_list = list(set(transformer_input_list) - set(this_batch_node_index.tolist()))

        transformer_input_list = this_batch_node_index.tolist() + transformer_input_list

        this_batch_node_index_map = {}  # 用来预测的边的位置进行标记
        for i, node_index in enumerate(this_batch_node_index):
            this_batch_node_index_map[node_index.item()] = i

        transformer_output = model.modified_transformer(z[transformer_input_list])

        temp = transformer_output[[this_batch_node_index_map[i.item()] for i in this_batch_edge_index[0]]]

        decoder_input = torch.cat((transformer_output[
                                       [this_batch_node_index_map[i.item()] for i in this_batch_edge_index[0]]],
                                   transformer_output[
                                       [this_batch_node_index_map[i.item()] for i in this_batch_edge_index[1]]]), dim=1)
        # this_batch_edge=z[]/
        # transpose is used to transform the data from (batch, # graphs, # features) into (batch, # features, # graphs)
        # the pooling operation is performed on the third dimension (graphs)
        # z = z.unsqueeze(1).reshape(z.shape[0], len(edge_index_list), -1).transpose(1, 2)

        link_logits = model.decode(decoder_input)
        link_probs = link_logits.sigmoid()
        link_labels = labels[start:(start + args.batch_size)]

        if args.balanced:
            pos_weight = torch.tensor(1)
        else:
            pos_weight = torch.tensor(args.pos_weight)

        batch_loss = F.binary_cross_entropy_with_logits(link_logits, link_labels, pos_weight=pos_weight)

        loss += batch_loss.item()
        start += args.batch_size


        all_link_probs=link_probs if all_link_probs ==None else torch.cat([all_link_probs,link_probs])
        all_link_labels=link_labels if all_link_labels == None else torch.cat([all_link_labels,link_labels])

    results = evaluate_performance(all_link_labels.cpu().numpy(), all_link_probs.cpu().numpy())

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
    with open(args.data_source+'_raw_data.pkl', 'rb') as f:
        raw_data = pickle.load(f)

    with open(args.data_source+'_random_walk_list.pkl', 'rb') as f:
        random_walk_list = pickle.load(f)

    num_features = raw_data['data'].x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # nvidia-smi

    # load model
    if args.model == "GCN_pool":
        model = GCN_pool(num_features, args.out_channels, len(raw_data['data'].edge_index_list)).to(device)
    elif args.model == 'GCN_conv':
        model = GCN_conv(num_features, args.out_channels, len(raw_data['data'].edge_index_list)).to(device)
    elif args.model == 'GCN_multi':
        model = GCN_multi(num_features, args.out_channels, len(raw_data['data'].edge_index_list)).to(device)
    elif args.model == 'Transformers_model':
        model = Transformers_model(args).to(device)
    elif args.model == 'GCN_transformer_pool':
        model = GCN_transformer_pool(num_features, args.out_channels, len(raw_data['data'].edge_index_list),args).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR)

    train_losses = []
    valid_losses = []
    # initialize the early_stopping object
    random_key = random.randint(1, 100000000)
    checkpoint_path = "checkpoint/{}.pt".format(str(random_key))
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, reverse=True, path=checkpoint_path)

    random.seed(123)
    np.random.seed(123)

    for epoch in range(1, args.epochs + 1):
        # in each epoch, using different negative samples
        train_pos_edge_index, train_neg_edge_index = generate_torch_edges(raw_data['SL_data_train'], args.balanced, True, device)
        train_loss = train_model(model, optimizer, raw_data,random_walk_list, device, train_pos_edge_index, train_neg_edge_index)
        train_losses.append(train_loss)
        val_loss, results = test_model(model, optimizer, raw_data, device, raw_data['val_pos_edge_index'], raw_data['val_neg_edge_index'])
        valid_losses.append(val_loss)
        print(
            'Epoch: {:03d}, loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}, val_loss: {:.4f}, precision@5: {:.4f}, precision@10: {:.4f}'.format(
                epoch,
                train_loss,results['acc'],results['F1_score'], results['AUC'], results['AUPR'], val_loss, results['precision@5'], results['precision@10']))

        # early_stopping(results['aupr'], model)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping!!!")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path))

    test_loss, results = test_model(model, optimizer, raw_data, device, raw_data['test_pos_edge_index'], raw_data['test_neg_edge_index'])
    print("\ntest result:")
    print('AUC: {:.4f}, AP: {:.4f}, precision@5: {:.4f}, precision@10: {:.4f}'.format(results['AUC'], results['AUPR'],
                                                                                      results['precision@5'],
                                                                                      results['precision@10']))

    save_dict = {**vars(args), **results}

    if args.save_results:
        with open("../results/MVGCN_{}_{}_{}.json".format(args.data_source, args.split_method, str(random_key)),
                  "w") as f:
            json.dump(save_dict, f)