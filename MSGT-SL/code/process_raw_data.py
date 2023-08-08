import pickle
import  torch
import numpy as np
import random

def build_random_walk_list():
    with open('Jurkat_gene_raw_data.pkl','rb') as f:
        raw_data=pickle.load(f)

    edge_index_map_list = []  # 将五种类型的边构建成字典的形式，否则挑选边的时间复杂度会过高
    for edge_index in raw_data['data'].edge_index_list:
        temp_edge_index = torch.transpose(edge_index, 0, 1)
        edge_index_map = {}
        for edge in temp_edge_index:
            if edge[0].item() not in edge_index_map:
                edge_index_map[edge[0].item()] = set()
            edge_index_map[edge[0].item()].add(edge[1].item())
            if edge[1].item() not in edge_index_map:
                edge_index_map[edge[1].item()] = set()
            edge_index_map[edge[1].item()].add(edge[0].item())

        edge_index_map_list.append(edge_index_map)




    random_walk_list=[]
    for index in range(len(raw_data['data'].x)):
        print(index)
        every_dict_random_walk_list=[]
        for dict_index, dict in enumerate(edge_index_map_list):
            if dict_index==0:
                continue

            paths = []
            for i in range(10):
                path=[index]
                every_path_word_set=set(path)
                temp=index
                for i in range(9):
                    if temp not in dict or set(path).issubset(dict[temp]):
                        break
                    random_number = random.choice(list(dict[temp]))
                    if random_number in set(path):
                        continue
                    path.append(random_number)
                    temp=random_number

                paths.append(path)
            every_dict_random_walk_list.append(paths)
        random_walk_list.append(every_dict_random_walk_list)

        with open('Jurkat_gene_random_walk_list.pkl','wb') as f:
            pickle.dump(random_walk_list,f)




    print()


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def process_node_and_edge(data, pos_edge_index, neg_edge_index):
    x = data.x   #保存了图中所有的点
    max=0

    edge_index_list = []  #用来保存五种类型的所有边
    for edge_index in data.edge_index_list:
        edge_index = edge_index
        edge_index_list.append(edge_index)

    edge_index_map_list = []  #将五种类型的边构建成字典的形式，否则挑选边的时间复杂度会过高
    for edge_index in data.edge_index_list:
        temp_edge_index = torch.transpose(edge_index, 0, 1)
        edge_index_map = {}
        for edge in temp_edge_index:
            if edge[0].item() not in edge_index_map:
                edge_index_map[edge[0].item()] = set()
            edge_index_map[edge[0].item()].add(edge[1].item())
            if edge[1].item() not in edge_index_map:
                edge_index_map[edge[1].item()] = set()
            edge_index_map[edge[1].item()].add(edge[0].item())

        edge_index_map_list.append(edge_index_map)


    # shuffle training edges and labels
    all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    labels = get_link_labels(pos_edge_index, neg_edge_index)
    num_samples = all_edge_index.shape[1]
    all_idx = list(range(num_samples))
    np.random.shuffle(all_idx)
    all_edge_index = all_edge_index[:, all_idx]
    labels = labels[all_idx]
    start = 0
    loss = 0

    #按照用来测试的edges,将这些数据处理成一个一个的子网数据
    dataset=[]
    batch_size=4 #将用来测试的edges，每四个作为一组,这个地方可以来修改
    while start <= num_samples-batch_size:
        this_batch_edge_index = all_edge_index[:, start:(start + batch_size)]#用来测试的边
        this_batch_labels = labels[start:(start + batch_size)]#标签
        this_batch_node_index = torch.flatten(this_batch_edge_index)# 这个batch中用来预测的边所有的节点
        start += batch_size

        this_batch_edges_and_neighbor_edges = []#用来储存这个batch关系到的所有的边（分别存储五种类型）
        this_batch_nodes_and_neighbor_nodes = set(this_batch_node_index.tolist())#用来储存这个batch关系到的所有的点
        for i,edge_index_map in enumerate(edge_index_map_list):
            temp_this_batch_edge_index = set()
            for node_index in this_batch_node_index:
                if node_index.item() not in edge_index_map:
                    continue
                for neighbor_node_index in edge_index_map[node_index.item()]:
                    if (i==2 or i==3)  and np.random.randint(0,15)>0:
                        continue
                    temp_this_batch_edge_index.add(torch.tensor([node_index.item(), neighbor_node_index]))
                    this_batch_nodes_and_neighbor_nodes.add(neighbor_node_index)
            if len(temp_this_batch_edge_index) == 0:
                this_batch_edges_and_neighbor_edges.append(None)
            else:
                temp_this_batch_edge_index = torch.transpose(torch.stack(list(temp_this_batch_edge_index)), 0, 1)
                this_batch_edges_and_neighbor_edges.append(temp_this_batch_edge_index)
        this_batch_nodes_and_neighbor_nodes = sorted(this_batch_nodes_and_neighbor_nodes)#对所有的点进行去重操作

        max=max if max > len(this_batch_nodes_and_neighbor_nodes) else len(this_batch_nodes_and_neighbor_nodes)
        node_dict = {}#记住x中对应序列的点在序列中对应的位置
        for i, index in enumerate(this_batch_nodes_and_neighbor_nodes):
            node_dict[index] = i

        for i in range(len(this_batch_edges_and_neighbor_edges)):#将边在x对应的下标和输入序列中的位置进行对齐，这个地方的边对应输入序列中所有的边
            if this_batch_edges_and_neighbor_edges[i] == None:
                continue
            temp = this_batch_edges_and_neighbor_edges[i].tolist()
            temp = [[node_dict[node] for node in temp[0]], [node_dict[node] for node in temp[1]]]
            this_batch_edges_and_neighbor_edges[i] = torch.tensor(temp)



        this_batch_edge_index = this_batch_edge_index.tolist()##将边在x对应的下标和输入序列中的位置进行对齐，这个地方的边对应用来测试的边
        this_batch_edge_index = [[node_dict[node] for node in this_batch_edge_index[0]], [node_dict[node] for node in this_batch_edge_index[1]]]

        data={'labels': this_batch_labels,
              'edge_index':this_batch_edge_index,
              'nodes_and_neighbor_nodes':this_batch_nodes_and_neighbor_nodes,
              'edges_and_neighbor_edge' :this_batch_edges_and_neighbor_edges,
              }
        dataset.append(data)#封装成为一个数据集

        print('已经处理的batch_size的数量：',len(dataset))
    return dataset

def process_raw_data():
    with open('Jurkat_gene_raw_data.pkl','rb') as f:
        raw_data=pickle.load(f)

    train_inputs = process_node_and_edge(raw_data['data'], raw_data['train_pos_edge_index'], raw_data['train_neg_edge_index'])
    test_inputs=process_node_and_edge(raw_data['data'], raw_data['test_pos_edge_index'], raw_data['test_neg_edge_index'])
    val_inputs=process_node_and_edge(raw_data['data'], raw_data['val_pos_edge_index'], raw_data['val_neg_edge_index'])
    datasets=[train_inputs,test_inputs,val_inputs]
    with open('Jurkat_gene_datasets.pkl','wb') as f:
        pickle.dump(datasets,f)

if __name__=='__main__':
    process_raw_data()
    build_random_walk_list()


