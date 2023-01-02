import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

def process_att_data(users_split_arr, white, y):
    node_att = list(users_split_arr)

    stack = []

    for user in node_att:
        stack.append([user, white[user, :], y[user,:]])

    users = []
    whites = []
    ys = []

    for [user, white, y] in stack:
        users.append(user)
        whites.append(white)
        ys.append(y)

    users = torch.LongTensor(users)
    whites = torch.LongTensor(whites).view(-1)
    ys = torch.LongTensor(ys).view(-1)

    return users, whites, ys

def load_data(data_path = 'dataset/credit/'):
    data = Data()
    persons = pd.read_csv(data_path+"credit.csv", index_col=False)


    label_encoder = preprocessing.LabelEncoder()
    age = label_encoder.fit_transform(persons.Age.values).reshape(-1,1)
    data.age = F.one_hot(torch.LongTensor(age).view(-1)).cuda()

    persons['MaxBillAmountOverLast6Months'] = (persons['MaxBillAmountOverLast6Months']-persons['MaxBillAmountOverLast6Months'].mean())/persons['MaxBillAmountOverLast6Months'].std()
    persons['MaxPaymentAmountOverLast6Months'] = (persons['MaxPaymentAmountOverLast6Months']-persons['MaxPaymentAmountOverLast6Months'].mean())/persons['MaxPaymentAmountOverLast6Months'].std()
    persons['MostRecentBillAmount'] = (persons['MostRecentBillAmount']-persons['MostRecentBillAmount'].mean())/persons['MostRecentBillAmount'].std()
    persons['MostRecentPaymentAmount'] = (persons['MostRecentPaymentAmount']-persons['MostRecentPaymentAmount'].mean())/persons['MostRecentPaymentAmount'].std()
    persons['TotalMonthsOverdue'] = (persons['TotalMonthsOverdue']-persons['TotalMonthsOverdue'].mean())/persons['TotalMonthsOverdue'].std()

    labels = label_encoder.fit_transform(persons.NoDefaultNextMonth.values).reshape(-1,1)
    data.y = torch.LongTensor(labels).cuda()
    persons_x = persons.drop(columns=['NoDefaultNextMonth'])
    data.x = torch.LongTensor(persons_x.values).cuda()
    edges_np = np.genfromtxt(data_path+"credit_edges.txt").astype('int')
    edges_df = pd.DataFrame(edges_np)
    edges_df.columns = ['person_1','person_2']

    # split training set, validation set, testing set
    shuffled_edges = edges_df.sample(frac=1).reset_index(drop=True)
    val_ratio_task = 0.1
    test_ratio_task = 0.1
    train_cutoff_row = int(np.round(len(shuffled_edges) * (1 - val_ratio_task - test_ratio_task)))
    val_cutoff_row = int(np.round(len(shuffled_edges) * (1 - test_ratio_task)))
    train_edges = shuffled_edges[:train_cutoff_row]
    val_edges = shuffled_edges[train_cutoff_row:val_cutoff_row]
    test_edges = shuffled_edges[val_cutoff_row:]
    train_edges.to_csv(data_path+"train_edges.csv")
    val_edges.to_csv(data_path+"val_edges.csv")
    test_edges.to_csv(data_path+"test_edges.csv")

    num_users = len(persons)
    person_id = [i for i in range(0, num_users)]
    person_idx = np.asarray(person_id)
    val_ratio_att = 0.1
    test_ratio_att = 0.1
    train_ratio_att = 0.8
    train_cutoff_row = int(np.round(len(person_idx) * (1 - val_ratio_att - test_ratio_att)))
    val_cutoff_row = int(np.round(len(person_idx) * (1 - test_ratio_att)))
    persons_train = person_idx[:train_cutoff_row]
    persons_val = person_idx[train_cutoff_row:val_cutoff_row]
    persons_test = person_idx[val_cutoff_row:]
    np.save(data_path+"persons_train_att.npy", persons_train)
    np.save(data_path+"persons_val_att.npy", persons_val)
    np.save(data_path+"persons_test_att.npy", persons_test)

    for split in ['train','val','test']:
      edges = pd.read_csv(data_path+f"{split}_edges.csv",index_col=0)
      data[f'{split}_pos_edges'] = torch.LongTensor(edges.values).cuda()
      edges_all = np.hstack((np.stack([edges_df['person_1'].values,edges_df['person_2'].values]),
                            np.stack([edges_df['person_2'].values,edges_df['person_1'].values])))
      data.edge_index = torch.LongTensor(edges_all).cuda()
      edges_ = np.hstack((np.stack([edges['person_1'].values,edges['person_2'].values]),
                        np.stack([edges['person_2'].values,edges['person_1'].values])))
      data[f'{split}_pos_edge_index'] = torch.LongTensor(edges_).cuda()
      persons_split_att = np.load(data_path+f"persons_{split}_att.npy")
      data[f'{split}_id_feat'] = process_att_data(persons_split_att, age,labels)
    return data

def load_data_cf(data_path = 'dataset/credit/'):
    data = Data()
    persons = pd.read_csv(data_path+"credit.csv", index_col=False)

    persons['Age'] = pd.to_numeric(persons['Age'])
    persons['Age'] = 1 - persons['Age']
    label_encoder = preprocessing.LabelEncoder()
    age = label_encoder.fit_transform(persons.Age.values).reshape(-1,1)
    data.age = F.one_hot(torch.LongTensor(age).view(-1)).cuda()

    persons['MaxBillAmountOverLast6Months'] = (persons['MaxBillAmountOverLast6Months']-persons['MaxBillAmountOverLast6Months'].mean())/persons['MaxBillAmountOverLast6Months'].std()
    persons['MaxPaymentAmountOverLast6Months'] = (persons['MaxPaymentAmountOverLast6Months']-persons['MaxPaymentAmountOverLast6Months'].mean())/persons['MaxPaymentAmountOverLast6Months'].std()
    persons['MostRecentBillAmount'] = (persons['MostRecentBillAmount']-persons['MostRecentBillAmount'].mean())/persons['MostRecentBillAmount'].std()
    persons['MostRecentPaymentAmount'] = (persons['MostRecentPaymentAmount']-persons['MostRecentPaymentAmount'].mean())/persons['MostRecentPaymentAmount'].std()
    persons['TotalMonthsOverdue'] = (persons['TotalMonthsOverdue']-persons['TotalMonthsOverdue'].mean())/persons['TotalMonthsOverdue'].std()

    labels = label_encoder.fit_transform(persons.NoDefaultNextMonth.values).reshape(-1,1)
    data.y = torch.LongTensor(labels).cuda()
    persons_x = persons.drop(columns=['NoDefaultNextMonth'])
    data.x = torch.LongTensor(persons_x.values).cuda()
    edges_np = np.genfromtxt(data_path+"credit_edges.txt").astype('int')
    edges_df = pd.DataFrame(edges_np)
    edges_df.columns = ['person_1','person_2']

    # split training set, validation set, testing set
    shuffled_edges = edges_df.sample(frac=1).reset_index(drop=True)
    val_ratio_task = 0.1
    test_ratio_task = 0.1
    train_cutoff_row = int(np.round(len(shuffled_edges) * (1 - val_ratio_task - test_ratio_task)))
    val_cutoff_row = int(np.round(len(shuffled_edges) * (1 - test_ratio_task)))
    train_edges = shuffled_edges[:train_cutoff_row]
    val_edges = shuffled_edges[train_cutoff_row:val_cutoff_row]
    test_edges = shuffled_edges[val_cutoff_row:]
    train_edges.to_csv(data_path+"train_edges_cf.csv")
    val_edges.to_csv(data_path+"val_edges_cf.csv")
    test_edges.to_csv(data_path+"test_edges_cf.csv")

    num_users = len(persons)
    person_id = [i for i in range(0, num_users)]
    person_idx = np.asarray(person_id)
    val_ratio_att = 0.1
    test_ratio_att = 0.1
    train_ratio_att = 0.8
    train_cutoff_row = int(np.round(len(person_idx) * (1 - val_ratio_att - test_ratio_att)))
    val_cutoff_row = int(np.round(len(person_idx) * (1 - test_ratio_att)))
    persons_train = person_idx[:train_cutoff_row]
    persons_val = person_idx[train_cutoff_row:val_cutoff_row]
    persons_test = person_idx[val_cutoff_row:]
    np.save(data_path+"persons_train_att_cf.npy", persons_train)
    np.save(data_path+"persons_val_att_cf.npy", persons_val)
    np.save(data_path+"persons_test_att_cf.npy", persons_test)

    for split in ['train','val','test']:
      edges = pd.read_csv(data_path+f"{split}_edges_cf.csv",index_col=0)
      data[f'{split}_pos_edges'] = torch.LongTensor(edges.values).cuda()
      edges_all = np.hstack((np.stack([edges_df['person_1'].values,edges_df['person_2'].values]),
                            np.stack([edges_df['person_2'].values,edges_df['person_1'].values])))
      data.edge_index = torch.LongTensor(edges_all).cuda()
      edges_ = np.hstack((np.stack([edges['person_1'].values,edges['person_2'].values]),
                        np.stack([edges['person_2'].values,edges['person_1'].values])))
      data[f'{split}_pos_edge_index'] = torch.LongTensor(edges_).cuda()
      persons_split_att = np.load(data_path+f"persons_{split}_att_cf.npy")
      data[f'{split}_id_feat'] = process_att_data(persons_split_att, age,labels)
    return data

def load_data_rb(data_path = 'dataset/credit/'):
    data = Data()
    persons = pd.read_csv(data_path+"credit.csv", index_col=False)


    label_encoder = preprocessing.LabelEncoder()
    age = label_encoder.fit_transform(persons.Age.values).reshape(-1,1)
    data.age = F.one_hot(torch.LongTensor(age).view(-1)).cuda()

    persons['MaxBillAmountOverLast6Months'] = (persons['MaxBillAmountOverLast6Months']-persons['MaxBillAmountOverLast6Months'].mean())/persons['MaxBillAmountOverLast6Months'].std()
    persons['MaxPaymentAmountOverLast6Months'] = (persons['MaxPaymentAmountOverLast6Months']-persons['MaxPaymentAmountOverLast6Months'].mean())/persons['MaxPaymentAmountOverLast6Months'].std()
    persons['MostRecentBillAmount'] = (persons['MostRecentBillAmount']-persons['MostRecentBillAmount'].mean())/persons['MostRecentBillAmount'].std()
    persons['MostRecentPaymentAmount'] = (persons['MostRecentPaymentAmount']-persons['MostRecentPaymentAmount'].mean())/persons['MostRecentPaymentAmount'].std()
    persons['TotalMonthsOverdue'] = (persons['TotalMonthsOverdue']-persons['TotalMonthsOverdue'].mean())/persons['TotalMonthsOverdue'].std()

    labels = label_encoder.fit_transform(persons.NoDefaultNextMonth.values).reshape(-1,1)
    data.y = torch.LongTensor(labels).cuda()
    persons_x = persons.drop(columns=['NoDefaultNextMonth'])
    data.x = torch.LongTensor(persons_x.values).cuda()
    data.x = data.x + torch.ones(data.x.shape).normal_(0,0.5).cuda()
    edges_np = np.genfromtxt(data_path+"credit_edges.txt").astype('int')
    edges_df = pd.DataFrame(edges_np)
    edges_df.columns = ['person_1','person_2']

    # split training set, validation set, testing set
    shuffled_edges = edges_df.sample(frac=1).reset_index(drop=True)
    val_ratio_task = 0.1
    test_ratio_task = 0.1
    train_cutoff_row = int(np.round(len(shuffled_edges) * (1 - val_ratio_task - test_ratio_task)))
    val_cutoff_row = int(np.round(len(shuffled_edges) * (1 - test_ratio_task)))
    train_edges = shuffled_edges[:train_cutoff_row]
    val_edges = shuffled_edges[train_cutoff_row:val_cutoff_row]
    test_edges = shuffled_edges[val_cutoff_row:]
    train_edges.to_csv(data_path+"train_edges_rb.csv")
    val_edges.to_csv(data_path+"val_edges_rb.csv")
    test_edges.to_csv(data_path+"test_edges_rb.csv")

    num_users = len(persons)
    person_id = [i for i in range(0, num_users)]
    person_idx = np.asarray(person_id)
    val_ratio_att = 0.1
    test_ratio_att = 0.1
    train_ratio_att = 0.8
    train_cutoff_row = int(np.round(len(person_idx) * (1 - val_ratio_att - test_ratio_att)))
    val_cutoff_row = int(np.round(len(person_idx) * (1 - test_ratio_att)))
    persons_train = person_idx[:train_cutoff_row]
    persons_val = person_idx[train_cutoff_row:val_cutoff_row]
    persons_test = person_idx[val_cutoff_row:]
    np.save(data_path+"persons_train_att_rb.npy", persons_train)
    np.save(data_path+"persons_val_att_rb.npy", persons_val)
    np.save(data_path+"persons_test_att_rb.npy", persons_test)

    for split in ['train','val','test']:
      edges = pd.read_csv(data_path+f"{split}_edges_rb.csv",index_col=0)
      data[f'{split}_pos_edges'] = torch.LongTensor(edges.values).cuda()
      edges_all = np.hstack((np.stack([edges_df['person_1'].values,edges_df['person_2'].values]),
                            np.stack([edges_df['person_2'].values,edges_df['person_1'].values])))
      data.edge_index = torch.LongTensor(edges_all).cuda()
      edges_ = np.hstack((np.stack([edges['person_1'].values,edges['person_2'].values]),
                        np.stack([edges['person_2'].values,edges['person_1'].values])))
      data[f'{split}_pos_edge_index'] = torch.LongTensor(edges_).cuda()
      persons_split_att = np.load(data_path+f"persons_{split}_att_rb.npy")
      data[f'{split}_id_feat'] = process_att_data(persons_split_att, age,labels)
    return data