import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import random

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

def load_data(data_path = 'dataset/income/'):
    data = Data()
    persons = pd.read_csv(data_path+"income.csv", index_col=False)

    persons['income'] = pd.to_numeric(persons['income'])
    persons['race'] = pd.to_numeric(persons['race'])
    persons['age'] = 2*(persons['age']-persons['age'].min()).div(persons['age'].max() - persons['age'].min()) - 1
    persons['workclass'] = 2*(persons['workclass']-persons['workclass'].min()).div(persons['workclass'].max() - persons['workclass'].min()) - 1
    persons['fnlwgt'] = 2*(persons['fnlwgt']-persons['fnlwgt'].min()).div(persons['fnlwgt'].max() - persons['fnlwgt'].min()) - 1
    persons['education'] = 2*(persons['education']-persons['education'].min()).div(persons['education'].max() - persons['education'].min()) - 1
    persons['education-num'] = 2*(persons['education-num']-persons['education-num'].min()).div(persons['education-num'].max() - persons['education-num'].min()) - 1
    persons['marital-status'] = 2*(persons['marital-status']-persons['marital-status'].min()).div(persons['marital-status'].max() - persons['marital-status'].min()) - 1
    persons['occupation'] = 2*(persons['occupation']-persons['occupation'].min()).div(persons['occupation'].max() - persons['occupation'].min()) - 1
    persons['relationship'] = 2*(persons['relationship']-persons['relationship'].min()).div(persons['relationship'].max() - persons['relationship'].min()) - 1
    persons['capital-gain'][persons['capital-gain'] != 0] = 1
    persons['capital-loss'][persons['capital-loss'] != 0] = 1
    persons['hours-per-week'] = 2*(persons['hours-per-week']-persons['hours-per-week'].min()).div(persons['hours-per-week'].max() - persons['hours-per-week'].min()) - 1
    persons['native-country'] = 2*(persons['native-country']-persons['native-country'].min()).div(persons['native-country'].max() - persons['native-country'].min()) - 1

    label_encoder = preprocessing.LabelEncoder()
    race = label_encoder.fit_transform(persons.race.values).reshape(-1,1)
    data.race = F.one_hot(torch.LongTensor(race).view(-1)).cuda()

    labels = label_encoder.fit_transform(persons.income.values).reshape(-1,1)
    data.y = torch.LongTensor(labels).cuda()
    persons_x = persons.drop(columns=['income'])
    data.x = torch.LongTensor(persons_x.values).cuda()

    edges_np = np.genfromtxt(data_path+"income_edges.txt").astype('int')
    edges_df = pd.DataFrame(edges_np)
    edges_df.columns = ['person_1','person_2']

    edges_all = np.hstack((np.stack([edges_df['person_1'].values,edges_df['person_2'].values]),
                                np.stack([edges_df['person_2'].values,edges_df['person_1'].values])))
    data.edge_index = torch.LongTensor(edges_all).cuda()

    num_users = len(persons)
    person_id = [i for i in range(0, num_users)]
    person_idx = np.asarray(person_id)
    random.seed(20)
    random.shuffle(person_idx)
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
      persons_split_att = np.load(data_path+f"persons_{split}_att.npy")
      data[f'{split}_id_feat'] = process_att_data(persons_split_att, race,labels)
    return data

def load_data_cf(data_path = 'dataset/income/'):
    data = Data()
    persons = pd.read_csv(data_path+"income.csv", index_col=False)

    persons['income'] = pd.to_numeric(persons['income'])
    persons['race'] = pd.to_numeric(persons['race'])
    persons['age'] = 2*(persons['age']-persons['age'].min()).div(persons['age'].max() - persons['age'].min()) - 1
    persons['workclass'] = 2*(persons['workclass']-persons['workclass'].min()).div(persons['workclass'].max() - persons['workclass'].min()) - 1
    persons['fnlwgt'] = 2*(persons['fnlwgt']-persons['fnlwgt'].min()).div(persons['fnlwgt'].max() - persons['fnlwgt'].min()) - 1
    persons['education'] = 2*(persons['education']-persons['education'].min()).div(persons['education'].max() - persons['education'].min()) - 1
    persons['education-num'] = 2*(persons['education-num']-persons['education-num'].min()).div(persons['education-num'].max() - persons['education-num'].min()) - 1
    persons['marital-status'] = 2*(persons['marital-status']-persons['marital-status'].min()).div(persons['marital-status'].max() - persons['marital-status'].min()) - 1
    persons['occupation'] = 2*(persons['occupation']-persons['occupation'].min()).div(persons['occupation'].max() - persons['occupation'].min()) - 1
    persons['relationship'] = 2*(persons['relationship']-persons['relationship'].min()).div(persons['relationship'].max() - persons['relationship'].min()) - 1
    persons['capital-gain'][persons['capital-gain'] != 0] = 1
    persons['capital-loss'][persons['capital-loss'] != 0] = 1
    persons['hours-per-week'] = 2*(persons['hours-per-week']-persons['hours-per-week'].min()).div(persons['hours-per-week'].max() - persons['hours-per-week'].min()) - 1
    persons['native-country'] = 2*(persons['native-country']-persons['native-country'].min()).div(persons['native-country'].max() - persons['native-country'].min()) - 1

    label_encoder = preprocessing.LabelEncoder()
    race = label_encoder.fit_transform(persons.race.values).reshape(-1,1)
    data.race = F.one_hot(torch.LongTensor(race).view(-1)).cuda()

    labels = label_encoder.fit_transform(persons.income.values).reshape(-1,1)
    data.y = torch.LongTensor(labels).cuda()
    persons_x = persons.drop(columns=['income'])
    data.x = torch.LongTensor(persons_x.values).cuda()

    edges_np = np.genfromtxt(data_path+"income_edges.txt").astype('int')
    edges_df = pd.DataFrame(edges_np)
    edges_df.columns = ['person_1','person_2']

    edges_all = np.hstack((np.stack([edges_df['person_1'].values,edges_df['person_2'].values]),
                                np.stack([edges_df['person_2'].values,edges_df['person_1'].values])))
    data.edge_index = torch.LongTensor(edges_all).cuda()

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
      persons_split_att = np.load(data_path+f"persons_{split}_att_cf.npy")
      data[f'{split}_id_feat'] = process_att_data(persons_split_att, race,labels)
    return data

def load_data_rb(data_path = 'dataset/income/'):
    data = Data()
    persons = pd.read_csv(data_path+"income.csv", index_col=False)

    persons['income'] = pd.to_numeric(persons['income'])
    persons['race'] = pd.to_numeric(persons['race'])
    persons['age'] = 2*(persons['age']-persons['age'].min()).div(persons['age'].max() - persons['age'].min()) - 1
    persons['workclass'] = 2*(persons['workclass']-persons['workclass'].min()).div(persons['workclass'].max() - persons['workclass'].min()) - 1
    persons['fnlwgt'] = 2*(persons['fnlwgt']-persons['fnlwgt'].min()).div(persons['fnlwgt'].max() - persons['fnlwgt'].min()) - 1
    persons['education'] = 2*(persons['education']-persons['education'].min()).div(persons['education'].max() - persons['education'].min()) - 1
    persons['education-num'] = 2*(persons['education-num']-persons['education-num'].min()).div(persons['education-num'].max() - persons['education-num'].min()) - 1
    persons['marital-status'] = 2*(persons['marital-status']-persons['marital-status'].min()).div(persons['marital-status'].max() - persons['marital-status'].min()) - 1
    persons['occupation'] = 2*(persons['occupation']-persons['occupation'].min()).div(persons['occupation'].max() - persons['occupation'].min()) - 1
    persons['relationship'] = 2*(persons['relationship']-persons['relationship'].min()).div(persons['relationship'].max() - persons['relationship'].min()) - 1
    persons['capital-gain'][persons['capital-gain'] != 0] = 1
    persons['capital-loss'][persons['capital-loss'] != 0] = 1
    persons['hours-per-week'] = 2*(persons['hours-per-week']-persons['hours-per-week'].min()).div(persons['hours-per-week'].max() - persons['hours-per-week'].min()) - 1
    persons['native-country'] = 2*(persons['native-country']-persons['native-country'].min()).div(persons['native-country'].max() - persons['native-country'].min()) - 1

    label_encoder = preprocessing.LabelEncoder()
    race = label_encoder.fit_transform(persons.race.values).reshape(-1,1)
    data.race = F.one_hot(torch.LongTensor(race).view(-1)).cuda()

    labels = label_encoder.fit_transform(persons.income.values).reshape(-1,1)
    data.y = torch.LongTensor(labels).cuda()
    persons_x = persons.drop(columns=['income'])
    data.x = torch.LongTensor(persons_x.values).cuda()

    edges_np = np.genfromtxt(data_path+"income_edges.txt").astype('int')
    edges_df = pd.DataFrame(edges_np)
    edges_df.columns = ['person_1','person_2']

    edges_all = np.hstack((np.stack([edges_df['person_1'].values,edges_df['person_2'].values]),
                                np.stack([edges_df['person_2'].values,edges_df['person_1'].values])))
    data.edge_index = torch.LongTensor(edges_all).cuda()

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
      persons_split_att = np.load(data_path+f"persons_{split}_att_rb.npy")
      data[f'{split}_id_feat'] = process_att_data(persons_split_att, race,labels)
    return data