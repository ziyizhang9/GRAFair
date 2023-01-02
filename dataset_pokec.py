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

def load_data(data_path = 'dataset/pokec/'):
    data = Data()
    persons = pd.read_csv(data_path+"region_job.csv", index_col=False)
    
    persons['I_am_working_in_field'][persons['I_am_working_in_field'] != -1] = 0
    persons['I_am_working_in_field'][persons['I_am_working_in_field'] == -1] = 1

    label_encoder = preprocessing.LabelEncoder()
    region = label_encoder.fit_transform(persons.region.values).reshape(-1,1)
    data.region = F.one_hot(torch.LongTensor(region).view(-1)).cuda()

    persons['region'] = pd.to_numeric(persons['region'])
    persons['AGE'] = 2*(persons['AGE']-persons['AGE'].min()).div(persons['AGE'].max() - persons['AGE'].min()) - 1
    persons['completion_percentage'] = 2*(persons['completion_percentage']-persons['completion_percentage'].min()).div(persons['completion_percentage'].max() - persons['completion_percentage'].min()) - 1

    idx_map = {j: i for i, j in enumerate(persons['user_id'])}

    labels = label_encoder.fit_transform(persons.I_am_working_in_field.values).reshape(-1,1)
    data.y = torch.LongTensor(labels).cuda()
    persons_x = persons.drop(columns=['I_am_working_in_field'])
    persons_x = persons.drop(columns=['user_id'])
    data.x = torch.LongTensor(persons_x.values).cuda()
    edges_np = np.genfromtxt(data_path+"region_job_relationship.txt").astype('int')
    edges_df = pd.DataFrame(edges_np)
    edges_df.columns = ['person_1','person_2']

    for index, row in edges_df.iterrows():
      row['person_1']=idx_map[row['person_1']]
      row['person_2']=idx_map[row['person_2']]

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
    np.save(data_path+"persons_train_att.npy", persons_train)
    np.save(data_path+"persons_val_att.npy", persons_val)
    np.save(data_path+"persons_test_att.npy", persons_test)


    for split in ['train','val','test']:
      persons_split_att = np.load(data_path+f"persons_{split}_att.npy")
      data[f'{split}_id_feat'] = process_att_data(persons_split_att, region,labels)

    return data

def load_data_cf(data_path = 'dataset/pokec/'):
    data = Data()
    persons = pd.read_csv(data_path+"region_job.csv", index_col=False)
    
    persons['I_am_working_in_field'][persons['I_am_working_in_field'] != -1] = 0
    persons['I_am_working_in_field'][persons['I_am_working_in_field'] == -1] = 1
    persons['region'] = pd.to_numeric(persons['region'])
    persons['region'] = 1 - persons['region']

    label_encoder = preprocessing.LabelEncoder()
    region = label_encoder.fit_transform(persons.region.values).reshape(-1,1)
    data.region = F.one_hot(torch.LongTensor(region).view(-1)).cuda()

    persons['AGE'] = 2*(persons['AGE']-persons['AGE'].min()).div(persons['AGE'].max() - persons['AGE'].min()) - 1
    persons['completion_percentage'] = 2*(persons['completion_percentage']-persons['completion_percentage'].min()).div(persons['completion_percentage'].max() - persons['completion_percentage'].min()) - 1

    idx_map = {j: i for i, j in enumerate(persons['user_id'])}

    labels = label_encoder.fit_transform(persons.I_am_working_in_field.values).reshape(-1,1)
    data.y = torch.LongTensor(labels).cuda()
    persons_x = persons.drop(columns=['I_am_working_in_field'])
    persons_x = persons.drop(columns=['user_id'])
    data.x = torch.LongTensor(persons_x.values).cuda()
    edges_np = np.genfromtxt(data_path+"region_job_relationship.txt").astype('int')
    edges_df = pd.DataFrame(edges_np)
    edges_df.columns = ['person_1','person_2']

    for index, row in edges_df.iterrows():
      row['person_1']=idx_map[row['person_1']]
      row['person_2']=idx_map[row['person_2']]
      
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
      data[f'{split}_id_feat'] = process_att_data(persons_split_att, region,labels)

    return data


def load_data_rb(data_path = 'dataset/pokec/'):
    data = Data()
    persons = pd.read_csv(data_path+"region_job.csv", index_col=False)
    
    persons['I_am_working_in_field'][persons['I_am_working_in_field'] != -1] = 0
    persons['I_am_working_in_field'][persons['I_am_working_in_field'] == -1] = 1

    label_encoder = preprocessing.LabelEncoder()
    region = label_encoder.fit_transform(persons.region.values).reshape(-1,1)
    data.region = F.one_hot(torch.LongTensor(region).view(-1)).cuda()

    persons['region'] = pd.to_numeric(persons['region'])
    persons['AGE'] = 2*(persons['AGE']-persons['AGE'].min()).div(persons['AGE'].max() - persons['AGE'].min()) - 1
    persons['completion_percentage'] = 2*(persons['completion_percentage']-persons['completion_percentage'].min()).div(persons['completion_percentage'].max() - persons['completion_percentage'].min()) - 1

    idx_map = {j: i for i, j in enumerate(persons['user_id'])}

    labels = label_encoder.fit_transform(persons.I_am_working_in_field.values).reshape(-1,1)
    data.y = torch.LongTensor(labels).cuda()
    persons_x = persons.drop(columns=['I_am_working_in_field'])
    persons_x = persons.drop(columns=['user_id'])
    data.x = torch.LongTensor(persons_x.values).cuda()
    data.x = data.x + torch.ones(data.x.shape).normal_(0,1).cuda()
    edges_np = np.genfromtxt(data_path+"region_job_relationship.txt").astype('int')
    edges_df = pd.DataFrame(edges_np)
    edges_df.columns = ['person_1','person_2']

    for index, row in edges_df.iterrows():
      row['person_1']=idx_map[row['person_1']]
      row['person_2']=idx_map[row['person_2']]
      
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
      data[f'{split}_id_feat'] = process_att_data(persons_split_att, region,labels)

    return data