# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
import random
import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from utils import utils
from scipy.sparse import csr_matrix

class BaseReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self._read_data()

        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        self.residual_clicked_set = dict()  # store the residual clicked item set of each user
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for uid, iid in zip(df['user_id'], df['item_id']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)

    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','item_id'])
            self.data_df[key] = utils.eval_list_columns(self.data_df[key]) ####此处把csv更改为.txt
            logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by=['user_id', 'item_id'])
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])

        logging.info('Counting dataset statistics...')
        key_columns = ['user_id', 'item_id', 'time']
        if 'label' in self.data_df['train'].columns:
            key_columns.append('label')

        self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1

        # 计算标签数量
        if 'tagID' in self.data_df['train'].columns:
            self.n_tags = self.data_df['train']['tagID'].max() + 1
        else:
            self.n_tags = 0

        for key in ['dev', 'test']:
            if 'neg_items' in self.data_df[key].columns:
                self.data_df[key]['neg_items'] = self.data_df[key]['neg_items'].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                )
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                if neg_items.ndim == 1:
                    neg_items = neg_items.astype(int)
                else:
                    neg_items = np.array([[int(item) for item in row] for row in neg_items])
                for row in neg_items:
                    if isinstance(row, (list, np.ndarray)):
                        assert all(item < self.n_items for item in row), f"Invalid item in {row}"
                    else:
                        assert row < self.n_items, f"Invalid item: {row}"
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df)))
        if 'label' in key_columns:
            positive_num = (self.all_df.label == 1).sum()
            logging.info('"# positive interaction": {} ({:.1f}%)'.format(
                positive_num, positive_num / self.all_df.shape[0] * 100))
        #######################################
        #######################################
        # 计算标签数量
        if 'tagID' in self.data_df['train'].columns:
            self.n_tags = self.data_df['train']['tagID'].max() + 1
        else:
            self.n_tags = 0

        # 计算邻接矩阵
        self.adj_mat = self._build_adjacency_matrix()
        #######################################
        logging.info('Counting dataset statistics...')
        key_columns = ['user_id','item_id','time']
        if 'label' in self.data_df['train'].columns: # Add label for CTR prediction
            key_columns.append('label')

       
        self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        for key in ['dev', 'test']:
            if 'neg_items' in self.data_df[key].columns:
                # 将字符串解析为列表（如果需要）
                self.data_df[key]['neg_items'] = self.data_df[key]['neg_items'].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                )

                # 转换为 numpy 数组
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())

                # 确保所有值都是整数
                if neg_items.ndim == 1:
                    neg_items = neg_items.astype(int)
                else:
                    neg_items = np.array([[int(item) for item in row] for row in neg_items])

                # 逐行检查
                for row in neg_items:
                    if isinstance(row, (list, np.ndarray)):
                        assert all(item < self.n_items for item in row), f"Invalid item in {row}"
                    else:
                        assert row < self.n_items, f"Invalid item: {row}"
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df)))
        if 'label' in key_columns:
            positive_num = (self.all_df.label==1).sum()
            logging.info('"# positive interaction": {} ({:.1f}%)'.format(
				positive_num, positive_num/self.all_df.shape[0]*100))
        ######################################AttributeError: 'BaseReader' object has no attribute 'user2tag'
        self.user2tag = self._build_user2tag()
        self.item2tag = self._build_item2tag()
    ####################

    def _build_adjacency_matrix(self):
        # 这里需要实现构建邻接矩阵的逻辑
        n_users = self.n_users
        n_items = self.n_items
        rows = []
        cols = []
        #data = []
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for _, row in df.iterrows():
                user_id = row['user_id']
                item_id = row['item_id']
                rows.append(user_id)
                cols.append(item_id)
                #data.append(1)
        adj_mat = csr_matrix((item_id, (rows, cols)), shape=(n_users, n_items))
        return scipy_to_torch_sparse(adj_mat)

    def save(self, path):
            with open(path, 'wb') as f:
                pickle.dump(self.__dict__, f)
############################################
    def _build_user2tag(self):
        user2tag = {}
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for _, row in df.iterrows():
                user_id = row['user_id']
                item_id = row['item_id'] 
                if user_id not in user2tag:
                    user2tag[user_id] = []
                user2tag[user_id].append(item_id)
        
        # 将字典转换为二维数组
        max_tags = max(len(tags) for tags in user2tag.values())
        user2tag_list = [user2tag[user_id] + [0] * (max_tags - len(tags)) for user_id, tags in user2tag.items()]
        
        return user2tag_list
    def _build_item2tag(self): 
        item2tag = {}
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for _, row in df.iterrows():
                user_id = row['user_id']
                item_id = row['item_id']
                if item_id not in item2tag:
                    item2tag[item_id] = []
                item2tag[item_id].append(user_id)
        
        # 将字典转换为二维数组
        max_tags = max(len(tags) for tags in item2tag.values())
        item2tag_list = [item2tag[item_id] + [0] * (max_tags - len(tags)) for item_id, tags in item2tag.items()]
        
        return item2tag_list
def scipy_to_torch_sparse(matrix):
    coo = matrix.tocoo()  # 转换为 COO 格式
    row = torch.LongTensor(coo.row)
    col = torch.LongTensor(coo.col)
    #data = torch.FloatTensor(coo.data)
    shape = coo.shape
    return torch.sparse.FloatTensor(torch.stack([row, col]), torch.Size(shape)) 