import pickle
import random
import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import util
from util import *

object_categories = {'black': 0, 'blue': 1, 'brown': 2, 'green': 3, 'red': 4, 'white': 5, 'dress': 6, 'pants': 7,
                     'shirt': 8, 'shoes': 9, 'shorts': 10}
image_path = "../data/apparel/"
folders = os.listdir(image_path)


# print(folders)

def glove2vec():
    # capture the  glove label embedding of apparel
    path = 'data/glove.6B.300d.txt'
    # path = 'apparel_vec.txt'

    vec_map = {}
    with open(path, 'rb') as file_to_read:
        for line in file_to_read.readlines():
            # lines = file_to_read.readline()  # 整行读取数据
            data = line.split()
            head = str(data[0])[2:-1]
            vec = data[1:]
            if head in object_categories.keys():
                vec_float = []
                for v in vec:
                    vec_float.append(float(v))
                vec_map[object_categories[head]] = vec_float
    glove_vec = []
    vec_path = 'data/apparel_/apparel_glove_word2vec.pkl'
    vec_file = open(vec_path, 'wb')
    for i in range(11):
        glove_vec.append(vec_map[i])
    pickle.dump(glove_vec, vec_file)
    vec_file.close()


def apparel_correspond():
    correspond = {}
    adj = np.zeros((11, 11))
    for folder in folders:
        name = folder.split('_')
        label1, label2 = object_categories[name[0]], object_categories[name[1]]
        images = os.listdir(image_path + folder)
        for img in images:
            labels = [-1] * 11
            labels[label1] = 1
            labels[label2] = 1
            correspond[img] = labels  # image name: label
            adj[label1][label2] += 1
    correspond_file = open('data/apparel_/image_label.pkl', 'wb')
    pickle.dump(correspond, correspond_file)
    adj_file = open('data/apparel_/apparel_adj.pkl', 'wb')
    nums = sum(adj, 1)
    pickle.dump({'adj': adj, 'nums': nums}, adj_file)


# apparel_correspond()

def split_dataset(image_label):
    img_l = list(image_label.items())
    random.seed(10)
    random.shuffle(img_l)

    trainval, test = img_l[:int(len(img_l) * 0.5)], img_l[int(len(img_l) * 0.5):]
    trainval, test = dict(trainval), dict(test)
    return trainval, test


# apparel_correspond()

class ApparelClassification(data.Dataset):
    def __init__(self, root, dataset_map, transform=None, inp_name=None):
        self.root = root
        self.img_list = list(dataset_map.keys())
        self.image_label = dataset_map
        self.transform = transform
        self.num_classes = 11

        with open(inp_name, 'rb') as f:
            self.inp = np.array(pickle.load(f))

    def __len__(self):
        return len(self.image_label)

    def __getitem__(self, index):
        filename = self.img_list[index]
        labels = self.image_label[filename]  # image file name
        img = Image.open(os.path.join(self.root, 'apparel', filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        for i in range(len(labels)):
            if labels[i] == 0:
                target[i] = 1
        return (img, filename, self.inp), target
