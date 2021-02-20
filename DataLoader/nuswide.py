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

object_categories = ['clouds', 'sky', 'person', 'street', 'window', 'tattoo', 'wedding', 'animal', 'cat', 'buildings',
                     'tree', 'airport', 'plane', 'water', 'grass', 'cars', 'road', 'snow', 'sunset', 'railroad',
                     'train', 'flowers', 'plants', 'house', 'military', 'horses', 'nighttime', 'lake', 'rocks',
                     'waterfall', 'sun', 'vehicle', 'sports', 'reflection', 'temple', 'statue', 'ocean', 'town',
                     'beach', 'tower', 'toy', 'book', 'bridge', 'fire', 'mountain', 'rainbow', 'garden', 'police',
                     'coral', 'fox', 'sign', 'dog', 'cityscape', 'sand', 'dancing', 'leaf', 'tiger', 'moon', 'birds',
                     'food', 'cow', 'valley', 'fish', 'harbor', 'bear', 'castle', 'boats', 'running', 'glacier',
                     'swimmers', 'elk', 'frost', 'protest', 'soccer', 'flags', 'zebra', 'surf', 'whales', 'computer',
                     'earthquake', 'map']
object_categories_map = {'clouds': 0, 'sky': 1, 'person': 2, 'street': 3, 'window': 4, 'tattoo': 5, 'wedding': 6,
                         'animal': 7, 'cat': 8, 'buildings': 9, 'tree': 10, 'airport': 11, 'plane': 12, 'water': 13,
                         'grass': 14, 'cars': 15, 'road': 16, 'snow': 17, 'sunset': 18, 'railroad': 19, 'train': 20,
                         'flowers': 21, 'plants': 22, 'house': 23, 'military': 24, 'horses': 25, 'nighttime': 26,
                         'lake': 27, 'rocks': 28, 'waterfall': 29, 'sun': 30, 'vehicle': 31, 'sports': 32,
                         'reflection': 33, 'temple': 34, 'statue': 35, 'ocean': 36, 'town': 37, 'beach': 38,
                         'tower': 39, 'toy': 40, 'book': 41, 'bridge': 42, 'fire': 43, 'mountain': 44, 'rainbow': 45,
                         'garden': 46, 'police': 47, 'coral': 48, 'fox': 49, 'sign': 50, 'dog': 51, 'cityscape': 52,
                         'sand': 53, 'dancing': 54, 'leaf': 55, 'tiger': 56, 'moon': 57, 'birds': 58, 'food': 59,
                         'cow': 60, 'valley': 61, 'fish': 62, 'harbor': 63, 'bear': 64, 'castle': 65, 'boats': 66,
                         'running': 67, 'glacier': 68, 'swimmers': 69, 'elk': 70, 'frost': 71, 'protest': 72,
                         'soccer': 73, 'flags': 74, 'zebra': 75, 'surf': 76, 'whales': 77, 'computer': 78,
                         'earthquake': 79, 'map': 80}


def glove2vec():
    # capture the  glove label embedding of apparel
    path = 'data/glove.6B.300d.txt'
    # path = 'apparel_vec.txt'

    vec_map = {}
    with open(path, 'rb') as file_to_read:
        c = 0
        for line in file_to_read.readlines():
            # lines = file_to_read.readline()  # 整行读取数据
            data = line.split()
            head = str(data[0])[2:-1]
            vec = data[1:]
            if head in object_categories:
                c += 1
                print('Found the word:', head)
                vec_float = []
                for v in vec:
                    vec_float.append(float(v))
                vec_map[object_categories_map[head]] = vec_float
        print(c)

    glove_vec = []
    vec_path = '../data/nuswide/nuswide_glove_word2vec.pkl'
    vec_file = open(vec_path, 'wb')
    for i in range(len(object_categories)):
        glove_vec.append(vec_map[i])
    pickle.dump(glove_vec, vec_file)
    vec_file.close()


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def read_object_labels_csv(file, set, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        adj = np.zeros((len(object_categories), len(object_categories)))
        for row in reader:
            if row[2] == set:
                if header and rownum == 0:
                    header = row
                else:
                    if num_categories == 0:
                        num_categories = len(row) - 1
                    name = row[0]
                    label = []
                    label_str = row[1][1:-1]
                    label_str_list = label_str.split(', ')
                    for l in label_str_list:
                        cur = l[1:-1]
                        label.append(object_categories_map[cur])
                    labels = (np.asarray(label)).astype(np.float32)
                    # Create adjcent matrix here
                    # for i in labels:
                    #     for j in labels:
                    #         adj[int(i)][int(j)] += 1
                    targets = [-1] * len(object_categories)
                    for l in labels:
                        targets[int(l)] = 1
                    targets = torch.tensor(targets)
                    item = (name, targets)
                    images.append(item)

                rownum += 1
        # adj_file = open('data/nuswide/nuswide_adj.pkl', 'wb')
        # nums = sum(adj, 1)
        # pickle.dump({'adj': adj, 'nums': nums}, adj_file)
        # print(adj)

    return images


class NusWideClassification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
        self.root = root
        self.path_images = root
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # define path of csv file
        # define filename of csv file
        file_csv = '/content/ML-GCN-SGD/data/nuswide/nus_wid_data.csv'

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv, set)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

        print('[dataset] Nus-Wide classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, path, torch.tensor(self.inp)), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
