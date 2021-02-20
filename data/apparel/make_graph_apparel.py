# example on VOC2007 dataset
# some code are refer to code of "Learning Semantic-Specific Graph Representation for Multi-Label Image Recognition"
# also refer to ideas from http://people.brunel.ac.uk/~icstmkw/ma1795/recent_printed_versions/print_2x2_ohpma1795_p2_03.pdf
from xml.dom.minidom import parse
import numpy as np
import xml.dom.minidom
import os
import pickle

from scipy import io as sio


def make_graph():
    category_info = {'black': 0, 'blue': 1, 'brown': 2, 'green': 3, 'red': 4, 'white': 5, 'dress': 6, 'pants': 7,
                     'shirt': 8, 'shoes': 9, 'shorts': 10}

    # img_dir = './data/VOCdevkit/VOC2007/JPEGImages'
    # anno_path = './data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
    # labels_path = './data/VOCdevkit/VOC2007/Annotations'
    img_dir = 'E:/dataset/VOCdevkit/VOC2012/JPEGImages'
    anno_path = 'E:/dataset/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
    labels_path = 'E:/dataset/VOCdevkit/VOC2012/Annotations'
    labels = []
    train_graph = np.zeros((11, 11))  # 共现矩阵


# Construct graph before training network
make_graph()
