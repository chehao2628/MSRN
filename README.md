# MSRN.pytorch

PyTorch implementation of [](https://arxiv.org/), .

## Update

1.

| Method    | COCO    | NUS-WIDE |VOC2007  |
|:---------:|:-------:|:-------:|:--------:|

| Ours | 83.4 | 61.5 | 94.8 |

### Requirements

Please install the following packages

- numpy
- torch-0.3.1
- torchnet
- torchvision-0.2.0
- tqdm

### Download pretrained models

checkpoint/voc ([GoogleDrive](https://drive.google.com/file/d/1--QgXcZiR6iI-luAT7FdiGBjWogM07xK/view?usp=sharing))

checkpoint/coco ([GoogleDrive](https://drive.google.com/file/d/1x-pSlk6VCEeUgP8ngKXwd07lHxxA6H49/view?usp=sharing))

checkpoint/nus-wide ([GoogleDrive](https://drive.google.com/file/d/1AvXK8j2Pu9YtvmBkKCtUgPQA3xI7Dtxc/view?usp=sharing))

checkpoint/Apparel ([GoogleDrive](https://drive.google.com/file/d/1yfFoAAVL8vb_8F39nErPnXiN9ibQJrzN/view?usp=sharing))

or

[Baidu](https://pan.baidu.com/s/1jgaSxURuxR1Z3AV_-hJ-KA 提取码：1qes )

### Options

- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `batch-size`: number of images per batch
- `image-size`: size of the image
- `epochs`: number of training epochs
- `evaluate`: evaluate model on validation set
- `resume`: path to checkpoint

### Demo VOC 2007

```
python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 -e --resume checkpoint/voc/voc_checkpoint.pth.tar
```

### Demo COCO 2014

```
python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 -e --resume checkpoint/coco/coco_checkpoint.pth.tar
```

### Demo NUS-WIDE 2014

```
python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 -e --resume checkpoint/coco/coco_checkpoint.pth.tar
```

## Citing this repository

```
@inproceedings{2019MSGDN,
author = {},
title = {},
booktitle = {},
year = {}
}
```

## Reference


