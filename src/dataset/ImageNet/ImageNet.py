import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import os
import random
from PIL import Image


def ImageNetPreProcess(image) -> torch.Tensor:
    # ImageNet image preprocess function
    # The hyperparameters in the Normalize is the average and standard deviation of ImageNet
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return preprocess(image).unsqueeze(0)


def get_discriptions(caffe_root):
    labels = OrderedDict()
    with open(os.path.join(caffe_root, "synsets.txt"), "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = " ".join(line.split(" ")[1:])
            label_list = line.split(", ")
            labels[i] = label_list
    return labels

def get_imagenet_labels(config):
    caffe_root = config.DATASET["imagenet_caffe"]


def get_classification(config):
    # caffe file is downloaded from url: http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    caffe_root = config.DATASET["imagenet_caffe"]
    labels = OrderedDict()
    with open(os.path.join(caffe_root, "synsets.txt"), "r") as f:
        for i, line in enumerate(f):
            label = line.strip()
            labels[label] = i

    return labels


def get_train_list(caffe_root):
    # caffe file is downloaded from url: http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    root_list = []
    label_list = []
    with open(os.path.join(caffe_root, "train.txt"), "r") as f:
        for line in f:
            root_list.append(line.split(" ")[0])
            label_list.append(int(line.split(" ")[1]))
    return root_list, label_list


def get_val_list(caffe_root):
    # caffe file is downloaded from url: http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    val_list = []
    label_list = []
    with open(os.path.join(caffe_root, "val.txt"), "r") as f:
        for line in f:
            val_list.append(line.split(" ")[0])
            label_list.append(int(line.split(" ")[1]))

    return val_list, label_list


class ImageNetTrainDataset(Dataset):
    def __init__(self, imagenet_train_root, caffe_root, image_preprocess=ImageNetPreProcess):
        super().__init__()

        self.imagenet_root = imagenet_train_root
        self.preprocess = image_preprocess
        self.root_list, self.label_list = get_train_list(caffe_root=caffe_root)
        self.root_list = [os.path.join(self.imagenet_root, x) for x in self.root_list]

    def __getitem__(self, index):
        image = Image.open(self.root_list[index]).convert("RGB")
        return self.preprocess(image).squeeze(0), self.label_list[index]

    def __len__(self):
        return len(self.root_list)


class ImageNetValDataset(Dataset):
    def __init__(self, imagenet_val_root, caffe_root, image_preprocess=ImageNetPreProcess):
        super().__init__()
        self.imagenet_root = imagenet_val_root
        self.preprocess = image_preprocess
        self.root_list, self.label_list = get_val_list(caffe_root=caffe_root)
        self.root_list = [os.path.join(self.imagenet_root, x) for x in self.root_list]

    def __getitem__(self, index):
        image = Image.open(self.root_list[index]).convert("RGB")
        return self.preprocess(image).squeeze(0), self.label_list[index]

    def __len__(self):
        return len(self.root_list)


class ImageNetTestDataset(Dataset):
    def __init__(self, imagenet_test_root, image_preprocess=ImageNetPreProcess):
        super().__init__()

        self.imagenet_root = imagenet_test_root
        self.preprocess = image_preprocess
        self.root_list = [os.path.join(self.imagenet_root, x) for x in os.listdir(self.imagenet_root)]

    def __getitem__(self, index):
        image = Image.open(self.root_list[index]).convert("RGB")
        return self.preprocess(image).squeeze(0)

    def __len__(self):
        return len(self.root_list)
