import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import json
import cv2
import numpy as np
from collections import OrderedDict
from torchvision import transforms
import math
import random
from PIL import Image


class CoCoCaptionTrainDataset(Dataset):
    def __init__(self, coco_root, preprocess=None, mission="caption", shuffle=False):
        # Expect the structure of coco dataset is the same with the structure of coco_2017
        # target list include "instances", "captions", "keypoints", et.al, see config.cfg for more details
        self.coco_root = coco_root
        if preprocess is None:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.preprocess = preprocess
        self.mission = mission
        self.annotations = []
        image_id_set = set()
        self.name2idx = {}
        with open(os.path.join(self.coco_root, "train_annotations", "captions_train2017.json"), "r") as f:
            train_annotations = json.load(f)["annotations"]
            if mission not in train_annotations[0].keys():
                raise NotImplementedError("loading coco dataset error, no such annotation key named {}".format(mission))
            
            for each in train_annotations:
                if each["image_id"] in image_id_set:
                    continue
                else:
                    image_id_set.add(each["image_id"])
                self.annotations.append((each["image_id"], each[self.mission]))
        
        # shuffle the ordered dict
        if shuffle:
            random.shuffle(self.annotations)
        
        for i, each in enumerate(self.annotations):
            self.name2idx["{:012}".format(each[0])] = i

    def __getitem__(self, index):
        file_name = os.path.join(self.coco_root, "train2017", "{:012}".format(self.annotations[index][0]) + ".jpg")
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.preprocess(img)

        return img, self.annotations[index][1]

    def getFromName(self, name):
        return self.__getitem__(self.name2idx[name])

    def __len__(self):
        return len(self.annotations)
    

class CoCoCaptionValDataset(Dataset):
    def __init__(self, coco_root, preprocess=None, mission="caption", shuffle=False):
        # Expect the structure of coco dataset is the same with the structure of coco_2017
        # target list include "instances", "captions", "keypoints", et.al, see config.cfg for more details
        self.coco_root = coco_root
        if preprocess is None:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.preprocess = preprocess
        self.mission = mission
        self.annotations = []
        image_id_set = set()
        self.name2idx = {}
        with open(os.path.join(self.coco_root, "train_annotations", "captions_val2017.json"), "r") as f:
            val_annotations = json.load(f)["annotations"]
            if mission not in val_annotations[0].keys():
                raise NotImplementedError("loading coco dataset error, no such annotation key named {}".format(mission))
            
            for each in val_annotations:
                if each["image_id"] in image_id_set:
                    continue
                else:
                    image_id_set.add(each["image_id"])
                self.annotations.append((each["image_id"], each[self.mission]))
        # shuffle the ordered dict
        if shuffle:
            random.shuffle(self.annotations)
        
        for i, each in enumerate(self.annotations):
            self.name2idx["{:012}".format(each[0])] = i

    def __getitem__(self, index):
        file_name = os.path.join(self.coco_root, "val2017", "{:012}".format(self.annotations[index][0]) + ".jpg")
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.preprocess(img)

        return img, self.annotations[index][1]

    def getFromName(self, name):
        return self.__getitem__(self.name2idx[name])

    def __len__(self):
        return len(self.annotations)

