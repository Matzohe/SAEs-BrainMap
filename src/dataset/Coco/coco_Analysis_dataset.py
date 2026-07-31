import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from .coco_dataset import CoCoCaptionValDataset, CoCoCaptionTrainDataset


class CocoAnalysisDataset(Dataset):
    def __init__(self, config, image_preprocess=None, text_preprocess=None):
        self.cocoTrainDataset = CoCoCaptionTrainDataset(config.DATASET['coco'], preprocess=image_preprocess)
        if text_preprocess is None:
            text_preprocess = lambda x: [x]
        self.textPreprocess = text_preprocess
        
    def __getitem__(self, index):
        img, caption = self.cocoTrainDataset[index]
        caption = self.textPreprocess(caption).squeeze(0)
        image_root = os.path.join("train2017", "{:012}".format(self.cocoTrainDataset.annotations[index][0]) + ".jpg")
        return img, caption, image_root
    
    def __len__(self):
        return len(self.cocoTrainDataset)