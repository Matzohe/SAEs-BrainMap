import torch
from torch.utils.data import Dataset
import os
from .Coco.coco_dataset import CoCoCaptionValDataset, CoCoCaptionTrainDataset
from .Broden.Broden_dataset import BrodenDataset

class CombineDataset(Dataset):
    def __init__(self, coco_root, broden_root, broden_info_root, image_preprocess=None):
        super(CombineDataset, self).__init__()
        self.val_coco_datasets = CoCoCaptionValDataset(coco_root, preprocess=image_preprocess)
        self.broden_datasets = BrodenDataset(broden_root, broden_info_root, image_preprocess=image_preprocess)

    def __len__(self):
        return len(self.val_coco_datasets) + len(self.broden_datasets)

    def __getitem__(self, index):
        if index > len(self.val_coco_datasets) - 1:
            img = self.broden_datasets[index - len(self.val_coco_datasets)]
            img_root = self.broden_datasets.information_list[index - len(self.val_coco_datasets)]['file_name']
        else:
            img, _ = self.val_coco_datasets[index]
            img_root = os.path.join("val2017", "{:012}".format(self.val_coco_datasets.annotations[index][0]) + ".jpg")
        return img, img_root