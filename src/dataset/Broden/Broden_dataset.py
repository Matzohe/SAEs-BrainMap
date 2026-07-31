import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import OrderedDict
from PIL import Image
import csv
import cv2
import os

# load the information of the Broden dataset
# all the image here is used for test
class BrodenDataset(Dataset):
    def __init__(self, broden_root, broden_info_root, image_preprocess=None,):
        super(BrodenDataset, self).__init__()
        self.image_preprocess = image_preprocess
        self.broden_root = broden_root
        self.broden_info_root = broden_info_root
        self.image_root = os.path.join(self.broden_root, "images")
        self.information_list = self.getImageInfo()

    def getImageInfo(self):
        info_root = self.broden_info_root
        if os.path.exists(info_root):
            return torch.load(info_root)
        information_list = []
        index_csv_root = os.path.join(self.broden_root, "index.csv")
        with open(index_csv_root, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for line in reader:
                line_information = {}
                image_root = line[0]
                line_information["file_name"] = image_root
                information_list.append(line_information)
        torch.save(information_list, info_root)
        return information_list

    def __len__(self):
        return len(self.information_list)

    def __getitem__(self, index):
        information_dict = self.information_list[index]
        image_root = information_dict['file_name']
        img = cv2.imread(os.path.join(self.image_root, image_root))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.image_preprocess(img)
        return img
    
    def getFromName(self, file_name):
        discription = self.discription_dict[file_name]
        img = cv2.imread(os.path.join(self.image_root, file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.image_preprocess(img)
        return img, discription