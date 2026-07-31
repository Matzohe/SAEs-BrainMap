import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
from PIL import Image

class CoCoExperimentDataset(Dataset):
    def __init__(self, coco_experiment_root, preprocess=None):
        self.coco_experiment_root = coco_experiment_root
        if preprocess is None:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.preprocess = preprocess
        self.experiment_labels = ["adult", "body", "car", "child", "corridor", "food", "house", "instrument", "limb", "number", "word"]
        self.image_root_list = []
        self.get_image_root_list()

    def get_image_root_list(self, index=0):
        assert index < len(self.experiment_labels)
        current_root = os.path.join(self.coco_experiment_root, self.experiment_labels[index])
        image_root_list = os.listdir(current_root)
        image_root_list = [os.path.join(current_root, each) for each in image_root_list]
        self.image_root_list = image_root_list
    
    def __getitem__(self, index):
        image_root = self.image_root_list[index]
        img = cv2.imread(image_root)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.preprocess(img)

        return img
    
    def __len__(self):
        return len(self.image_root_list)

    def getClassNumber(self):
        return len(self.experiment_labels)