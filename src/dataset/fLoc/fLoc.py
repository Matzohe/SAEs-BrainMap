from torch.utils.data import Dataset
import os
from PIL import Image


class fLocDataset(Dataset):
    def __init__(self, floc_root, image_preprocess=None, floc_label=["adult", "body", "car", "child", "corridor", "house", "instrument", "limb", "number", "scrambled", "word"]):

        self.floc_root_list = []
        for label in floc_label:
            root = floc_root.format(label)
            self.floc_root_list.extend([os.path.join(root, each) for each in os.listdir(root)])

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.floc_root_list)
    
    def __getitem__(self, index):
        root = self.floc_root_list[index]
        image = Image.open(root).convert("RGB")
        return self.image_preprocess(image).squeeze(0)

