import torch
from torch.utils.data import Dataset
from .coco_dataset import CoCoCaptionTrainDataset, CoCoCaptionValDataset
from ..NSD.NSDDataLoader import NSDDataset


class AnalysisDataset(Dataset):
    def __init__(self, args, image_preprocess=None, text_preprocess=None):
        self.NSDDataset = NSDDataset(args)
        subj = int(args.exp.subj)
        roi_name = args.exp.full_roi
        self.cocoTrainDataset = CoCoCaptionTrainDataset(args.dataset.coco, preprocess=image_preprocess)
        self.cocoValDataset = CoCoCaptionValDataset(args.dataset.coco, preprocess=image_preprocess)
        self.image_root_list = self.NSDDataset.extract_image_root(subj=subj, save=False)
        self.individual_mask, self.same_mask = self.NSDDataset.load_individual_and_same_image_bool(subj=subj)
        self.all_image_root_list = self.NSDDataset.extract_image_root(subj=subj, save=False)
        self.all_BrainActivation = self.NSDDataset.load_avg_activation_value(subj=subj, roi_name=roi_name)
        self.BrainActivation = self.NSDDataset.load_avg_activation_value(subj=subj, roi_name=roi_name)
        self.voxel_num = self.all_BrainActivation.shape[-1]
        if text_preprocess is None:
            text_preprocess = lambda x: [x]
        self.textPreprocess = text_preprocess

    def __getitem__(self, index):
        image_root = self.image_root_list[index]
        name = image_root.split("/")[-1].split(".")[0]
        if "val" in image_root:
            img, caption = self.cocoValDataset.getFromName(name)
        else:
            img, caption = self.cocoTrainDataset.getFromName(name)
        caption = self.textPreprocess(caption).squeeze(0)
        return img, caption, self.BrainActivation[index]
    
    def IndividualCondition(self):
        new_brain_activation = self.all_BrainActivation[self.individual_mask]
        new_image_root_list = [self.all_image_root_list[i] if self.individual_mask[i] else "0" for i in range(len(self.individual_mask))]
        try:
            while(True):
                new_image_root_list.remove("0")
        except:
            pass
        self.image_root_list = new_image_root_list
        self.BrainActivation = new_brain_activation

    def SameCondition(self):
        new_image_root_list = [self.all_image_root_list[i] if self.same_mask[i] else "0" for i in range(len(self.same_mask))]
        try:
            while(True):
                new_image_root_list.remove("0")
        except:
            pass
        new_brain_activation = self.all_BrainActivation[self.same_mask]
        self.image_root_list = new_image_root_list
        self.BrainActivation = new_brain_activation

    def __len__(self):
        return len(self.image_root_list)

    def getVoxelNum(self):
        return self.BrainActivation.shape[-1]