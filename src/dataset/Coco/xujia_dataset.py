import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from ..NSD.NSDDataLoader import NSDDataset  # 适当修改路径

class BrainDataset(Dataset):
    def __init__(self, config, subj=1, roi_name="ventral_visual_pathway_roi", resolution=(224, 224)):
        self.NSDDataset = NSDDataset(config)
        subj = int(config.EXP['subj'])  # 你可以从 config 中直接传入
        self.image_root_list = self.NSDDataset.extract_image_root(subj=subj, save=False)  # 获取图像路径
        self.fmri = self.NSDDataset.load_avg_activation_value(subj=subj, roi_name=roi_name).numpy()  # 获取 fMRI 数据

        # 图像预处理
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # 确保图像和 fMRI 数据数量一致
        assert len(self.image_root_list) == self.fmri.shape[0], "Image and fMRI size mismatch"
        print(f"Loaded {len(self.image_root_list)} samples.")

    def __len__(self):
        return len(self.image_root_list)

    def __getitem__(self, idx):
        # 加载图像
        image_path = os.path.join(self.NSDDataset.image_dir, self.image_root_list[idx])  # 获取图像路径
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        # 获取对应的 fMRI 数据
        fmri_vector = self.fmri[idx]
        fmri_vector = torch.from_numpy(fmri_vector).float()

        return image, fmri_vector
