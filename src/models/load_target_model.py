import torch
from torchvision.transforms import transforms
from .Vision import clip
from .Vision.NACLIP.naclip import load as naclip_load
from .Vision.DinoV2.modifiedDiNO import ModifiedDiNOv2
from .Vision.SAM.ModifiedSAM import ModifiedSAM
from .Vision.MAE.ModifiedMAE import ModifiedMAE
from .Vision.ImageNet.imagenet import ModifiedImgNet


def load_target_model(model_name):
    if model_name == "clip_vit-b_16":
        target_model, image_preprocess = clip.load("ViT-B/16", device="cpu")
    elif model_name == "naclip_vit-b_16":
        target_model, image_preprocess = naclip_load("ViT-B/16", device="cpu")
        # naclip的模型加载和clip一样，区别在于后续的forward里:
    elif model_name == "clip_resnet50":
        target_model, image_preprocess = clip.load("RN50", device="cpu")
    elif model_name == "sam":
        image_preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        target_model = ModifiedSAM()
        # 插入导入模型的函数
        # 返回两个东西，第一个是模型本体（包含两个额外的函数，第一个encode_image，返回第一个位置是image embedding，第二个位置是模型的逐层残差流输出，第二个额外的函数 getVisualDim，输入layer层，返回对应layer的feature dim，int值（比如768））
        # 第二个是模型对应的图像预处理方法（这个图像预处理方法，输入是PIL.Image.imread()的输入，不是numpy），用transforms构建一个一样的
    elif model_name == "dinov2":
        image_preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        target_model = ModifiedDiNOv2(model_path="/home/brainai1/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")
    elif model_name == "mae":
        image_preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        target_model = ModifiedMAE()
    elif model_name == "imagenet":
        image_preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        target_model = ModifiedImgNet()
    else:
        raise NotImplementedError()
    return target_model, image_preprocess