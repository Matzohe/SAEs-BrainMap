import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import h5py
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path
import shutil
from easydict import EasyDict
from torch.utils.data import DataLoader
from ...dataset.Coco.coco_experiment_dataset import CoCoExperimentDataset
from ...dataset.ImageNet.ImageNet import ImageNetTestDataset
from ...models.Vision import clip
from ...util import check_path
from ...models.load_target_model import load_target_model
from ...SAEs.sae_loader import load_pretrained_autoencoder


def clip_loading(device="cuda:0"): 
    """
    加载用于分析的CLIP模型, 以及五个类别的prompt
    """
    faces_prompt = [
        "A photo of a person's face",
        "A portrait photo of a face",
        "A face facing the camera",
        "A photo of a face",
        "A photo of an animal's face",
        "A photo of faces",
        "People looking at the camera",
        "A portrait of a person",
        "A portrait photo",
    ]

    bodies_prompt = [
        "A photo of a torso",
        "A photo of limbs",
        "A photo of bodies",
        "A photo of people", 
        "A photo of animals",
        "A photo of a body",
        "A person's arms",
        "A person's legs",
        "A photo of hands",
    ]

    places_prompt = [
        "A photo of a bedroom",
        "A photo of an office",
        "A photo of a hallway",
        "A photo of a doorway",
        "A photo of interior design",
        "A photo of a building",
        "A photo of a house",
        "A photo of nature",
        "A photo of a landscape",
    ]

    food_prompt = [
        "A photo of food",
        "A photo of cuisine",
        "A photo of fruit",
        "A photo of foodstuffs",
        "A photo of a meal",
        "A photo of bread",
        "A photo of rice",
        "A photo of a snack",
        "A photo of pastries",
    ]

    words_prompt = [
        "A photo of words",
        "A photo of glyphs",
        "A photo of a glyph",
        "A photo of text",
        "A photo of numbers",
        "A photo of a letter",
        "A photo of letters",
        "A photo of writing",
        "A photo of text on an object",
    ]

    others_prompt = [
        "A photo of a single solid color",
        "A photo emphasizing smooth color gradients",
        "A photo emphasizing high contrast",
        "A photo of texture",
        "A photo containing simple geometric shapes only",
        "A photo emphasizing edges and contours",
        "A photo of grid-like pattern", 
        "A photo with aligned elements",
        "A blurred photo",
    ]

    model, preprocess = clip.load("ViT-B/16", device=device)
    faces_tokens = clip.tokenize(faces_prompt).to(device)
    bodies_tokens = clip.tokenize(bodies_prompt).to(device)
    places_tokens = clip.tokenize(places_prompt).to(device)
    food_tokens = clip.tokenize(food_prompt).to(device)
    words_tokens = clip.tokenize(words_prompt).to(device)
    others_tokens = clip.tokenize(others_prompt).to(device)

    return model, preprocess, faces_tokens, bodies_tokens, places_tokens, food_tokens, words_tokens, others_tokens

def SAEs_image_loading(
        roi_dict_save_root = "/home/brainai1/zmmao/VLM-Memory/experiments/output/sae_evaluation/clip_evaluation/roi_dict.pt"
    ):
    """
    这个函数的作用主要是，将每个特征的选择性图像提取出来，然后保存每个特征的top选择性图片路径
    """
    if os.path.exists(roi_dict_save_root):
        roi_dict = torch.load(roi_dict_save_root, weights_only=False)
        return roi_dict
    img_root = "/home/brainai1/zmmao/VLM-Memory/experiments/output/roi_selected_feature_evaluation/heatmap_independent/subj5/{}/{}_original_rate16/layer{}"
    model_list = ["clip_vit-b_16", "imagenet", "dinov2", "mae"]
    layer_list = [i for i in range(12)]
    roi_list = ["FFA", "EBA", "RSC", "FOOD", "VWFA"]
    img_save_root = "/home/brainai1/VDisk2/BrainAi1/VLM-Memory/experiments/output/roi_selected_feature_evaluation/heatmap_independent_clip/subj5/{}/{}_original_rate16/layer{}"
    
    # 这里的保存格式是一个字典
    # 保存的信息是{roi_name: {model_name: List[model_layer_list[ordered_dict{feature_id: [feature_paths, from 0-19]}], ]}}
    # 最外层是roi名称
    roi_dict = {}
    for roi in roi_list:
        model_information_dict = {}
        for model_name in model_list:
            model_information_dict[model_name] = []
        roi_dict[roi] = model_information_dict

    for roi in roi_list:
        for model_name in model_list:
            for layer in layer_list:
                layer_image_root = OrderedDict()
                current_img_root = img_root.format(roi, model_name, layer)
                current_img_save_root = img_save_root.format(roi, model_name, layer)
                current_img_dir_list = os.listdir(current_img_root)
                # 按照相关性排序
                current_img_dir_list = sorted(current_img_dir_list, key=lambda x: int(x.split("top")[-1].split("_")[0]))
                for img_dir in current_img_dir_list:
                    feature_id = int(img_dir.split("_")[1])
                    feature_image_root_list = []
                    img_dir_path = os.path.join(current_img_root, img_dir)
                    img_name_list = os.listdir(img_dir_path)
                    for img_name in img_name_list:
                        if "original" not in img_name:
                            continue
                        img_save_dir = os.path.join(current_img_save_root, img_dir, img_name)
                        feature_image_root_list.append(img_save_dir)
                    feature_image_root_list = sorted(feature_image_root_list, key=lambda x: int(x.split("original_")[-1].split(".")[0]))
                    layer_image_root[feature_id] = feature_image_root_list
                roi_dict[roi][model_name].append(layer_image_root)

    check_path(roi_dict_save_root)

    torch.save(roi_dict, roi_dict_save_root)

    return roi_dict


def SAEs_feature_evaluation(device="cuda:0"):
    roi_list = ["FFA", "EBA", "RSC", "VWFA", "FOOD"]
    model_list = ["clip_vit-b_16"] # , "imagenet", "dinov2", "mae"]
    layer_list = [i for i in range(12)]
    roi_dict = SAEs_image_loading()
    clip_model, img_preprocess, faces_tokens, bodies_tokens, places_tokens, food_tokens, words_tokens, others_tokens = clip_loading(device)
    text_prompt = torch.cat([faces_tokens, bodies_tokens, places_tokens, food_tokens, words_tokens, others_tokens], dim=0)
    roi_save_info_dict = {}
    for roi in roi_list:
        model_information_dict = {}
        for model_name in model_list:
            model_information_dict[model_name] = []
        roi_save_info_dict[roi] = model_information_dict

    for roi in roi_list:
        for model_name in model_list:
            for layer in layer_list:
                layer_image_root = roi_dict[roi][model_name][layer]
                layer_all_info = OrderedDict()
                for feature_id, feature_image_root_list in tqdm(layer_image_root.items(), total=len(layer_image_root)):
                    # 这里有很多张图片
                    # 首先我们要确定这个特征的选择性是什么相关的
                    # 计算方式是，统计这个特征的每张图片与所有类别的相关性，看最相关的是属于哪个类的
                    # 然后统计这个特征的所有图片在所有类上的分布，当归属于一个类的图片超过一个阈值的时候
                    # 我们认为这个类的选择性是与某一个特征直接相关的
                    # 这里保存的信息为[每张图片的最高选择性的特征id]，[完成统计后的每个类的数量，总共为5类]
                    # 后续这个信息可以用于计算准确率以及分布情况
                    # 其次我们想要统计，这个特征的选择性图片和某一个类别的相关性的最大值
                    # 这里做一个softmax，统计每个特征的最高相关度
                    all_image_list = []
                    for image_path in feature_image_root_list:
                        img = cv2.imread(image_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img)
                        img = img_preprocess(img)
                        all_image_list.append(img.unsqueeze(0))
                    all_image_list = torch.cat(all_image_list, dim=0).to(device=device, dtype=torch.half)
                    with torch.no_grad():
                        logits_per_image = clip_model(all_image_list, text_prompt)[0]
                        selected_class_id = torch.argmax(logits_per_image, dim=1).view(-1).cpu()
                        selected_class = selected_class_id // 9
                        selected_class = [(selected_class == ids).cpu().sum() for ids in range(6)]
                        selected_class_similarity = torch.softmax(logits_per_image, dim=1).cpu()
                        max_selected_class_similarity = torch.max(selected_class_similarity, dim=1).values.view(-1)
                    layer_all_info[feature_id] = [selected_class, selected_class_id, max_selected_class_similarity]

                roi_save_info_dict[roi][model_name].append(layer_all_info)
    
    roi_save_info_dict_save_root = "/home/brainai1/VDisk2/BrainAi1/VLM-Memory/experiments/output/roi_save_info_dict_with_others.pt"
    check_path(roi_save_info_dict_save_root)
    torch.save(roi_save_info_dict, roi_save_info_dict_save_root)

    return roi_save_info_dict

def data_evaluation():
    roi_save_info_dict_save_root = "/home/brainai1/VDisk2/BrainAi1/VLM-Memory/experiments/output/roi_save_info_dict_with_others.pt"
    roi_save_info_dict = torch.load(roi_save_info_dict_save_root)
    roi_list = ["FFA", "EBA", "RSC", "FOOD", "VWFA"]
    model_list = ["clip_vit-b_16", "imagenet", "dinov2", "mae"]
    for model_name in model_list:
        for j, roi_name in enumerate(roi_list):
            for layer in range(12):
                current_layer_info = roi_save_info_dict[roi_name][model_name][layer]
                top10_classes = torch.zeros(size=(6,))
                # 这一步是统计top10的类分布，看选出来的图片占比为多少
                num = 0
                for feature_id, feature_info in current_layer_info.items():
                    classes_info = feature_info[1]
                    classes_info_top10 = classes_info // 9
                    top10_classes += torch.bincount(classes_info_top10, minlength=6)
                    num += 1
                    if num == 10:
                        break
                selected_feature_number = 0
                feature_id_list = []
                for feature_id, feature_info in current_layer_info.items():
                    classes_info = feature_info[0]
                    classes_info = (torch.tensor(classes_info) > 10)
                    selected_feature_number += classes_info.sum().item()
                    if classes_info.sum() > 0 and classes_info[j] > 0:
                        feature_id_list.append(feature_id)
                print("current model:", model_name, "current roi:", roi_name, "current layer:", layer, "selected feature number:", selected_feature_number, "top10 classes:", top10_classes.tolist())
                print(feature_id_list)
                print("")


def data_collection_depending_on_data_evaluation():
    roi_list = ["FFA", "EBA", "RSC", "FOOD", "VWFA"]
    model_list = ["clip_vit-b_16", "imagenet", "dinov2", "mae"]
    image_collect_root = "/home/brainai1/VDisk2/BrainAi1/VLM-Memory/experiments/output/roi_selected_feature_evaluation/heatmap_independent_clip/subj5/{}/{}_original_rate16/layer{}"
    image_new_collect_root = "/home/brainai1/VDisk2/BrainAi1/VLM-Memory/experiments/output/roi_selected_feature_evaluation/heatmap_independent_clip_and_selected/subj5/{}/{}_original_rate16/layer{}"
    roi_save_info_dict_save_root = "/home/brainai1/VDisk2/BrainAi1/VLM-Memory/experiments/output/roi_save_info_dict_with_others.pt"
    roi_save_info_dict = torch.load(roi_save_info_dict_save_root, weights_only=False)
    for j, roi_name in enumerate(roi_list):
        for model_name in model_list:
            for layer in range(12):
                image_collect_root_current = image_collect_root.format(roi_name, model_name, layer)
                image_new_collect_root_current = image_new_collect_root.format(roi_name, model_name, layer)
                image_files = os.listdir(image_collect_root_current)
                current_info = roi_save_info_dict[roi_name][model_name][layer]
                for feature_id, feature_info in current_info.items():
                    classes_info = feature_info[0]
                    classes_info = (torch.tensor(classes_info) > 10)
                    if classes_info.sum() > 0 and classes_info[j] > 0:
                        image_file_name = [image_files[i] for i in range(len(image_files)) if "_{}".format(feature_id) in image_files[i]][0]
                        src = Path(image_collect_root_current + "/" + image_file_name)
                        dst_dir = Path(image_new_collect_root_current + "/" + image_file_name)
                        shutil.copytree(src, dst_dir, dirs_exist_ok=True)

def ablation_study_depending_on_data_evaluation(args: EasyDict):
    class_labels = ["adult", "body", "car", "child", "corridor", "food", "house", "instrument", "limb", "number", "word"]
    roi_list = ["FFA", "EBA", "RSC", "FOOD", "VWFA"]
    model_list = ["clip_vit-b_16", "imagenet", "dinov2", "mae"]
    # 首先提取出每个模型，每个roi，每层的saes信息
    roi_save_info_dict_save_root = "/home/brainai1/VDisk2/BrainAi1/VLM-Memory/experiments/output/roi_save_info_dict_with_others.pt"
    roi_save_info_dict = torch.load(roi_save_info_dict_save_root, weights_only=False)
    device = args.exp.device
    sae_type = eval(args.autoencoder.dtype)
    all_ablation_save_root = "/home/brainai1/VDisk2/BrainAi1/VLM-Memory/experiments/output/ablation/{}/{}_mean_and_max_ablation_result.pt"
    for model_name in model_list:
        # 加载目标模型
        target_model, image_preprocess = load_target_model(model_name=model_name)
        target_model = target_model.to(device=device)
        p = next(target_model.parameters())
        current_dtype = p.dtype
        args.exp.model_name = model_name
        # 加载saes模型
        saes_list = []
        for i in range(args.exp.layers):
            saes = load_pretrained_autoencoder(args, i)
            saes_list.append(saes.to(device=device, dtype=sae_type))
        target_dataset = CoCoExperimentDataset(args.dataset.coco_experiment, preprocess=image_preprocess)
        for i in tqdm(range(len(class_labels))):
            
            target_dataset.get_image_root_list(i)
            target_dataloader = DataLoader(target_dataset, batch_size=8, shuffle=False)
            roi_feature_dict = {}
            for j, roi_name in enumerate(roi_list):
                all_layer_feature_list = []
                for layer in range(args.exp.layers):
                    current_layer_info = roi_save_info_dict[roi_name][model_name][layer]
                    feature_id_list = []
                    for feature_id, feature_info in current_layer_info.items():
                        classes_info = feature_info[0]
                        classes_info = (torch.tensor(classes_info) > 10)
                        if classes_info.sum() > 0 and classes_info[j] > 0:
                            feature_id_list.append(feature_id)
                    feature_id_list = torch.tensor(feature_id_list, dtype=torch.long).to(device=device)
                    all_layer_feature_list.append(feature_id_list)
                roi_feature_dict[roi_name] = all_layer_feature_list
            # 下面将测试数据放入模型中，然后将选择出来的特征进行ablation，然后查看每一张图片
            # 在进行ablation之后，与原先的feature之间有多少差异
            # 这里需要一个保存损失的list
            # 每个roi，每个层，每个feature id，都要进行保存
            # 同时，需要保存一个随机ablation的情况来进行对比
            # 这一块估计会非常消耗时间，以及空间
            MSE_max_difference_list = torch.zeros(size=(len(roi_list), args.exp.layers)).cpu()
            angle_max_difference_list = torch.zeros(size=(len(roi_list), args.exp.layers)).cpu()
            MSE_mean_difference_list = torch.zeros(size=(len(roi_list), args.exp.layers)).cpu()
            angle_mean_difference_list = torch.zeros(size=(len(roi_list), args.exp.layers)).cpu()
            random_MSE_mean_difference_list = torch.zeros(size=(len(roi_list), args.exp.layers)).cpu()
            random_angle_mean_difference_list = torch.zeros(size=(len(roi_list), args.exp.layers)).cpu()
            random_MSE_max_difference_list = torch.zeros(size=(len(roi_list), args.exp.layers)).cpu()
            random_angle_max_difference_list = torch.zeros(size=(len(roi_list), args.exp.layers)).cpu()
            with torch.no_grad():
                for images in target_dataloader:
                    images = images.to(device=device, dtype=current_dtype)
                    _, info = target_model.encoder_multilayer_information(images, target_layer=[i for i in range(args.exp.layers)])
                    for k in range(args.exp.layers):
                        current_layer_saes = saes_list[k]
                        activation = info[k].permute(1, 0, 2)[:, 1:, :]
                        activation = activation.to(dtype=sae_type)
                        middle_activation, std_info = current_layer_saes.encode(activation)
                        
                        for j, roi_name in enumerate(roi_list):
                            roi_info = roi_feature_dict[roi_name][k]
                            if len(roi_info) < 0:
                                continue
                            copy_activation = middle_activation.detach().clone()
                            copy_activation[:, :, roi_info] = 0.
                            reconstruct_activation = current_layer_saes.decode(copy_activation, std_info)

                            # 两个损失的衡量指标，第一个是MSE loss，第二个是角度的损失
                            # 先前做的结果是，取mean，下面进行更改，将mean改为196 patch max
                            # 同时更改random normalize的计算方式，并且每次random都random 100次

                            normalized_mse_loss = ((reconstruct_activation - activation) ** 2).mean(dim=-1)
                            normalized_mse_loss = (normalized_mse_loss) / ((activation**2).mean(dim=-1) + 1e-8)
                            normalized_max_mse_loss, _ = torch.max(normalized_mse_loss, dim=-1)
                            normalized_max_mse_loss = normalized_max_mse_loss.sum().detach().cpu().item()
                            normalized_mean_mse_loss = (normalized_mse_loss.sum() / activation.shape[1]).sum().detach().cpu().item()
                            cos_loss = F.cosine_similarity(reconstruct_activation, activation, dim=-1)
                            cos_max_loss, _ = torch.min(cos_loss, dim=-1)
                            cos_max_loss = cos_max_loss.sum().detach().cpu().item()
                            cos_mean_loss = (cos_loss.sum() / activation.shape[1]).detach().cpu().item()
                            random_max_cos_loss = 0.
                            random_mean_cos_loss = 0.
                            random_max_normalized_mse_loss = 0.
                            random_mean_normalized_mse_loss = 0.
                            for random_step in range(100):
                                # 随机ablation 100次，取100次的平均，这里同时计算所有patch上mean的平均和max的平均
                                random_roi_info = torch.randint(0, middle_activation.shape[-1], size=(roi_info.shape[-1], )).to(device=device)
                                random_copy_activation = middle_activation.detach().clone()
                                random_copy_activation[:, :, random_roi_info] = 0.
                                random_reconstruct_activation = current_layer_saes.decode(random_copy_activation, std_info)
                                random_normalized_mse_loss = ((random_reconstruct_activation - activation) ** 2).mean(dim=-1)
                                random_normalized_mse_loss = (random_normalized_mse_loss) / ((activation**2).mean(dim=-1) + 1e-8)
                                random_mean_normalized_mse_loss += (random_normalized_mse_loss.sum() / activation.shape[1]).detach().cpu().item() / 100
                                random_max_normalized_mse_loss_currently, _ = torch.max(random_normalized_mse_loss, dim=-1)
                                random_max_normalized_mse_loss += random_max_normalized_mse_loss_currently.sum().detach().cpu().item() / 100
                                random_cos_loss = F.cosine_similarity(random_reconstruct_activation, activation, dim=-1)
                                random_mean_cos_loss += (random_cos_loss.sum() / activation.shape[1]).detach().cpu().item() / 100
                                random_max_cos_loss_currently, _ = torch.min(random_cos_loss, dim=-1)
                                random_max_cos_loss += random_max_cos_loss_currently.sum().detach().cpu().item() / 100

                            MSE_max_difference_list[j, k] = MSE_max_difference_list[j, k] + normalized_max_mse_loss
                            MSE_mean_difference_list[j, k] = MSE_mean_difference_list[j, k] + normalized_mean_mse_loss
                            angle_max_difference_list[j, k] = angle_max_difference_list[j, k] + cos_max_loss
                            angle_mean_difference_list[j, k] = angle_mean_difference_list[j, k] + cos_mean_loss
                            random_MSE_mean_difference_list[j, k] = random_MSE_mean_difference_list[j, k] + random_mean_normalized_mse_loss
                            random_angle_mean_difference_list[j, k] = random_angle_mean_difference_list[j, k] + random_mean_cos_loss
                            random_MSE_max_difference_list[j, k] = random_MSE_max_difference_list[j, k] + random_max_normalized_mse_loss
                            random_angle_max_difference_list[j, k] = random_angle_max_difference_list[j, k] + random_max_cos_loss
            MSE_max_difference_list = MSE_max_difference_list / len(target_dataset)
            angle_max_difference_list = angle_max_difference_list / len(target_dataset)
            MSE_mean_difference_list = MSE_mean_difference_list / len(target_dataset)
            angle_mean_difference_list = angle_mean_difference_list / len(target_dataset)
            random_MSE_max_difference_list = random_MSE_max_difference_list / len(target_dataset)
            random_angle_max_difference_list = random_angle_max_difference_list / len(target_dataset)
            random_MSE_mean_difference_list = random_MSE_mean_difference_list / len(target_dataset)
            random_angle_mean_difference_list = random_angle_mean_difference_list / len(target_dataset)
            
            current_save_root = all_ablation_save_root.format(model_name, class_labels[i])
            check_path(current_save_root)
            torch.save([MSE_max_difference_list, random_MSE_max_difference_list, 
                        MSE_mean_difference_list, random_MSE_mean_difference_list, 
                        angle_max_difference_list, random_angle_max_difference_list,
                        random_angle_mean_difference_list, random_angle_mean_difference_list], current_save_root)


def data_evaluation_visualization():
    # 首先可视化，每个类别，被对应脑区选择出来的稳定的特征有多少，绘制折线图
    roi_save_info_dict_save_root = "/home/brainai1/VDisk2/BrainAi1/VLM-Memory/experiments/output/roi_save_info_dict_with_others.pt"
    roi_save_info_dict = torch.load(roi_save_info_dict_save_root, weights_only=False)

    roi_list = ["FFA", "EBA", "RSC", "FOOD", "VWFA"]
    model_list = ["clip_vit-b_16", "imagenet", "dinov2", "mae"]
    layer_list = [i for i in range(12)]

    roi_color_list = [
        "#FFA500", 
        "#FFB7DD", 
        "#FFFF77", 
        "#00AAAA", 
        "#BBFFEE",
    ]

    roi_face_color_list = [
        "#FFA500", 
        "#FFB7DD", 
        "#CCCC44",
        "#00AAAA", 
        "#88CCBB",
    ]
    roi_line_color_list = [
        "#EAB76A",
        "#F1C6DB",
        "#FFFF77",
        "#66CCCC",
        "#BBFFEE",
    ]
    
    for model_name in model_list:
        roi_rate_list = []
        roi_number_list = []
        for j, roi_name in enumerate(roi_list):
            layer_rate_list = []
            true_number_list = []
            for layer in layer_list:
                current_layer_info = roi_save_info_dict[roi_name][model_name][layer]
                selected_feature_number = 0
                feature_id_list = []
                for feature_id, feature_info in current_layer_info.items():
                    classes_info = feature_info[0]
                    classes_info = (torch.tensor(classes_info)[:5] > 10)
                    selected_feature_number += classes_info.sum().item()
                    if classes_info.sum() > 0 and classes_info[j] > 0:
                        feature_id_list.append(feature_id)
                # 这里首先查看，每个ROI，每层选择出来的符合功能的稳定特征占比多少，这里可以说是选择的准确率
                true_feature_number = len(feature_id_list)
                layer_rate_list.append(true_feature_number / (selected_feature_number + 1e-8))
                true_number_list.append(true_feature_number)
            # 将layer_rate_list 进行打印
            # print("current model:", model_name, "current roi:", roi_name, "layer_rate_list:", layer_rate_list, "ture_number_list:", true_number_list)
            print("current model:", model_name, "max selection:", max(layer_rate_list), "max selected number:", max(true_number_list), "max selected number rate:", layer_rate_list[true_number_list.index(max(true_number_list))])
            # 同时需要进行折线图的绘制，一个模型的rate画一张折线图，稳定数量画一个折线图，五个脑区对应五条线，五个颜色
            # 具体的颜色与PPT中的颜色保持一致

            roi_rate_list.append(layer_rate_list)
            roi_number_list.append(true_number_list)
        # 将roi_rate_list 进行打印
        # 两个图，第一个是rate的折线图，第二个是数量的折线图

        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.set_box_aspect(1)
        for i, roi_name in enumerate(roi_list):
            if i == 3:
                plt.plot(
                    layer_list,
                    roi_rate_list[i + 1],
                    marker="D",
                    markersize=6,
                    markeredgecolor='black',
                    markeredgewidth=0.5,
                    markerfacecolor=roi_face_color_list[i + 1],
                    linestyle='-', 
                    linewidth=2,
                    label=roi_list[i + 1],
                    color=roi_line_color_list[i + 1],
                )
            elif i == 4:
                plt.plot(
                    layer_list,
                    roi_rate_list[i - 1],
                    marker="D",
                    markersize=6,
                    markeredgecolor='black',
                    markeredgewidth=0.5,
                    markerfacecolor=roi_face_color_list[i - 1],
                    linestyle='-', 
                    linewidth=2,
                    label=roi_list[i - 1],
                    color=roi_line_color_list[i - 1],
                )
            else:
                plt.plot(
                    layer_list,
                    roi_rate_list[i],
                    marker="D",
                    markersize=6,
                    markeredgecolor='black',
                    markeredgewidth=0.5,
                    markerfacecolor=roi_face_color_list[i],
                    linestyle='-', 
                    linewidth=2,
                    label=roi_list[i],
                    color=roi_line_color_list[i],
                )
        # plt.xlabel("Layer")
        # plt.ylabel("Stable feature rate")
        # plt.title(f"Stable Feature Rate vs Layer ({model_name})")
        plt.xticks(layer_list)
        plt.grid(True, alpha=0.5)
        ax = plt.gca()
        ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
        # plt.legend()
        plt.tight_layout()
        stable_feature_rate_save_root = f"experiments/paper_image/selectivity_stability_evaluation/{model_name}_stable_feature_rate.png"
        check_path(stable_feature_rate_save_root)
        plt.savefig(stable_feature_rate_save_root, dpi=300)
        plt.close()

        # -----------------------------
        # 画图 2：数量 折线图（一个模型一张图，五个 ROI 五条线）
        # -----------------------------
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.set_box_aspect(1)
        for i, roi_name in enumerate(roi_list):
            if i == 3:
                plt.plot(
                    layer_list,
                    roi_number_list[i + 1],
                    marker="D",
                    markersize=6,
                    markeredgecolor='black',
                    markeredgewidth=0.5,
                    markerfacecolor=roi_face_color_list[i + 1],
                    linestyle='-', 
                    linewidth=2,
                    label=roi_list[i + 1],
                    color=roi_line_color_list[i + 1],
                )
            elif i == 4:
                plt.plot(
                    layer_list,
                    roi_number_list[i - 1],
                    marker="D",
                    markersize=6,
                    markeredgecolor='black',
                    markeredgewidth=0.5,
                    markerfacecolor=roi_face_color_list[i - 1],
                    linestyle='-', 
                    linewidth=2,
                    label=roi_list[i - 1],
                    color=roi_line_color_list[i - 1],
                )
            else:
                plt.plot(
                    layer_list,
                    roi_number_list[i],
                    marker="D",
                    markersize=6,
                    markeredgecolor='black',
                    markeredgewidth=0.5,
                    markerfacecolor=roi_face_color_list[i],
                    linestyle='-', 
                    linewidth=2,
                    label=roi_list[i],
                    color=roi_line_color_list[i],
                )
        # plt.xlabel("Layer")
        # plt.ylabel("Stable feature count")
        # plt.title(f"Stable Feature Count vs Layer ({model_name})")
        plt.xticks(layer_list)
        plt.grid(True, alpha=0.5)
        ax = plt.gca()
        ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
        # plt.legend()
        plt.tight_layout()
        stable_feature_count_save_root = f"experiments/paper_image/selectivity_stability_evaluation/{model_name}_stable_feature_count.png"
        check_path(stable_feature_count_save_root)
        plt.savefig(stable_feature_count_save_root, dpi=300)
        plt.savefig(stable_feature_count_save_root, dpi=300)
        plt.close()


def ablation_result_visualization():
    # 这个是消融实验的可视化结果，以及能够生成一个表格
    # 这里主要针对的是ablation的normalized MSE的结果可视化
    class_labels = ["adult", "body", "car", "child", "corridor", "food", "house", "instrument", "limb", "number", "word"]
    classes_ids = [0, 1, 2, 0, 2, 3, 2, 2, 1, 4, 4]
    classes_number = [2, 2, 4, 1, 2]
    roi_list = ["FFA", "EBA", "RSC", "FOOD", "VWFA"]
    model_list = ["clip_vit-b_16", "imagenet", "dinov2", "mae"]
    all_ablation_save_root = "/home/brainai1/VDisk2/BrainAi1/VLM-Memory/experiments/output/ablation/{}/{}_mean_and_max_ablation_result.pt"
    for model_name in model_list:
        # 最后的保存结果是，每个模型有五组信息，每组信息里面包含六个柱状图，分别代表着五个脑区和random的ablation结果
        max_plot_info_list = torch.zeros(size=(5, 6))
        mean_plot_info_list = torch.zeros(size=(5, 6))
        for i, class_label in enumerate(class_labels):
            print(class_label)
            current_info = torch.load(all_ablation_save_root.format(model_name, class_label), weights_only=False)
            max_mse_loss, max_random_mse_loss, _, _, _, _, _, _ = current_info
            # 这些loss的形状都为[5 * 12]
            # 首先来绘制Max MSE loss的柱状图
            current_class_max_random_mse_loss = max_random_mse_loss.mean(dim=0).max(dim=-1).values.view(1, -1)
            current_max_mse_loss = max_mse_loss.max(dim=-1).values.view(-1)
            print(current_max_mse_loss)
            max_plot_info_list[classes_ids[i], :5] = max_plot_info_list[classes_ids[i], :5] + current_max_mse_loss / classes_number[classes_ids[i]]
            max_plot_info_list[classes_ids[i], 5] = max_plot_info_list[classes_ids[i], 5] + current_class_max_random_mse_loss / classes_number[classes_ids[i]]

            # 下面来绘制mean MSE loss的柱状图
            current_class_mean_random_mse_loss = max_random_mse_loss.mean().view(1, -1)
            current_mean_mse_loss = max_mse_loss.mean(dim=-1).view(-1)
            mean_plot_info_list[classes_ids[i], :5] = mean_plot_info_list[classes_ids[i], :5] + current_mean_mse_loss / classes_number[classes_ids[i]]
            mean_plot_info_list[classes_ids[i], 5] = mean_plot_info_list[classes_ids[i], 5] + current_class_mean_random_mse_loss / classes_number[classes_ids[i]]
        raise RuntimeError()
        # 下面是画图
        
        max_mse_loss_all_layer_max_ablation_result_save_root = f"experiments/paper_image/ablation_evaluation/{model_name}/max_mse_loss_max_ablation_result.png"
        max_mse_loss_all_layer_mean_ablation_result_save_root = f"experiments/paper_image/ablation_evaluation/{model_name}/max_mse_loss_mean_ablation_result.png"
        check_path(max_mse_loss_all_layer_max_ablation_result_save_root)
        check_path(max_mse_loss_all_layer_mean_ablation_result_save_root)

        bar_plot(max_plot_info_list.T.cpu().numpy(), max_mse_loss_all_layer_max_ablation_result_save_root)
        bar_plot(mean_plot_info_list.T.cpu().numpy(), max_mse_loss_all_layer_mean_ablation_result_save_root)

    for model_name in model_list:
        # 这部分是看均值情况下，结果如何
        max_plot_info_list = torch.zeros(size=(5, 6))
        mean_plot_info_list = torch.zeros(size=(5, 6))
        for i, class_label in enumerate(class_labels):
            current_info = torch.load(all_ablation_save_root.format(model_name, class_label), weights_only=False)
            _, _, mean_mse_loss, mean_random_mse_loss, _, _, _, _ = current_info
            # 这些loss的形状都为[5 * 12]
            # 首先来绘制Max MSE loss的柱状图
            current_class_max_random_mse_loss = mean_random_mse_loss.mean(dim=0).max(dim=-1).values.view(1, -1)
            current_max_mse_loss = mean_mse_loss.max(dim=-1).values.view(-1)
        
            max_plot_info_list[classes_ids[i], :5] = mean_plot_info_list[classes_ids[i], :5] + current_max_mse_loss / classes_number[classes_ids[i]]
            max_plot_info_list[classes_ids[i], 5] = mean_plot_info_list[classes_ids[i], 5] + current_class_max_random_mse_loss / classes_number[classes_ids[i]]

            # 下面来绘制mean MSE loss的柱状图
            current_class_mean_random_mse_loss = mean_random_mse_loss.mean().view(1, -1)
            current_mean_mse_loss = mean_mse_loss.mean(dim=-1).view(-1)
            mean_plot_info_list[classes_ids[i], :5] = mean_plot_info_list[classes_ids[i], :5] + current_mean_mse_loss / classes_number[classes_ids[i]]
            mean_plot_info_list[classes_ids[i], 5] = mean_plot_info_list[classes_ids[i], 5] + current_class_mean_random_mse_loss / classes_number[classes_ids[i]]

        # 下面是画图
        mean_mse_loss_all_layer_max_ablation_result_save_root = f"experiments/paper_image/ablation_evaluation/{model_name}/mean_mse_loss_max_ablation_result.png"
        mean_mse_loss_all_layer_mean_ablation_result_save_root = f"experiments/paper_image/ablation_evaluation/{model_name}/mean_mse_loss_mean_ablation_result.png"
        check_path(mean_mse_loss_all_layer_max_ablation_result_save_root)
        check_path(mean_mse_loss_all_layer_mean_ablation_result_save_root)
        bar_plot(max_plot_info_list.T.cpu().numpy(), mean_mse_loss_all_layer_max_ablation_result_save_root)
        bar_plot(mean_plot_info_list.T.cpu().numpy(), mean_mse_loss_all_layer_mean_ablation_result_save_root)
    
    for model_name in model_list:
        # 角度的结果
        max_plot_info_list = torch.zeros(size=(5, 6))
        mean_plot_info_list = torch.zeros(size=(5, 6))
        for i, class_label in enumerate(class_labels):
            current_info = torch.load(all_ablation_save_root.format(model_name, class_label), weights_only=False)
            _, _, _, _, max_angle_loss, max_random_angle_loss, _, _ = current_info
            max_angle_loss = 1 - max_angle_loss
            max_random_angle_loss = 1 - max_random_angle_loss
            # 这些loss的形状都为[5 * 12]
            # 首先来绘制Max MSE loss的柱状图
            current_class_max_random_mse_loss = max_random_angle_loss.mean(dim=0).max(dim=-1).values.view(1, -1)
            current_max_mse_loss = max_angle_loss.max(dim=-1).values.view(-1)
        
            max_plot_info_list[classes_ids[i], :5] = max_plot_info_list[classes_ids[i], :5] + current_max_mse_loss / classes_number[classes_ids[i]]
            max_plot_info_list[classes_ids[i], 5] = max_plot_info_list[classes_ids[i], 5] + current_class_max_random_mse_loss / classes_number[classes_ids[i]]

            # 下面来绘制mean MSE loss的柱状图
            current_class_mean_random_mse_loss = max_random_angle_loss.mean().view(1, -1)
            current_mean_mse_loss = max_angle_loss.mean(dim=-1).view(-1)
            mean_plot_info_list[classes_ids[i], :5] = mean_plot_info_list[classes_ids[i], :5] + current_mean_mse_loss / classes_number[classes_ids[i]]
            mean_plot_info_list[classes_ids[i], 5] = mean_plot_info_list[classes_ids[i], 5] + current_class_mean_random_mse_loss / classes_number[classes_ids[i]]

        # 下面是画图
        max_angle_loss_all_layer_max_ablation_result_save_root = f"experiments/paper_image/ablation_evaluation/{model_name}/max_angle_loss_max_ablation_result.png"
        max_angle_loss_all_layer_mean_ablation_result_save_root = f"experiments/paper_image/ablation_evaluation/{model_name}/max_angle_loss_mean_ablation_result.png"
        check_path(max_angle_loss_all_layer_max_ablation_result_save_root)
        check_path(max_angle_loss_all_layer_mean_ablation_result_save_root)
        bar_plot(max_plot_info_list.T.cpu().numpy(), max_angle_loss_all_layer_max_ablation_result_save_root)
        bar_plot(mean_plot_info_list.T.cpu().numpy(), max_angle_loss_all_layer_mean_ablation_result_save_root)

    for model_name in model_list:
        # 这部分是看均值情况下，结果如何
        max_plot_info_list = torch.zeros(size=(5, 6))
        mean_plot_info_list = torch.zeros(size=(5, 6))
        for i, class_label in enumerate(class_labels):
            current_info = torch.load(all_ablation_save_root.format(model_name, class_label), weights_only=False)
            _, _, _, _, _, _, mean_angle_loss, mean_random_angle_loss = current_info
            mean_angle_loss = 1 - mean_angle_loss
            mean_random_angle_loss = 1 - mean_random_angle_loss
            # 这些loss的形状都为[5 * 12]
            # 首先来绘制Max MSE loss的柱状图
            current_class_max_random_mse_loss = mean_random_angle_loss.mean(dim=0).max(dim=-1).values.view(1, -1)
            current_max_mse_loss = mean_angle_loss.max(dim=-1).values.view(-1)
        
            max_plot_info_list[classes_ids[i], :5] = mean_plot_info_list[classes_ids[i], :5] + current_max_mse_loss / classes_number[classes_ids[i]]
            max_plot_info_list[classes_ids[i], 5] = mean_plot_info_list[classes_ids[i], 5] + current_class_max_random_mse_loss / classes_number[classes_ids[i]]

            # 下面来绘制mean MSE loss的柱状图
            current_class_mean_random_mse_loss = mean_random_angle_loss.mean().view(1, -1)
            current_mean_mse_loss = mean_angle_loss.mean(dim=-1).view(-1)
            mean_plot_info_list[classes_ids[i], :5] = mean_plot_info_list[classes_ids[i], :5] + current_mean_mse_loss / classes_number[classes_ids[i]]
            mean_plot_info_list[classes_ids[i], 5] = mean_plot_info_list[classes_ids[i], 5] + current_class_mean_random_mse_loss / classes_number[classes_ids[i]]

        # 下面是画图
        mean_angle_loss_all_layer_max_ablation_result_save_root = f"experiments/paper_image/ablation_evaluation/{model_name}/mean_angle_loss_max_ablation_result.png"
        mean_angle_loss_all_layer_mean_ablation_result_save_root = f"experiments/paper_image/ablation_evaluation/{model_name}/mean_angle_loss_mean_ablation_result.png"
        check_path(mean_angle_loss_all_layer_max_ablation_result_save_root)
        check_path(mean_angle_loss_all_layer_mean_ablation_result_save_root)
        bar_plot(max_plot_info_list.T.cpu().numpy(), mean_angle_loss_all_layer_max_ablation_result_save_root)
        bar_plot(mean_plot_info_list.T.cpu().numpy(), mean_angle_loss_all_layer_mean_ablation_result_save_root)

def bar_plot(plot_info_list, save_name):
    # 这里输入的时候，记得转置
    group_names = ["Faces", "Bodies", "Places", "Food", "Words"]
    bar_label_names = ["FFA", "EBA", "RSC", "VWFA", "FOOD", "Random"]
    roi_color_list = [
        "#FFA500", 
        "#FFB7DD", 
        "#CCCC44",
        "#00AAAA", 
        "#88CCBB",
        "#FF6969",
    ]

    x = np.arange(5)          # 5个组中心位置
    bar_w = 0.12              # 单根柱子的宽度（可调）
    offsets = (np.arange(6) - (5)/2) * bar_w  # 让6根柱子围绕组中心对称展开

    for i in range(6):
        if i == 3:
            plt.bar(x + offsets[i], plot_info_list[i + 1], width=bar_w, color=roi_color_list[i + 1], label=bar_label_names[i])
        elif i == 4:
            plt.bar(x + offsets[i], plot_info_list[i - 1], width=bar_w, color=roi_color_list[i - 1], label=bar_label_names[i])
        else:
            plt.bar(x + offsets[i], plot_info_list[i], width=bar_w, color=roi_color_list[i], label=bar_label_names[i])
    plt.xticks(x, group_names)
    # plt.xlabel("Category (5 groups)")
    # plt.ylabel("Value")
    # plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    plt.savefig(save_name, dpi=300)
    plt.close()