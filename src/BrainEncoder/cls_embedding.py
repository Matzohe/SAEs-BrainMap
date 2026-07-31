import torch
from tqdm import tqdm
from ..util import check_path


def save_cls_embedding(target_model, dataloader, save_path, device):

    cls_embedding = []
    for images, _, _ in tqdm(dataloader, total=len(dataloader), desc="Extracting cls embedding"):
        images = images.to(device=device)
        with torch.no_grad():
            image_embedding = target_model.encode_image(images)
        cls_embedding.append(image_embedding)
    cls_embedding = torch.cat(cls_embedding, dim=0).cpu()
    check_path(save_path)
    torch.save(cls_embedding, save_path)
    return cls_embedding
