import os
import numpy as np
from dataclasses import dataclass, make_dataclass, field
from typing import List, Dict
import torch
from .volume_visualize import make_volume, project_vals_to_3d
from ....dataset.NSD.NSDDataLoader import NSDDataset
from ....util import check_path
import cortex
import PIL.Image as Image
import shutil
import nibabel as nib
import configparser
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
from easydict import EasyDict

def fdr_correct_p(var):
    from statsmodels.stats.multitest import fdrcorrection
    n = var.shape[0]
    p_vals = np.sum(var < 0, axis=0) / n  # proportions of permutation below 0
    fdr_p = fdrcorrection(p_vals)  # corrected p
    return fdr_p

def Norm(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def visualize(
        args: EasyDict, 
        target_discription: str, 
        visual_activation: np.ndarray, 
        cmap: str = "hot",
    ):

    visual_save_path = args.visualize.cortex_visualize_save_path.format(target_discription)
    check_path(visual_save_path)
    # target_discription应该包括被试，模型名称，roi
    target_roi = args.exp.full_roi
    subj = args.exp.subj
    roi_mask_root = args.NSD.roi_mask_save_root.format(args.exp.subj, target_roi)
    roi_mask = torch.load(roi_mask_root, weights_only=False)
    roi_index = torch.arange(roi_mask.flatten().shape[0])[roi_mask.flatten()]
    target_vals = torch.full(roi_mask.shape, float("nan")).flatten().numpy()    
    target_vals[roi_index] = visual_activation

    roi_in_nsdgeneral_mask = np.load("%s/%s_in_nsdgeneral_mask_subj%02d.npy"%(args.NSD.visualize_utils_root.format(subj), target_roi, subj))
    roi_to_nsdgeneral_index = np.load("%s/%s_to_nsdgeneral_index_subj%02d.npy"%(args.NSD.visualize_utils_root.format(subj), target_roi, subj))
    
    cortical_mask = np.load("%s/cortical_mask_subj%02d.npy" % (args.NSD.visualize_utils_root.format(subj), subj))

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    val = np.full((sum(cortical_mask.flatten())),np.nan)
    val[roi_in_nsdgeneral_mask > 0] = visual_activation[roi_to_nsdgeneral_index]
    val = project_vals_to_3d(val, cortical_mask)

    vmin = float(np.min(visual_activation))
    vmax = float(np.max(visual_activation))

    

    vol_data = cortex.dataset.Volume(
        val,
        "subj%02d" % args.exp.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # cortex.webgl.show(vol_data, open_browser=True, port=8080)
    # input("Press Enter to stop server...")

    _ = cortex.quickflat.make_png( 
            visual_save_path,
            vol_data,
            labelsize="20pt",
            with_curvature=True,
            recache=False,
            with_labels=True,
            with_colorbar=False,
            dpi=5000,
            vmax = vmax,
            vmin = vmin,
        )