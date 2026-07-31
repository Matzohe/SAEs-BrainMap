import numpy as np
import torch
from tqdm import tqdm


def r2_score(Real, Pred):
    SSres = torch.mean((Real - Pred) ** 2, dim=0)
    SStot = torch.var(Real, dim=0)
    return torch.nan_to_num(1 - SSres / SStot)

# Specialize for NSD Datasets
def zscore_by_run(mat, run_n=480):
    from scipy.stats import zscore

    run_n = np.ceil(
        mat.shape[0] / 62.5
    )  # should be 480 for subject with full experiment\
    zscored_mat = np.zeros(mat.shape)
    index_so_far = 0
    for i in tqdm(range(int(run_n)), desc="NSD dataset zscore processing..."):
        if i % 2 == 0:
            zscore_value = np.nan_to_num(zscore(mat[index_so_far : index_so_far + 62, :]), nan=0.0, posinf=0.0, neginf=0.0)
            zscored_mat[index_so_far : index_so_far + 62, :] = zscore_value
            index_so_far += 62
        else:
            zscore_value = np.nan_to_num(zscore(mat[index_so_far : index_so_far + 63, :]), nan=0.0, posinf=0.0, neginf=0.0)
            zscored_mat[index_so_far : index_so_far + 63, :] = zscore_value
            index_so_far += 63

    return zscored_mat

def ev(data, biascorr=True):
    """
    Computes the amount of variance in a voxel's response that can be explained by the
    mean response of that voxel over multiple repetitions of the same stimulus.

    If [biascorr], the explainable variance is corrected for bias, and will have mean zero
    for random datasets.

    Data is assumed to be a 2D matrix: time x repeats.
    """
    ev = 1 - torch.var(data.T - torch.nanmean(data, axis=1)) / torch.var(data)
    if biascorr:
        return ev - ((1 - ev) / (data.shape[1] - 1.0))
    else:
        return ev
    
def load_target_roi_mask(root):
    roi_index_tensor = torch.from_numpy(np.loadtxt(root, dtype=np.float32, delimiter=",")) > 0
    return roi_index_tensor