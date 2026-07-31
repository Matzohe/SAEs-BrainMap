"This scripts visualize prediction performance with pycortex."
from nibabel.volumeutils import working_type
import numpy as np
import cortex
from numpy.core.fromnumeric import nonzero
from tqdm import tqdm


def project_vals_to_3d(vals, mask):
    all_vals = np.zeros(mask.shape)
    all_vals[mask] = vals
    all_vals = np.swapaxes(all_vals, 0, 2)
    return all_vals

def make_volume(
    vals,
    subj,
    cortical_mask,
    roi_in_nsdgeneral_mask,
    roi_to_nsdgeneral_index,
    measure="corr",
    noise_corrected=False,
    cmap="hot", #BuGn
    vmin=0,
    vmax=None,
):
    if vmax is None:
        if measure == "rsq":
            vmax = 0.6
        else:
            vmax = 1
        if noise_corrected:
            vmax = 0.85
        if measure == "pvalue":
            vmax = 0.06

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )
    print("max:" + str(max(vals[~np.isnan(vals)])))
    val = np.full((sum(cortical_mask.flatten())),np.nan) / max(vals[~np.isnan(vals)])
    val[roi_in_nsdgeneral_mask] = vals[roi_to_nsdgeneral_index]
    all_vals = project_vals_to_3d(val, cortical_mask)
    vol_data = cortex.dataset.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return vol_data