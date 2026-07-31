from sklearn.decomposition import PCA
import numpy as np
import torch
import matplotlib.pyplot as plt


def PCAVisualize(
        data: torch.Tensor, 
        n_components=2, 
        save_root = "pca.png",
    ):
    """
    PCAVisualize

    args:
    - data: torch.tensor, shape (n_samples, n_features)
    - n_components: dimention after PCA projection

    returns:
    - low_dimensional_data: torch.tensor, shape (n_samples, n_components)
    """
    
    assert len(data.shape) == 2, "data must be a 2D tensor"

    reducer = PCA(n_components=n_components)
    
    low_dimensional_data = reducer.fit_transform(data.numpy())
    low_dimensional_data = torch.from_numpy(low_dimensional_data)


    plt.hexbin(low_dimensional_data[:, 0], low_dimensional_data[:, 1], gridsize=200, cmap="viridis", bins="log")
    plt.colorbar()
    plt.savefig(save_root)
    plt.close()

    return low_dimensional_data