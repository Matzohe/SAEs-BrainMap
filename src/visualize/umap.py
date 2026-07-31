import torch
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

UMAPCOLORLIST = torch.tensor([[0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [255, 0, 0]], dtype=torch.float32) / 255. # green, blue, yellow, magenta


def UMAPProcess(data, n_neighbors=50, min_dist=0.7, n_components=2):
    """
    UMAPProcess

    args:
    - data: torch.tensor, shape (n_samples, n_features)
    - n_neighbors: the number of neighbors to consider for UMAP
    - min_dist: the min distance between two samples for UMAP
    - n_components: dimention after UMAP projection

    returns:
    - low_dimensional_data: torch.tensor, shape (n_samples, n_components)
    """
    
    assert len(data.shape) == 2, "data must be a 2D tensor"
    original_device = data.device
    data = data.to('cpu').numpy() 

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', n_components=n_components, n_jobs=-1) # , metric='cosine',
    
    low_dimensional_data = reducer.fit_transform(data)
    low_dimensional_data = torch.from_numpy(low_dimensional_data).to(original_device)

    return low_dimensional_data


def UMAPVisualize(data, n_neighbors=15, min_dist=0.1, n_components=2, 
                  show=False, save=False, save_path=None, color_list=None):

    """
    UMAPVisualize

    args:
    - data: torch.tensor, shape (n_samples, n_features)
    - labels: torch.tensor, shape (n_samples,)
    - n_neighbors: the number of neighbors to consider for UMAP 
    - min_dist: the min distance between two samples for UMAP
    - n_components: dimention after UMAP projection
    - show: whether to show the plot
    - save: whether to save the plot
    - save_path: the path to save the plot

    returns:    
    - low_dimensional_data: torch.tensor, shape (n_samples, n_components), the UMAP features
    - low_dimensional_data_color: torch.tensor, shape (n_samples, 3). the color for umap features
    """

    assert n_components <= 5, "Too much components not support this visualize function, this visualiize function only support 4 components"

    low_dimensional_data = UMAPProcess(data, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    low_dimensional_data = low_dimensional_data.detach().cpu()

    # translate the UMAP value to colors
    low_dimensional_data_color = torch.abs(low_dimensional_data) / torch.sum(torch.abs(low_dimensional_data), dim=1, keepdim=True)
    low_dimensional_data_color = low_dimensional_data_color.detach().cpu()
    low_dimensional_data_color = low_dimensional_data_color @ UMAPCOLORLIST[: n_components]
    if color_list is not None:
        low_dimensional_data_color = color_list

    if show:
        if low_dimensional_data.shape[1] == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hexbin(low_dimensional_data[:, 0], low_dimensional_data[:, 1], gridsize=50, cmap="viridis", bins="log")
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            if save:
                plt.savefig(save_path)
            else:
                plt.show()

        elif low_dimensional_data.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(low_dimensional_data[:, 0], low_dimensional_data[:, 1], low_dimensional_data[:, 2], 
                       c=low_dimensional_data_color, s=0.05)
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.set_zlabel("UMAP3")

            if save:
                plt.savefig(save_path)
            else:
                plt.show()
        
        elif low_dimensional_data.shape[1] == 4:
            UMAPVisualizeSecializedFor4D(low_dimensional_data, low_dimensional_data_color, s=0.05, save=save, save_path=save_path)
        
        elif low_dimensional_data.shape[1] == 5:
            UMAPVisualizeSecializedFor5D(low_dimensional_data, low_dimensional_data_color, s=0.05, save=save, save_path=save_path)

    return low_dimensional_data, low_dimensional_data_color

    
def UMAPVisualizeSecializedFor4D(data, color, s, save, save_path):
    
    labels = ["UMAP1", "UMAP2", "UMAP3", "UMAP4"]


    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(231)
    ax.scatter(data[:, 0], data[:, 1], c=color, s=s)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    ax = fig.add_subplot(232)
    ax.scatter(data[:, 0], data[:, 2], c=color, s=s)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[2])

    ax = fig.add_subplot(233)
    ax.scatter(data[:, 0], data[:, 3], c=color, s=s)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[3])

    ax = fig.add_subplot(234)
    ax.scatter(data[:, 1], data[:, 2], c=color, s=s)
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[2])

    ax = fig.add_subplot(235)
    ax.scatter(data[:, 1], data[:, 3], c=color, s=s)
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[3])

    ax = fig.add_subplot(236)
    ax.scatter(data[:, 2], data[:, 3], c=color, s=s)
    ax.set_xlabel(labels[2])
    ax.set_ylabel(labels[3])
    if save:
        plt.savefig(save_path)
    else:
        plt.show()


def UMAPVisualizeSecializedFor5D(data, color, s, save, save_path):
    fig, axes = plt.subplots(2, 5, figsize=(12, 20))
    labels = ["UMAP1", "UMAP2", "UMAP3", "UMAP4", "UMAP5"]
    data_pairs = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 3],
        [2, 4],
        [3, 4],
    ]
    for i, ax in enumerate(axes.flat):
        ax.scatter(data[:, data_pairs[i][0]], data[:, data_pairs[i][1]], c=color, s=s)
        ax.set_xlabel(labels[data_pairs[i][0]])
        ax.set_ylabel(labels[data_pairs[i][1]])

    if save:
        plt.savefig(save_path)
    else:
        plt.show()