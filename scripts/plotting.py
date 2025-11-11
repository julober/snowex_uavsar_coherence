import math 
import rioxarray as rxa
import matplotlib.pyplot as plt
import pandas as pd

def plot_tifs_grid(tif_inputs, 
                   is_file=True,
                   titles=None) :
    """
    Plots a list of .tif files or arrays in a grid (3 columns), with a shared colorbar.

    Parameters:
    - tif_inputs: list of file paths (if is_file=True) or list of 2D numpy arrays (if is_file=False)
    - is_file: boolean, set to True if tif_inputs is a list of .tif file paths, False if they are already arrays
    """
    
    # Load data arrays if file paths are provided
    if is_file:
        arrays = [rxa.open_rasterio(fp)[0].values for fp in tif_inputs]
        if (titles == None) : 
            titles = [f"Image pair {i+1}" for i in range(len(tif_inputs))]  # use file name as title
    else:
        arrays = tif_inputs
        if (titles == None) : 
            titles = [f"Image pair {i+1}" for i in range(len(arrays))]

    n_images = len(arrays)
    n_cols = 3
    n_rows = math.ceil(n_images / n_cols)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), constrained_layout=True)
    axes = axes.flatten()  # flatten in case it's 2D

    # vmin/vmax for coherence
    vmin, vmax = 0, 1

    # Plot each image
    imgs = []
    for i, (arr, title) in enumerate(zip(arrays, titles)):
        img = axes[i].imshow(arr, cmap='magma', vmin=vmin, vmax=vmax)
        axes[i].set_title(title, fontsize=14)
        axes[i].axis('off')
        imgs.append(img)

    # Turn off unused axes
    for j in range(n_images, len(axes)):
        axes[j].axis('off')

    # Add one shared colorbar
    cbar = fig.colorbar(imgs[0], ax=axes[:n_images], orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Coherence', fontsize=12)

    plt.show()

def plot_coherence_matrix(coh_matrix, dates):
    """
    Plots a coherence matrix with dates as labels.

    Parameters:
    - coh_matrix: 2D numpy array representing the coherence matrix
    - dates: list of date strings corresponding to the coherence matrix axes
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(coh_matrix, cmap='viridis', vmin=0, vmax=1)

    dates_dt = pd.to_datetime(dates)

    # Set ticks and labels
    ax.set_xticks(range(len(dates)))
    ax.set_yticks(range(len(dates)))
    ax.set_xticklabels([d.strftime('%y/%m/%d') for d in dates_dt], rotation=90)
    ax.set_yticklabels([d.strftime('%y/%m/%d') for d in dates_dt])

    # Add colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label('Coherence', fontsize=12)

    plt.title('Coherence Matrix', fontsize=16)
    plt.xlabel('Dates', fontsize=14)
    plt.ylabel('Dates', fontsize=14)
    plt.show()
