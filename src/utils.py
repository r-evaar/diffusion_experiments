from math import ceil
from matplotlib import pyplot as plt
from pathlib import Path
from torch import nn, save
import torchinfo
import itertools
import pandas as pd
import seaborn as sn
import requests
import torch
import os
from tqdm.auto import tqdm
from datasets import load_dataset, logging

"""
File containing various utility functions for PyTorch model training.
"""


def save_model(model:nn.Module, directory:str='.', filename:str='model.pt'):
    """Saves a PyTorch model to a target directory

    Args:
        model: A target PyTorch model to save.
        directory: A directory for saving the model to.
        filename: A filename for the saved model. Should include either ".ppth" or ".pt" as the file extension.

    Example usage:
        save_model(
            model=model_0,
            directory="models",
            filename="my_model.pt"
    """

    assert filename.endswith('.pt') or filename.endswith('.pth'), \
        "filename should end with '.pt' or '.pth"
    
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / filename

    print(f"Saving model into '{save_path}'")
    save(obj=model.state_dict(), f=save_path)

    print(f"Total size = {save_path.stat().st_size / 1024**2:.2f} MB")
    return save_path


def plot_progress(progress: dict):
    fig, ax = plt.subplots(1,2)
    fig.set_figwidth(15)
    
    x_b = progress['batch']['x']
    x_t = progress['train']['x']
    x_v = progress['val']['x']

    ax[0].plot(x_b, progress['batch']['loss'], color='#704c70',  label='batch')
    ax[0].plot(x_t, progress['train']['loss'],
               markerfacecolor='none', markeredgecolor='#179ac2',
               linestyle='dashed', marker='o',  label='train')
    ax[0].plot(x_v, progress['val']['loss'],
               markerfacecolor='none', markeredgecolor='#735773',
               linestyle='dashed', marker='o',  label='val')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(x_b, progress['batch']['acc'], color='#59704c', label='batch')
    ax[1].plot(x_t, progress['train']['acc'],
               markerfacecolor='none', markeredgecolor='#179ac2',
               linestyle='dashed', marker='o',  label='train')
    ax[1].plot(x_v, progress['val']['acc'],
               markerfacecolor='none', markeredgecolor='#735773',
               linestyle='dashed', marker='o',  label='val')
    ax[1].set_title('Accuracy')
    ax[1].legend()

    return fig


def summary(input_size, **kw):

    def method(self):
        print(torchinfo.summary(
            self,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["depth"],
            depth=5,
            **kw
        ))
    return method


def prepare_experiments(search_space):
    keys = search_space.keys()
    all_combinations = itertools.product(*search_space.values())

    experiments_as_dicts = [
        dict(zip(keys, combination)) for combination in all_combinations
    ]

    return experiments_as_dicts


class ExpWrapper:

    def __init__(self, obj, expression):
        self.obj = obj
        self.exp = expression

    def __str__(self):
        return self.exp

    def __repr__(self):
        return self.__str__()

    def extract(self):
        return self.obj


def visualize(loader, n=5, classes=None):
    col = 3
    row = ceil(n / col)
    i = 0
    stop = False
    for x, y in loader:
        if stop: break
        for x_i, y_i in zip(x, y):
            if i >= n:
                stop = True
                break
            plt.subplot(row, col, i+1)
            plt.imshow(x_i.detach().cpu().permute(1, 2, 0))
            title = y_i.item()
            if classes:
                title = classes[title]
            plt.title(title)
            plt.axis('off')

            i += 1
    plt.show()


def plot_confusion_matrix(cm):
    classes = ['negative', 'positive']  # Negative first cuz index starts at 0
    df_cm = pd.DataFrame(cm.numpy(), index=classes, columns=classes)
    ax = sn.heatmap(df_cm, annot=True, fmt='.5f')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('target')
    plt.show()


def _write_to_file(f, response):
    block_size = 1024**2
    unit = 'MiB'
    total_size = round(int(response.headers.get("content-length", 0)) / block_size, 2)

    print(f'Total size = {total_size} {unit}')

    bar = tqdm(
        total=ceil(total_size), unit=unit,
        position=0, leave=True)

    for chunk in response.iter_content(chunk_size=block_size):
        f.write(chunk)
        bar.update()

    bar.close()
    assert bar.n == ceil(total_size), 'Download Failure: downloaded file is corrupted'


def download_file(url):

    data_path = Path('../data')
    filename = data_path / Path(url).name

    if filename.exists():
        print(f'[INFO] File \'{filename}\' already exists.')
        return filename

    response = requests.get(url)
    response.raise_for_status()

    data_path.mkdir(parents=True, exist_ok=True)
    with open(filename, 'wb') as f:
        _write_to_file(f, response)

    print(f'[INFO] Downloaded \'{filename}\'.')
    return filename


def clear_cuda_mem():
    torch.cuda.empty_cache()
    available, mx = [m/1024**3 for m in torch.cuda.mem_get_info()]
    print(f"Memory Usage [{mx-available:.2f}/{mx:.2f}]GB")


def load_and_split_dataset(dataframe, splits):
    assert sum(splits.values()) == 1, 'Sum of splits must be 1'
    assert all([0 < split < 1 for split in splits.values()]), 'Splits must be between 0 and 1'

    logging.disable_progress_bar()

    df = dataframe.copy()

    data_files = {}
    factor = 0
    for name, split in splits.items():
        # Calculate the next factor of dataset reduction (due to subset drop) to compensate the next split
        next_factor = factor + split

        # Extract a subset and drop it from the original dataset
        split = min(round(split / (1-factor), 5), 1)
        subset = df.sample(frac=split)
        df = df.drop(subset.index)

        # Save the subset to a temporary file
        filename = f'temp_{name}.csv'
        subset.to_csv(filename, index=False)
        data_files[name] = filename

        # Update the reduction factor
        factor = round(next_factor, 5)

    # Load the dataset
    raw_dataset = load_dataset('csv', data_files=data_files)

    # Delete the temporary files
    for name in data_files:
        os.remove(data_files[name])

    logging.enable_progress_bar()
    return raw_dataset

# Create a function to align targets to tokens
def align_to_ner_token(word_ids: list, ner_tags: list, rep_map: dict):

    aligned_target = []
    i_prev = -1
    for i in word_ids:
        if i is None:
            target = -100
        else:
            inside = i == i_prev    # Check if current word/phrase is part of (inside) the same tag
            target = ner_tags[i]    # Get IOB representation (target)
            eligible = target in rep_map.keys()

            # map the repeated B tag to I tag
            if i == i_prev and inside and eligible:
                target = rep_map[target]

        i_prev = i
        aligned_target.append(target)

    return aligned_target