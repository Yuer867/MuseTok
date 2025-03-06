import os
import shutil
import torch
import pickle
from scipy.spatial import distance
from typing import Union, List, Tuple, Collection

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def numpy_to_tensor(arr, use_gpu=True, device='cuda:0'):
    if use_gpu:
        return torch.tensor(arr).to(device).float()
    else:
        return torch.tensor(arr).float()

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def pickle_load(f):
    return pickle.load(open(f, 'rb'))

def pickle_dump(obj, f):
    pickle.dump(obj, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def save_txt(filepath: str, data: list):
    """Save a list to a TXT file."""
    with open(filepath, "w", encoding = "utf8") as f:
        for item in data:
            f.write(f"{item}\n")


def load_txt(filepath: str):
    """Load a TXT file as a list."""
    with open(filepath, encoding = "utf8") as f:
        return [line.strip() for line in f]


def unique(l: Union[List, Tuple]) -> list:
    """Returns the unique values from a list while retaining order."""
    return list(dict.fromkeys(list(l)))

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))