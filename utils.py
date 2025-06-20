import torch
import pickle

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

def save_txt(filepath, data):
    with open(filepath, "w", encoding = "utf8") as f:
        for item in data:
            f.write(f"{item}\n")

def load_txt(filepath):
    with open(filepath, encoding = "utf8") as f:
        return [line.strip() for line in f]