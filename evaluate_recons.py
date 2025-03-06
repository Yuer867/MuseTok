import yaml
import sys, os, random, time
from tqdm import tqdm
from dataloader import REMIFullSongTransformerDataset

from utils import load_txt, pickle_load


def edit_distance(sentence1, sentence2):
    n = len(sentence1)
    m = len(sentence2)

    if n * m == 0:
        return n + m

    D = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = D[i - 1][j] + 1  # deletion
            down = D[i][j - 1] + 1  # insertion
            left_down = D[i - 1][j - 1]  # substitution
            if sentence1[i - 1] != sentence2[j - 1]:
                left_down += 1
            D[i][j] = min(left, down, left_down)

    return D[n][m]


def get_accuracy(orig_file, samp_file, metric='acc'):
    seq1 = [i for i in load_txt(orig_file) if 'Time Signature' not in i]
    seq2 = [i for i in load_txt(samp_file) if 'Time Signature' not in i]
    
    if metric == 'acc':
        # exact accuracy
        length = min(len(seq1), len(seq2))
        acc = sum(seq1[l] == seq2[l] for l in range(length)) / len(seq1)
    elif metric == 'edit':
        # edit distance
        acc = 1 - (edit_distance(seq1, seq2) / len(seq1))
        # print(edit_distance(seq1, seq2))

    return acc


def inference_accuracy(dir, metric='acc'):
    files = [i for i in os.listdir(dir) if '.txt' in i]
    files = sorted(files)
    print('# samples:', len(files) // 2)
    
    acc_total = []
    for i in tqdm(range(len(files) // 2)):
        assert files[2*i].split('_')[0] == files[2*i+1].split('_')[0]
        
        orig_file = os.path.join(dir, files[2*i])
        samp_file = os.path.join(dir, files[2*i+1])
        acc = get_accuracy(orig_file, samp_file, metric=metric)
        if acc > 0:
            acc_total.append(acc)
        else:
            print(orig_file, acc)
        
    dir_acc = sum(acc_total) / len(acc_total)
    print('sample_dir: {}\n num_samples: {}, accuracy: {}'.format(dir, len(files)//2, dir_acc))
    return dir_acc


if __name__ == "__main__":
    
    
    
    metric = 'edit'

    # dir = 'generations/reconstructions-pdmx-balanced-simplified-2'
    # inference_accuracy(dir, metric=metric)

    # dir = 'generations/reconstructions-pdmx-weighted-2'
    # inference_accuracy(dir, metric=metric)

    dir = 'generations/rvq-pdmx-n8-s1024-d128-beta1-polyphonic'
    inference_accuracy(dir, metric=metric)

    dir = 'generations/rvq-pdmx-n8-s1024-d128-beta1-density-polyphonic'
    inference_accuracy(dir, metric=metric)
