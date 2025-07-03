import os
import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

from utils import load_txt

###########################################
# evaluate reconstruction
###########################################
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


def get_accuracy(orig_file, samp_file, metric='acc', token_type=None):
    seq1 = [i for i in load_txt(orig_file) if 'Time Signature' not in i]
    seq2 = [i for i in load_txt(samp_file) if 'Time Signature' not in i]
    
    if token_type is not None:
        seq1 = [i for i in seq1 if token_type in i]
        seq2 = [i for i in seq2 if token_type in i]
    
    if metric == 'acc':
        # exact accuracy
        length = min(len(seq1), len(seq2))
        acc = sum(seq1[l] == seq2[l] for l in range(length)) / len(seq1)
    elif metric == 'edit':
        # edit distance
        acc = 1 - (edit_distance(seq1, seq2) / len(seq1))

    return acc


def reconstruction_accuracy(dir, metric='acc'):
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


###########################################
# evaluate generation
###########################################
def chroma(events):
    pitch_classes = [int(event.split('_')[-1]) % 12 for event in events if 'Note' in event]
    if len(pitch_classes):
        count = np.bincount(pitch_classes, minlength=12)
        count = count / np.sqrt(np.sum(count ** 2))
    else:
        count = np.array([1/12] * 12)
    return count


def groove(events, pos_per_bar=48):
    onsets = [int(event.split('_')[-1]) for event in events if 'Beat' in event]
    if len(onsets):
        count = np.bincount(onsets, minlength=pos_per_bar)
        # count = np.convolve(count, [1, 4, 1], 'same')
        count = count / np.sqrt(np.sum(count ** 2))
    else:
        count = np.array([1/pos_per_bar] * pos_per_bar)
    return count


def get_density(events):
    note_number = defaultdict(int)
    bar = 0
    beat = None
    for evs in events:
        if 'Bar' in evs:
            bar += 1
        if 'Beat' in evs:
            beat = evs.split('_')[1]
        if 'Note_Pitch' in evs:
            note_number["bar{}_beat{}".format(bar, beat)] += 1
        if 'Time_Signature' in evs:
            time_sig = evs.split('_')[2]
            numerator,denominator = time_sig.split('/')
            
    # note_density = sum([note_number[i] for i in note_number]) / len(note_number)
    note_density = sum([note_number[i] for i in note_number]) / (bar * 4 * int(numerator) / int(denominator))
    # note_density = sum([note_number[i] for i in note_number])

    return note_density


def get_duration(events, metric='min'):
    durations = []
    for evs in events:
        if 'Note_Duration' in evs:
            durations.append(int(evs.split('_')[2]))
    if metric == 'mean':
        return np.mean(durations)
    if metric == 'min':
        return np.min(durations)
    if metric == 'max':
        return np.max(durations)


def pitch_entropy(events):
    pitch_values = [int(event.split('_')[-1]) for event in events if 'Note' in event]
    if len(pitch_values):
        count = Counter(pitch_values)
        count = np.array(list(count.values()))
        prob = count / len(pitch_values)
        entropy = -np.sum(prob * np.log(prob))
    else:
        entropy = -np.log(1/88)
        
    return entropy


def sequence_similarity(sequence, primer_n_bar, reduce='max'):
    bar_pos = [i for i in range(len(sequence)) if sequence[i] == 'Bar_None']
    if len(bar_pos) <= primer_n_bar:
        return None, None, None, None, None, None
    bar_pos.append(len(sequence))
    
    primer = sequence[:bar_pos[primer_n_bar]]
    generated = sequence[bar_pos[primer_n_bar]:]
    
    primer_density = get_density(primer)
    primer_duration = get_duration(primer, metric='min')
    primer_pitch_entropy = pitch_entropy(primer)
    generated_pitch_entropy = pitch_entropy(generated)
    
    sequence_chroma = []
    sequence_groove = []
    for b_st, b_end in zip(bar_pos[:-1], bar_pos[1:]):
        events = sequence[b_st:b_end]
        sequence_chroma.append(chroma(events))
        sequence_groove.append(groove(events))

    reference_chroma = sequence_chroma[:primer_n_bar]
    generate_chroma = sequence_chroma[primer_n_bar:]
    chroma_similarity = []
    for g in generate_chroma:
        sim = []
        for r in reference_chroma:
            sim.append(np.sum(g * r))
        if reduce == 'max':
            chroma_similarity.append(max(sim))
        elif reduce == 'mean':
            chroma_similarity.append(np.mean(sim))
    chroma_similarity = np.mean(chroma_similarity)
            
    reference_groove = sequence_groove[:primer_n_bar]
    generate_groove = sequence_groove[primer_n_bar:]
    groove_similarity = []
    for g in generate_groove:
        sim = []
        for r in reference_groove:
            sim.append(np.sum(g * r))
        if reduce == 'max':
            groove_similarity.append(max(sim))
        elif reduce == 'mean':
            groove_similarity.append(np.mean(sim))
    groove_similarity = np.mean(groove_similarity)
        
    return chroma_similarity, groove_similarity, primer_density, primer_duration, primer_pitch_entropy, generated_pitch_entropy


if __name__ == "__main__":
    # compute reconstruction accuracy
    metric = 'edit'
    dir = 'samples/reconstruction'
    reconstruction_accuracy(dir, metric=metric)

    # compute generation similarity
    dir = 'samples/generation'
    files = glob(os.path.join(dir, '*primer*.txt'))
    
    chroma_score = []
    groove_score = []
    piece_primer_pitch_entropy = []
    piece_pitch_entropy = []
    
    fail_generation = 0
    success_generation = 0
    for file in files:
        file_num = int(os.path.basename(file).split('_')[0][2:])
        sequence = load_txt(file)
        chroma_sim, groove_sim, _, _, primer_entropy, generated_entropy = sequence_similarity(sequence, primer_n_bar=4)
        if chroma_sim is None:
            fail_generation += 1
        else:
            success_generation += 1
            chroma_score.append(chroma_sim)
            groove_score.append(groove_sim)
            piece_primer_pitch_entropy.append(primer_entropy)
            piece_pitch_entropy.append(generated_entropy)
            
    print('{} +/- {}'.format(round(np.mean(chroma_score), 4), round(1.96 * np.std(chroma_score) / np.sqrt(len(chroma_score)), 4)), 
          '{} +/- {}'.format(round(np.mean(groove_score), 4), round(1.96 * np.std(groove_score) / np.sqrt(len(groove_score)), 4)), 
          '{} +/- {}'.format(round(np.mean(piece_pitch_entropy), 4), round(1.96 * np.std(piece_pitch_entropy) / np.sqrt(len(piece_pitch_entropy)), 4)),
            np.mean(piece_primer_pitch_entropy), fail_generation, success_generation)