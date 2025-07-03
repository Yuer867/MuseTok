import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
import miditoolkit
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import defaultdict
import random
random.seed(42)

# DEFAULT_TIME_SIGNATURE = ['4/4', '2/4', '1/4', '3/4', '6/4', '2/2', '3/2', '3/8', '4/8', '6/8', '9/8', '12/8']
DEFAULT_TIME_SIGNATURE = ['4/4', '2/4', '3/4', '2/2', '3/8', '6/8']


def pickle_load(f):
    return pickle.load(open(f, 'rb'))


def split_emopia(output_dir):
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    data_dir = 'EMOPIA/{}/'.format(events_dir)

    # data split by running provided file: scripts/prepare_split.ipynb
    train = pd.read_csv("/deepfreeze/jingyue/data/EMOPIA/split/train_clip.csv", index_col=0)
    valid = pd.read_csv("/deepfreeze/jingyue/data/EMOPIA/split/val_clip.csv", index_col=0)
    test = pd.read_csv("/deepfreeze/jingyue/data/EMOPIA/split/test_clip.csv", index_col=0)

    # --- training dataset --- #
    train_set = []
    for i in range(len(train)):
        train_set.append(os.path.join(data_dir, train.iloc[i].clip_name[:-4] + '.pkl'))
        # train_set.append(train.iloc[i].clip_name[:-4] + '.pkl')
    pickle.dump(train_set, open(os.path.join(output_dir, 'train.pkl'), 'wb'))

    # --- valid dataset --- #
    valid_set = []
    for i in range(len(valid)):
        valid_set.append(os.path.join(data_dir, valid.iloc[i].clip_name[:-4] + '.pkl'))
        # valid_set.append(valid.iloc[i].clip_name[:-4] + '.pkl')
    pickle.dump(valid_set, open(os.path.join(output_dir, 'valid.pkl'), 'wb'))
        
    # --- test dataset --- #
    test_set = []
    for i in range(len(test)):
        test_set.append(os.path.join(data_dir, test.iloc[i].clip_name[:-4] + '.pkl'))
        # test_set.append(test.iloc[i].clip_name[:-4] + '.pkl')
    pickle.dump(test_set, open(os.path.join(output_dir, 'test.pkl'), 'wb'))

    num_files = len(train_set) + len(valid_set) + len(test_set)
    print(' > num files: ', num_files)
    print(' > train, valid, test:', len(train_set), len(valid_set), len(test_set))
    print()
    
    return num_files


def split_pop1k7(output_dir):
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    data_dir = 'Pop1k7/{}/'.format(events_dir)

    data_home = '/deepfreeze/jingyue/data/Pop1k7/{}'.format(events_dir)
    pkl_files = [os.path.join(data_dir, file) for file in os.listdir(data_home)]
    train_set, valid_set = train_test_split(pkl_files, test_size=0.2, random_state=42)
    valid_set, test_set = train_test_split(valid_set, test_size=0.5, random_state=42)

    pickle.dump(train_set, open(os.path.join(output_dir, 'train.pkl'), 'wb'))
    pickle.dump(valid_set, open(os.path.join(output_dir, 'valid.pkl'), 'wb'))
    pickle.dump(test_set, open(os.path.join(output_dir, 'test.pkl'), 'wb'))

    num_files = len(train_set) + len(valid_set) + len(test_set)
    print(' > num files: ', num_files)
    print(' > train, valid, test:', len(train_set), len(valid_set), len(test_set))
    print()

    return num_files


def split_POP909(output_dir):
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    data_dir = 'POP909/{}/'.format(events_dir)

    data_home = '/deepfreeze/jingyue/data/POP909/{}'.format(events_dir)
    pkl_files = [os.path.join(data_dir, file) for file in os.listdir(data_home)]
    train_set, valid_set = train_test_split(pkl_files, test_size=0.2, random_state=42)
    valid_set, test_set = train_test_split(valid_set, test_size=0.5, random_state=42)

    pickle.dump(train_set, open(os.path.join(output_dir, 'train.pkl'), 'wb'))
    pickle.dump(valid_set, open(os.path.join(output_dir, 'valid.pkl'), 'wb'))
    pickle.dump(test_set, open(os.path.join(output_dir, 'test.pkl'), 'wb'))

    num_files = len(train_set) + len(valid_set) + len(test_set)
    print(' > num files: ', num_files)
    print(' > train, valid, test:', len(train_set), len(valid_set), len(test_set))
    print()

    return num_files


def split_others(output_dir, data_name):
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    data_dir = '{}/{}/'.format(data_name, events_dir)
    
    non_split_files = glob('/deepfreeze/jingyue/data/{}/midi_quantized_480/*.mid'.format(data_name))
    midi_files = [os.path.basename(file)[:-4] for file in non_split_files]
    file2split = dict()
    train_files, valid_files = train_test_split(midi_files, test_size=0.2, random_state=42)
    valid_files, test_files = train_test_split(valid_files, test_size=0.5, random_state=42)
    for file in train_files:
        file2split[file] = 'train'
    for file in valid_files:
        file2split[file] = 'valid'
    for file in test_files:
        file2split[file] = 'test'
    
    data_home = '/deepfreeze/jingyue/data/{}/{}'.format(data_name, events_dir)
    midi_home = '/deepfreeze/jingyue/data/{}/midi_quantized_480_split'.format(data_name)
    pkl_files = [os.path.join(data_dir, file) for file in os.listdir(data_home)]
    data_split = {'train':[], 'valid':[], 'test':[]}
    invalid_files = 0
    for file in tqdm(pkl_files):
        filename = os.path.basename(file)[:-4]
        if not valid_time_signature(os.path.join(midi_home, filename + '.mid')):
            invalid_files += 1
            continue
        if filename in file2split:
            data_split[file2split[filename]].append(file)
        elif filename[:-2] in file2split:
            data_split[file2split[filename[:-2]]].append(file)
        elif filename[:-3] in file2split:
            data_split[file2split[filename[:-3]]].append(file)
        else:
            raise ValueError('file {} not belong to any splits'.format(file))
    
    pickle.dump(data_split['train'], open(os.path.join(output_dir, 'train.pkl'), 'wb'))
    pickle.dump(data_split['valid'], open(os.path.join(output_dir, 'valid.pkl'), 'wb'))
    pickle.dump(data_split['test'], open(os.path.join(output_dir, 'test.pkl'), 'wb'))

    num_files = len(data_split['train']) + len(data_split['valid']) + len(data_split['test'])
    print(' > num files: ', num_files)
    print(' > train, valid, test:', len(data_split['train']), len(data_split['valid']), len(data_split['test']))
    print(' > invalid files: ', invalid_files)
    print()

    return num_files, invalid_files


def pkl2split(file, file2split):
    data_dir = 'PDMX/{}/'.format(events_dir)
    filename = os.path.basename(file)[:-4]
    file_subdir = '/'.join(file.split('/')[-3:-1])
    
    midi_home = '/deepfreeze/jingyue/data/PDMX/data_midi_piano_split'
    if not valid_time_signature(os.path.join(midi_home, file_subdir, filename + '.mid')):
        return 1, None, None
    
    pkl_home = 'data/PDMX/{}'.format(events_dir)
    _, events = pickle_load(os.path.join(pkl_home, file_subdir, filename + '.pkl'))
    if is_monophonic(events):
        piece_type = 'monophonic'
    else:
        piece_type = 'contrapuntal'
    
    if filename in file2split:
        return 0, os.path.join(data_dir, file_subdir, filename + '.pkl'), file2split[filename], piece_type
    elif filename[:-2] in file2split:
        return 0, os.path.join(data_dir, file_subdir, filename + '.pkl'), file2split[filename[:-2]], piece_type
    elif filename[:-3] in file2split:
        return 0, os.path.join(data_dir, file_subdir, filename + '.pkl'), file2split[filename[:-3]], piece_type
    elif filename[:-4] in file2split:
        return 0, os.path.join(data_dir, file_subdir, filename + '.pkl'), file2split[filename[:-4]], piece_type
    else:
        raise ValueError('file {} not belong to any splits'.format(file))


def split_PDMX(output_dir):
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    
    non_split_files = glob('/deepfreeze/jingyue/data/PDMX/data_midi_piano/*/*/*.mid')
    midi_files = [os.path.basename(file)[:-4] for file in non_split_files]
    file2split = dict()
    train_files, valid_files = train_test_split(midi_files, test_size=0.2, random_state=42)
    valid_files, test_files = train_test_split(valid_files, test_size=0.5, random_state=42)
    for file in train_files:
        file2split[file] = 'train'
    for file in valid_files:
        file2split[file] = 'valid'
    for file in test_files:
        file2split[file] = 'test'
    
    pkl_files = glob('data/PDMX/{}/*/*/*.pkl'.format(events_dir))
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(partial(pkl2split, file2split=file2split), pkl_files, chunksize=16), desc='Preprocess', total=len(pkl_files)))
    
    invalid_files = sum([i[0] for i in results])
    train_monophonic_set = [i[1] for i in results if i[2] == 'train' and i[3] == 'monophonic']
    train_contrapuntal_set = [i[1] for i in results if i[2] == 'train' and i[3] == 'contrapuntal']
    train_set = train_monophonic_set + train_contrapuntal_set
    
    valid_monophonic_set = [i[1] for i in results if i[2] == 'valid' and i[3] == 'monophonic']
    valid_contrapuntal_set = [i[1] for i in results if i[2] == 'valid' and i[3] == 'contrapuntal']
    valid_set = valid_monophonic_set + valid_contrapuntal_set
    
    test_monophonic_set = [i[1] for i in results if i[2] == 'test' and i[3] == 'monophonic']
    test_contrapuntal_set = [i[1] for i in results if i[2] == 'test' and i[3] == 'contrapuntal']
    test_set = test_monophonic_set + test_contrapuntal_set
    
    pickle.dump(train_set, open(os.path.join(output_dir, 'train.pkl'), 'wb'))
    pickle.dump(train_monophonic_set, open(os.path.join(output_dir, 'train_mono.pkl'), 'wb'))
    pickle.dump(train_contrapuntal_set, open(os.path.join(output_dir, 'train_contra.pkl'), 'wb'))
    
    pickle.dump(valid_set, open(os.path.join(output_dir, 'valid.pkl'), 'wb'))
    pickle.dump(valid_monophonic_set, open(os.path.join(output_dir, 'valid_mono.pkl'), 'wb'))
    pickle.dump(valid_contrapuntal_set, open(os.path.join(output_dir, 'valid_contra.pkl'), 'wb'))
    
    pickle.dump(test_set, open(os.path.join(output_dir, 'test.pkl'), 'wb'))
    pickle.dump(test_monophonic_set, open(os.path.join(output_dir, 'test_mono.pkl'), 'wb'))
    pickle.dump(test_contrapuntal_set, open(os.path.join(output_dir, 'test_contra.pkl'), 'wb'))
    
    density2files_train = {'monophonic': train_monophonic_set, 
                            'contrapuntal': train_contrapuntal_set}
    pickle.dump(density2files_train, open(os.path.join(output_dir, 'density2pieces_train.pkl'), 'wb'))
    
    density2files_valid = {'monophonic': valid_monophonic_set, 
                            'contrapuntal': valid_contrapuntal_set}
    pickle.dump(density2files_valid, open(os.path.join(output_dir, 'density2pieces_valid.pkl'), 'wb'))
    
    density2files_test = {'monophonic': test_monophonic_set, 
                            'contrapuntal': test_contrapuntal_set}
    pickle.dump(density2files_test, open(os.path.join(output_dir, 'density2pieces_test.pkl'), 'wb'))

    num_files = len(train_monophonic_set) + len(train_contrapuntal_set) + \
                len(valid_monophonic_set) + len(valid_contrapuntal_set) + \
                len(test_monophonic_set) + len(test_contrapuntal_set)
    print(' > num files: ', num_files)
    print(' > train: {} ({} + {}), valid: {} ({} + {}), test: {} ({} + {})'.format(
            len(train_set), len(train_monophonic_set), len(train_contrapuntal_set),
            len(valid_set), len(valid_monophonic_set), len(valid_contrapuntal_set), 
            len(test_set), len(test_monophonic_set), len(test_contrapuntal_set)))
    print(' > invalid files: ', invalid_files)
    print()

    return num_files, invalid_files


def valid_time_signature(file):
    midi_obj = miditoolkit.midi.parser.MidiFile(file)
    time_sig = str(midi_obj.time_signature_changes[0]).split(' ')[0]
    if time_sig in DEFAULT_TIME_SIGNATURE:
        return True
    else:
        return False


def combine_split(split_path, PDMX_split_path, output_dir):
    all_files = 0
    
    print('processing train set...')
    total_train_contrapuntal = []
    total_train_monophonic = []
    total_train_polyphonic = []
    total_train_samples = []
    
    for file in glob(os.path.join(split_path, '*/train.pkl')):
        if 'PDMX' in file:
            continue
        total_train_polyphonic += pickle_load(file)
        total_train_samples += pickle_load(file)
        
    total_train_contrapuntal += pickle_load(os.path.join(PDMX_split_path, 'train_contra.pkl'))
    total_train_samples += pickle_load(os.path.join(PDMX_split_path, 'train_contra.pkl'))
    
    total_train_monophonic += pickle_load(os.path.join(PDMX_split_path, 'train_mono.pkl'))
    total_train_samples += pickle_load(os.path.join(PDMX_split_path, 'train_mono.pkl'))

    pickle.dump(total_train_samples, open(os.path.join(output_dir, 'all_train.pkl'), 'wb'))
    all_files += len(total_train_contrapuntal) + len(total_train_monophonic) + len(total_train_polyphonic)
    
    density2files_train = {'monophonic': total_train_monophonic, 
                           'contrapuntal': total_train_contrapuntal, 
                           'polyphonic': total_train_polyphonic}
    pickle.dump(density2files_train, open(os.path.join(output_dir, 'density2pieces_train.pkl'), 'wb'))
    
    print('processing valid set...')
    total_valid_contrapuntal = []
    total_valid_monophonic = []
    total_valid_polyphonic = []
    total_valid_samples = []
    
    for file in glob(os.path.join(split_path, '*/valid.pkl')):
        if 'PDMX' in file:
            continue
        total_valid_polyphonic += pickle_load(file)
        total_valid_samples += pickle_load(file)
        
    total_valid_contrapuntal += pickle_load(os.path.join(PDMX_split_path, 'valid_contra.pkl'))
    total_valid_samples += pickle_load(os.path.join(PDMX_split_path, 'valid_contra.pkl'))
    
    total_valid_monophonic += pickle_load(os.path.join(PDMX_split_path, 'valid_mono.pkl'))
    total_valid_samples += pickle_load(os.path.join(PDMX_split_path, 'valid_mono.pkl'))

    pickle.dump(total_valid_samples, open(os.path.join(output_dir, 'all_valid.pkl'), 'wb'))
    all_files += len(total_valid_contrapuntal) + len(total_valid_monophonic) + len(total_valid_polyphonic)
    
    density2files_valid = {'monophonic': total_valid_monophonic, 
                           'contrapuntal': total_valid_contrapuntal, 
                           'polyphonic': total_valid_polyphonic}
    pickle.dump(density2files_valid, open(os.path.join(output_dir, 'density2pieces_valid.pkl'), 'wb'))
    
    small_valid_samples = total_valid_polyphonic + \
                          random.sample(total_valid_monophonic, 1200) + \
                          random.sample(total_valid_contrapuntal, 1200)
    pickle.dump(small_valid_samples, open(os.path.join(output_dir, 'small_valid.pkl'), 'wb'))
    
    print('processing test set...')
    total_test_contrapuntal = []
    total_test_monophonic = []
    total_test_polyphonic = []
    total_test_samples = []
    
    for file in glob(os.path.join(split_path, '*/test.pkl')):
        if 'PDMX' in file:
            continue
        total_test_polyphonic += pickle_load(file)
        total_test_samples += pickle_load(file)
        
    total_test_contrapuntal += pickle_load(os.path.join(PDMX_split_path, 'test_contra.pkl'))
    total_test_samples += pickle_load(os.path.join(PDMX_split_path, 'test_contra.pkl'))
    
    total_test_monophonic += pickle_load(os.path.join(PDMX_split_path, 'test_mono.pkl'))
    total_test_samples += pickle_load(os.path.join(PDMX_split_path, 'test_mono.pkl'))
    
    pickle.dump(total_test_samples, open(os.path.join(output_dir, 'all_test.pkl'), 'wb'))
    all_files += len(total_test_contrapuntal) + len(total_test_monophonic) + len(total_test_polyphonic)
    
    density2files_test = {'monophonic': total_test_monophonic, 
                          'contrapuntal': total_test_contrapuntal, 
                          'polyphonic': total_test_polyphonic}
    pickle.dump(density2files_test, open(os.path.join(output_dir, 'density2pieces_test.pkl'), 'wb'))
    
    print(' > num files: ', all_files)
    print(' > train: {} ({} + {} + {}), valid: {} ({} + {} + {}), test: {} ({} + {} + {})'.format(
            len(total_train_samples), len(total_train_monophonic), len(total_train_contrapuntal), len(total_train_polyphonic), 
            len(total_valid_samples), len(total_valid_monophonic), len(total_valid_contrapuntal), len(total_valid_polyphonic), 
            len(total_test_samples), len(total_test_monophonic), len(total_test_contrapuntal), len(total_test_polyphonic)))
    print()
    
    return all_files


def is_monophonic(events):
    note_number = defaultdict(int)
    bar = 0
    beat = None
    for evs in events:
        if evs['name'] == 'Bar':
            bar += 1
        if evs['name'] == 'Beat':
            beat = evs['value']
        if evs['name'] == 'Note_Pitch':
            note_number["bar{}_beat{}".format(bar, beat)] += 1
    note_density = sum([note_number[i] for i in note_number]) / len(note_number)
    
    if note_density < 1.5:
        return True
    elif note_density >= 1.5:
        return False


if __name__ == '__main__':
    # split each dataset into training, validation and test sets (8:1:1)
    #   - if have pre-designed split requirements for downstream tasks (e.g. EMOPIA, Pianist8), follow the previous works
    #   - if have multiple pieces came from the same songs because of different time signatures, make sure they will be included in the same set (e.g. PDMX)
    #   - otherwise randomly split
    events_dir = 'data_events'
    split_dir = 'data_splits_events'
    
    PDMX_events_dir = 'data_events'
    PDMX_split_dir = 'data_splits_events'
    
    print('EMOPIA')
    emopia_files = split_emopia('data/{}/EMOPIA/'.format(split_dir))
    assert emopia_files == len(glob('data/EMOPIA/{}/*.pkl'.format(events_dir)))
    
    print('Pop1k7')
    pop1k7_files = split_pop1k7('data/{}/Pop1k7/'.format(split_dir))
    assert pop1k7_files == len(glob('data/Pop1k7/{}/*.pkl'.format(events_dir)))
    
    print('POP909')
    pop909_files = split_POP909('data/{}/POP909/'.format(split_dir))
    assert pop909_files == len(glob('data/POP909/{}/*.pkl'.format(events_dir)))
    
    print('Multipianomide-Classic')
    multipianomide_files, invalid_files = split_others('data/{}/Multipianomide-Classic/'.format(split_dir), 'Multipianomide-Classic')
    assert multipianomide_files + invalid_files == len(glob('data/Multipianomide-Classic/{}/*.pkl'.format(events_dir)))
    
    print('Hymnal-Folk')
    hymnal_files, invalid_files = split_others('data/{}/Hymnal-Folk/'.format(split_dir), 'Hymnal-Folk')
    assert hymnal_files + invalid_files == len(glob('data/Hymnal-Folk/{}/*.pkl'.format(events_dir)))
    
    print('Ragtime-perfect-Jazz')
    ragtime_files, invalid_files = split_others('data/{}/Ragtime-perfect-Jazz/'.format(split_dir), 'Ragtime-perfect-Jazz')
    assert ragtime_files + invalid_files == len(glob('data/Ragtime-perfect-Jazz/{}/*.pkl'.format(events_dir)))
    
    print('PDMX')
    pdmx_files, invalid_files = split_PDMX('data/{}/PDMX/'.format(split_dir))
    assert pdmx_files + invalid_files == len(glob('data/PDMX/{}/*/*/*.pkl'.format(events_dir)))
    # pdmx_files = len(glob('/deepfreeze/jingyue/data/PDMX/{}/*/*/*.pkl'.format(PDMX_events_dir)))
    
    # combine all samples into super sets
    split_path = 'data/{}/'.format(split_dir)
    PDMX_split_path = 'data/{}/PDMX'.format(PDMX_split_dir)
    output_path = 'data/{}/all'.format(split_dir)
    os.makedirs(output_path, exist_ok=True)
    all_files = combine_split(split_path, PDMX_split_path, output_path)
    assert emopia_files + pop1k7_files + pop909_files + multipianomide_files + hymnal_files + ragtime_files + pdmx_files == all_files