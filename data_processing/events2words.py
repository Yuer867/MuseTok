import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm


BEAT_RESOL = 480
TICK_RESOL = BEAT_RESOL // 12
TRIPLET_RESOL = BEAT_RESOL // 24

DEFAULT_TIME_SIGNATURE = ['4/4', '2/4', '1/4', '3/4', '6/4', '2/2', '3/2', '3/8', '4/8', '6/8', '9/8', '12/8']
DEFAULT_VELOCITY_BINS = np.linspace(4, 127, 42, dtype=int)
DEFAULT_DURATION_BINS = np.arange(BEAT_RESOL / 12, BEAT_RESOL * 8 + 1, BEAT_RESOL / 12)

default_normal_duration = np.array([BEAT_RESOL // 4,                   # 1/16       - 120
                                    BEAT_RESOL // 2,                   # 1/8        - 240
                                    BEAT_RESOL // 2 + BEAT_RESOL // 4, # 1/8 + 1/16 - 360
                                    BEAT_RESOL,                        # 1/4        - 480
                                    BEAT_RESOL + BEAT_RESOL // 2,      # 1/4 + 1/8  - 720
                                    2 * BEAT_RESOL,                    # 1/2        - 960
                                    2 * BEAT_RESOL + BEAT_RESOL,       # 1/2 + 1/4  - 1440
                                    4 * BEAT_RESOL])                   # 1          - 1920
default_triplet_duration = np.array([BEAT_RESOL // 6,                  # 1/8 // 3   - 80
                                    BEAT_RESOL // 3,                   # 1/4 // 3   - 160
                                    2 * BEAT_RESOL // 3,               # 1/2 // 3   - 320
                                    4 * BEAT_RESOL // 3])              # 1 // 3     - 640
default_duration = np.concat([default_normal_duration, default_triplet_duration])


def build_full_vocab(add_velocity=True):
    vocab = []
    
    # --- Bar & EOS --- #
    vocab.append('Bar_None')
    vocab.append('EOS_None')
    
    # --- Beat --- #
    for t in DEFAULT_TIME_SIGNATURE:
        numerator, denominator = t.split('/')
        quarters_per_bar = 4 * int(numerator) / int(denominator)
        bar_resol = int(BEAT_RESOL * quarters_per_bar)
        default_onset = np.unique(np.concat([np.arange(0, bar_resol, TRIPLET_RESOL*4), np.arange(0, bar_resol, TICK_RESOL*3), np.array([bar_resol])]))
        for onset in default_onset:
            vocab.append('Beat_{}'.format(onset // TICK_RESOL))
        # for timing in range(0, bar_resol, TICK_RESOL):
        #     vocab.append('Beat_{}'.format(timing // TICK_RESOL))

    # --- note --- #
    # note pitch
    for p in range(21, 109):
        vocab.append('Note_Pitch_{}'.format(p))
    # note velocity
    if add_velocity:
        for v in np.linspace(4, 127, 42, dtype=int):
            vocab.append('Note_Velocity_{}'.format(int(v)))
    # note duration
    # for d in np.arange(BEAT_RESOL / 12, BEAT_RESOL * 12 + 1, BEAT_RESOL / 12):
    #     vocab.append('Note_Duration_{}'.format(int(d)))
    for d in default_duration:
        vocab.append('Note_Duration_{}'.format(int(d)))

    # --- time signature ---- #
    for t in DEFAULT_TIME_SIGNATURE:
        vocab.append('Time_Signature_{}'.format(t))

    return vocab


def events2dictionary(event_path, dictionary_path, add_velocity=False):
    # list files
    event_files = os.listdir(event_path)
    n_files = len(event_files)
    print(' > num files:', n_files)

    # generate dictionary
    all_events = []
    for file in tqdm(event_files):
        _, events = pickle.load(open(os.path.join(event_path, file), 'rb'))
        for event in events:
            all_events.append('{}_{}'.format(event['name'], event['value']))
    full_vocab = build_full_vocab(add_velocity=add_velocity)
    for evs in list(set(all_events)):
        if evs not in full_vocab:
            print(evs)
    all_events = all_events + full_vocab
    unique_events = sorted(set(all_events), key=lambda x: (not isinstance(x, int), x))
    event2word = {key: i for i, key in enumerate(unique_events)}
    word2event = {i: key for i, key in enumerate(unique_events)}
    # print(event2word)
    print(' > num classes:', len(word2event))
    print()
    print(word2event)

    # save
    pickle.dump((event2word, word2event), open(dictionary_path, 'wb'))


if __name__ == '__main__':
    dictionary_path = '/deepfreeze/jingyue/data/dictionary_strict.pkl'
    events_path = '/deepfreeze/jingyue/data/EMOPIA/data_events_timeLast_strict'
    events2dictionary(events_path, dictionary_path, add_velocity=True)
    
    # vocab = build_full_vocab(add_velocity=True)
    # print(sorted(set(vocab), key=lambda x: (not isinstance(x, int), x)))
    # print(len(set(vocab)))
