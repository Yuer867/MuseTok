import os, pickle, random
from glob import glob
from collections import defaultdict

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

IDX_TO_KEY = {
    0: 'A',
    1: 'A#',
    2: 'B',
    3: 'C',
    4: 'C#',
    5: 'D',
    6: 'D#',
    7: 'E',
    8: 'F',
    9: 'F#',
    10: 'G',
    11: 'G#'
}
KEY_TO_IDX = {
    v:k for k, v in IDX_TO_KEY.items()
}

def get_chord_tone(chord_event):
    tone = chord_event['value'].split('_')[0]
    return tone

def transpose_chord(chord_event, n_keys):
    if chord_event['value'] == 'N_N':
        return chord_event

    orig_tone = get_chord_tone(chord_event)
    orig_tone_idx = KEY_TO_IDX[orig_tone]
    new_tone_idx = (orig_tone_idx + 12 + n_keys) % 12
    new_chord_value = chord_event['value'].replace(
        '{}_'.format(orig_tone), '{}_'.format(IDX_TO_KEY[new_tone_idx])
    )
    new_chord_event = {'name': chord_event['name'], 'value': new_chord_value}
    # print ('keys={}. {} --> {}'.format(n_keys, chord_event, new_chord_event))

    return new_chord_event

def check_extreme_pitch(raw_events):
    low, high = 128, 0
    for ev in raw_events:
        if ev['name'] == 'Note_Pitch':
            low = min(low, int(ev['value']))
            high = max(high, int(ev['value']))

    return low, high

def transpose_events(raw_events, n_keys):
    transposed_raw_events = []

    for ev in raw_events:
        if ev['name'] == 'Note_Pitch':
            transposed_raw_events.append(
                {'name': ev['name'], 'value': int(ev['value']) + n_keys}
            )
        elif ev['name'] == 'Chord':
            transposed_raw_events.append(
                transpose_chord(ev, n_keys)
            )
        else:
            transposed_raw_events.append(ev)

    assert len(transposed_raw_events) == len(raw_events)
    return transposed_raw_events

def pickle_load(path):
    return pickle.load(open(path, 'rb'))

def convert_event(event_seq, event2idx, to_ndarr=True):
    if isinstance(event_seq[0], dict):
        event_seq = [event2idx['{}_{}'.format(e['name'], e['value'])] for e in event_seq]
    else:
        event_seq = [event2idx[e] for e in event_seq]

    if to_ndarr:
        return np.array(event_seq)
    else:
        return event_seq

class REMIFullSongTransformerDataset(Dataset):
    def __init__(self, data_dir, vocab_file, 
                model_enc_seqlen=128, model_dec_seqlen=1280, model_max_bars=16,
                pieces=[], do_augment=True, augment_range=range(-6, 7), 
                min_pitch=22, max_pitch=107, pad_to_same=True, use_attr_cls=True,
                appoint_st_bar=None, dec_end_pad_value=None, 
                balanced_time=False, time2pieces=None, 
                balanced_density=False, density2pieces=None, 
                shuffle=True):
        self.vocab_file = vocab_file
        self.read_vocab()

        self.data_dir = data_dir
        self.pieces = pieces
        self.build_dataset()

        self.model_enc_seqlen = model_enc_seqlen
        self.model_dec_seqlen = model_dec_seqlen
        self.model_max_bars = model_max_bars

        self.do_augment = do_augment
        self.augment_range = augment_range
        self.min_pitch, self.max_pitch = min_pitch, max_pitch
        self.pad_to_same = pad_to_same
        self.use_attr_cls = use_attr_cls
        
        self.balanced_time = balanced_time
        self.class_time_sig = ['4/4', '6/8', '2/4', '2/2', '3/4']
        self.time2pieces = time2pieces
        
        self.balanced_density = balanced_density
        self.class_density = ['monophonic', 'polyphonic', 'performance']
        # self.class_density = ['monophonic', 'polyphonic']
        self.density2pieces = density2pieces

        self.appoint_st_bar = appoint_st_bar
        if dec_end_pad_value is None:
            self.dec_end_pad_value = self.pad_token
        elif dec_end_pad_value == 'EOS':
            self.dec_end_pad_value = self.eos_token
        else:
            self.dec_end_pad_value = self.pad_token
            
        self.shuffle = shuffle
        self.generate_queue()

    def read_vocab(self):
        vocab = pickle_load(self.vocab_file)[0]
        self.idx2event = pickle_load(self.vocab_file)[1]
        orig_vocab_size = len(vocab)
        self.event2idx = vocab
        self.bar_token = self.event2idx['Bar_None']
        self.eos_token = self.event2idx['EOS_None']
        self.pad_token = orig_vocab_size
        self.vocab_size = self.pad_token + 1
    
    def build_dataset(self):
        if not self.pieces:
            self.pieces = sorted( glob(os.path.join(self.data_dir, '*.pkl')) )
        else:
            self.pieces = sorted( [os.path.join(self.data_dir, p) for p in self.pieces] )
        self.piece2idx = {self.pieces[idx]:idx for idx in range(len(self.pieces))}

        self.piece_bar_pos = []

        for i, p in enumerate(self.pieces):
            bar_pos, p_evs = pickle_load(p)
            if not i % 1000:
                print ('[preparing data] now at #{}'.format(i))
            if bar_pos[-1] == len(p_evs):
                print ('piece {}, got appended bar markers'.format(p))
                bar_pos = bar_pos[:-1]
            if len(p_evs) - bar_pos[-1] == 2:
                # got empty trailing bar
                bar_pos = bar_pos[:-1]

            bar_pos.append(len(p_evs))

            self.piece_bar_pos.append(bar_pos)

    def get_sample_from_file(self, piece_idx):
        piece_evs = pickle_load(self.pieces[piece_idx])[1]
        # piece_evs = [{'name': '_'.join(i.split('_')[0:-1]), 'value': i.split('_')[-1]} for i in piece_evs]
        if len(self.piece_bar_pos[piece_idx]) > self.model_max_bars and self.appoint_st_bar is None:
            picked_st_bar = random.choice(
                range(len(self.piece_bar_pos[piece_idx]) - self.model_max_bars)
            )
        elif self.appoint_st_bar is not None and self.appoint_st_bar < len(self.piece_bar_pos[piece_idx]) - self.model_max_bars:
            picked_st_bar = self.appoint_st_bar
        else:
            picked_st_bar = 0

        piece_bar_pos = self.piece_bar_pos[piece_idx]

        if len(piece_bar_pos) > self.model_max_bars:
            piece_evs = piece_evs[ piece_bar_pos[picked_st_bar] : piece_bar_pos[picked_st_bar + self.model_max_bars] ]
            picked_bar_pos = np.array(piece_bar_pos[ picked_st_bar : picked_st_bar + self.model_max_bars ]) - piece_bar_pos[picked_st_bar]
            n_bars = self.model_max_bars
        else:
            picked_bar_pos = np.array(piece_bar_pos + [piece_bar_pos[-1]] * (self.model_max_bars - len(piece_bar_pos)))
            n_bars = len(piece_bar_pos) - 1
            assert len(picked_bar_pos) == self.model_max_bars

        return piece_evs, picked_st_bar, picked_bar_pos, n_bars

    def pad_sequence(self, seq, maxlen, pad_value=None):
        if pad_value is None:
            pad_value = self.pad_token

        seq.extend( [pad_value for _ in range(maxlen- len(seq))] )

        return seq

    def pitch_augment(self, bar_events):
        bar_min_pitch, bar_max_pitch = check_extreme_pitch(bar_events)
        
        n_keys = random.choice(self.augment_range)
        while bar_min_pitch + n_keys < self.min_pitch or bar_max_pitch + n_keys > self.max_pitch:
            n_keys = random.choice(self.augment_range)

        augmented_bar_events = transpose_events(bar_events, n_keys)
        return augmented_bar_events

    def get_attr_classes(self, piece, st_bar):
        polyph_cls = pickle_load(os.path.join(self.data_dir, 'attr_cls/polyph', piece))[st_bar : st_bar + self.model_max_bars]
        rfreq_cls = pickle_load(os.path.join(self.data_dir, 'attr_cls/rhythm', piece))[st_bar : st_bar + self.model_max_bars]

        polyph_cls.extend([0 for _ in range(self.model_max_bars - len(polyph_cls))])
        rfreq_cls.extend([0 for _ in range(self.model_max_bars - len(rfreq_cls))])

        assert len(polyph_cls) == self.model_max_bars
        assert len(rfreq_cls) == self.model_max_bars

        return polyph_cls, rfreq_cls

    def get_encoder_input_data(self, bar_positions, bar_events):
        assert len(bar_positions) == self.model_max_bars + 1
        enc_padding_mask = np.ones((self.model_max_bars, self.model_enc_seqlen), dtype=bool)
        enc_padding_mask[:, :2] = False
        padded_enc_input = np.full((self.model_max_bars, self.model_enc_seqlen), dtype=int, fill_value=self.pad_token)
        enc_lens = np.zeros((self.model_max_bars,))

        for b, (st, ed) in enumerate(zip(bar_positions[:-1], bar_positions[1:])):
            enc_padding_mask[b, : (ed-st)] = False
            enc_lens[b] = ed - st
            within_bar_events = self.pad_sequence(bar_events[st : ed], self.model_enc_seqlen, self.pad_token)
            within_bar_events = np.array(within_bar_events)

            padded_enc_input[b, :] = within_bar_events[:self.model_enc_seqlen]

        return padded_enc_input, enc_padding_mask, enc_lens

    def generate_queue(self):
        self.queue = []      
        if self.balanced_time:
            print('[preparing data] balanced time signature')
            while len(self.queue) < len(self.pieces):
                class_set = self.class_time_sig[:]
                random.shuffle(class_set)
                self.queue += [self.piece2idx[os.path.join(self.data_dir, self.time2pieces[d][random.randint(0, len(self.time2pieces[d]) - 1)])] for d in class_set]
            self.queue = self.queue[:len(self.pieces)]
        elif self.balanced_density:
            print('[preparing data] balanced note density')
            print('[preparing data] class: ', self.class_density)
            while len(self.queue) < len(self.pieces):
                class_set = self.class_density[:]
                random.shuffle(class_set)
                self.queue += [self.piece2idx[os.path.join(self.data_dir, self.density2pieces[d][random.randint(0, len(self.density2pieces[d]) - 1)])] for d in class_set]
            self.queue = self.queue[:len(self.pieces)]
        else:
            self.queue = [*range(len(self.pieces))]
            if self.shuffle:
                random.shuffle(self.queue)
        print ('[preparing data] queue generated')
    
    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        idx = self.queue[idx]
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        bar_events, st_bar, bar_pos, enc_n_bars = self.get_sample_from_file(idx)
        if self.do_augment:
            bar_events = self.pitch_augment(bar_events)

        if self.use_attr_cls:
            polyph_cls, rfreq_cls = self.get_attr_classes(os.path.basename(self.pieces[idx]), st_bar)
            polyph_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
            rfreq_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
            for i, (b_st, b_ed) in enumerate(zip(bar_pos[:-1], bar_pos[1:])):
                polyph_cls_expanded[b_st:b_ed] = polyph_cls[i]
                rfreq_cls_expanded[b_st:b_ed] = rfreq_cls[i]
        else:
            polyph_cls, rfreq_cls = [0], [0]
            polyph_cls_expanded, rfreq_cls_expanded = [0], [0]

        bar_tokens = convert_event(bar_events, self.event2idx, to_ndarr=False)
        bar_pos = bar_pos.tolist() + [len(bar_tokens)]

        enc_inp, enc_padding_mask, enc_lens = self.get_encoder_input_data(bar_pos, bar_tokens)

        length = len(bar_tokens)
        if self.pad_to_same:
            inp = self.pad_sequence(bar_tokens, self.model_dec_seqlen + 1) 
        else:
            inp = self.pad_sequence(bar_tokens, len(bar_tokens) + 1, pad_value=self.dec_end_pad_value)
        target = np.array(inp[1:], dtype=int)
        inp = np.array(inp[:-1], dtype=int)
        assert len(inp) == len(target)

        return {
            'id': idx,
            # 'piece_id': int(os.path.basename(self.pieces[idx]).replace('.pkl', '')),
            'piece_path': self.pieces[idx],
            'piece_id': os.path.basename(self.pieces[idx]).replace('.pkl', ''),
            'st_bar_id': st_bar,
            'bar_pos': np.array(bar_pos, dtype=int),
            'enc_input': enc_inp,
            'dec_input': inp[:self.model_dec_seqlen],
            'dec_target': target[:self.model_dec_seqlen],
            'polyph_cls': polyph_cls_expanded,
            'rhymfreq_cls': rfreq_cls_expanded,
            'polyph_cls_bar': np.array(polyph_cls),
            'rhymfreq_cls_bar': np.array(rfreq_cls),
            'length': min(length, self.model_dec_seqlen),
            'enc_padding_mask': enc_padding_mask,
            'enc_length': enc_lens,
            'enc_n_bars': enc_n_bars,
            # 'time_sig': bar_events[1]['value'],
            # 'note_density': note_density
        }

class RVQTokensDataset(Dataset):
    def __init__(self, data_dir, pieces=[], model_max_bars=16, 
                    num_tokens=8, codebook_size=1024,
                    balanced_time=False, time2pieces=None, 
                    balanced_density=False, density2pieces=None, 
                    shuffle=True, appoint_st_bar=None,
                    first_token_only=False, first_token_first=False):
        self.data_dir = data_dir
        self.pieces = pieces
        self.pieces = sorted([os.path.join(self.data_dir, p) for p in self.pieces])
        self.piece2idx = {self.pieces[idx]:idx for idx in range(len(self.pieces))}
        
        self.first_token_only = first_token_only
        self.first_token_first = first_token_first
        
        self.model_max_bars = model_max_bars
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.appoint_st_bar = appoint_st_bar
        
        if first_token_only:
            vocab = list(range(1 * codebook_size))
        else:
            vocab = list(range(num_tokens * codebook_size))
        orig_vocab_size = len(vocab)
        self.bos_token = orig_vocab_size
        self.eos_token = orig_vocab_size + 1
        self.pad_token = orig_vocab_size + 2
        self.vocab_size = self.pad_token + 1
        
        self.balanced_time = balanced_time
        self.class_time_sig = ['4/4', '6/8', '2/4', '2/2', '3/4']
        self.time2pieces = time2pieces
        
        self.balanced_density = balanced_density
        self.class_density = ['monophonic', 'polyphonic', 'performance']
        # self.class_density = ['monophonic', 'polyphonic']
        self.density2pieces = density2pieces
        
        self.shuffle = shuffle
        self.generate_queue()
    
    def generate_queue(self):
        self.queue = []      
        if self.balanced_time:
            print('[preparing data] balanced time signature')
            while len(self.queue) < len(self.pieces):
                class_set = self.class_time_sig[:]
                random.shuffle(class_set)
                self.queue += [self.piece2idx[os.path.join(self.data_dir, self.time2pieces[d][random.randint(0, len(self.time2pieces[d]) - 1)])] for d in class_set]
            self.queue = self.queue[:len(self.pieces)]
        elif self.balanced_density:
            print('[preparing data] balanced note density')
            print('[preparing data] class: ', self.class_density)
            while len(self.queue) < len(self.pieces):
                class_set = self.class_density[:]
                random.shuffle(class_set)
                self.queue += [self.piece2idx[os.path.join(self.data_dir, self.density2pieces[d][random.randint(0, len(self.density2pieces[d]) - 1)])] for d in class_set]
            self.queue = self.queue[:len(self.pieces)]
        else:
            self.queue = [*range(len(self.pieces))]
            if self.shuffle:
                random.shuffle(self.queue)
        print ('[preparing data] queue generated')
    
    def pad_sequence(self, seq, maxlen, pad_value=None):
        if pad_value is None:
            pad_value = self.pad_token

        seq.extend([pad_value for _ in range(maxlen - len(seq))])

        return seq    
    
    def get_sample_from_file(self, piece_idx):
        piece_evs = pickle_load(self.pieces[piece_idx])
        assert len(piece_evs) % self.num_tokens == 0
        n_bars = len(piece_evs) // self.num_tokens
        
        if n_bars > self.model_max_bars and self.appoint_st_bar is None:
            picked_st_bar = random.choice(range(n_bars + 1 - self.model_max_bars))
        elif self.appoint_st_bar is not None and self.appoint_st_bar < n_bars - self.model_max_bars:
            picked_st_bar = self.appoint_st_bar
        else:
            picked_st_bar = 0

        if n_bars > self.model_max_bars:
            st = picked_st_bar * self.num_tokens
            ed = (picked_st_bar + self.model_max_bars) * self.num_tokens
            if n_bars == picked_st_bar + self.model_max_bars:
                picked_evs = piece_evs[st:ed] + [self.eos_token]
            else:
                picked_evs = piece_evs[st:ed+1]
            picked_bars = self.model_max_bars
        else:
            picked_evs = self.pad_sequence(piece_evs + [self.eos_token], self.model_max_bars * self.num_tokens + 1)
            picked_bars = n_bars
        assert len(picked_evs) == self.model_max_bars * self.num_tokens + 1

        return picked_evs, picked_st_bar, picked_bars
    
    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        idx = self.queue[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tokens, st_bar, n_bars = self.get_sample_from_file(idx)
        if self.first_token_only:
            tokens = np.array(tokens).reshape(-1, self.num_tokens)[:, 0].tolist()
            bar_pos = np.concatenate(([self.model_max_bars], np.arange(self.model_max_bars)))
        elif self.first_token_first:
            tokens = np.array(tokens).reshape(-1, self.num_tokens).T.reshape(-1).tolist()
            bar_pos = np.concatenate(([self.model_max_bars], np.tile(np.arange(self.model_max_bars), self.num_tokens)))
        else:
            bar_pos = np.concatenate(([self.model_max_bars], np.repeat(np.arange(self.model_max_bars), self.num_tokens)))
            
        tokens = [self.bos_token] + tokens
        inp = np.array(tokens[:-1], dtype=int)
        target = np.array(tokens[1:], dtype=int)
        assert len(inp) == len(target)
        assert len(inp) == len(bar_pos)

        return {
            'id': idx,
            'piece_id': '/'.join(self.pieces[idx].split('/')[-3:]),
            'st_bar_id': st_bar,
            'dec_input': inp,
            'dec_target': target,
            'dec_bar_pos': bar_pos,
            'length': n_bars
        }


if __name__ == "__main__":
    # codes below are for unit test
    dset = REMIFullSongTransformerDataset(
        './remi_dataset', './pickles/remi_vocab.pkl', do_augment=True, use_attr_cls=True,
        model_max_bars=16, model_dec_seqlen=1280, model_enc_seqlen=128, min_pitch=22, max_pitch=107
    )
    print (dset.bar_token, dset.pad_token, dset.vocab_size)
    print ('length:', len(dset))

    # for i in random.sample(range(len(dset)), 100):
    # for i in range(len(dset)):
    #   sample = dset[i]
        # print (i, len(sample['bar_pos']), sample['bar_pos'])
        # print (i)
        # print ('******* ----------- *******')
        # print ('piece: {}, st_bar: {}'.format(sample['piece_id'], sample['st_bar_id']))
        # print (sample['enc_input'][:8, :16])
        # print (sample['dec_input'][:16])
        # print (sample['dec_target'][:16])
        # print (sample['enc_padding_mask'][:32, :16])
        # print (sample['length'])

    dloader = DataLoader(dset, batch_size=4, shuffle=False, num_workers=24)
    for i, batch in enumerate(dloader):
        for k, v in batch.items():
            if torch.is_tensor(v):
                print (k, ':', v.dtype, v.size())
        print ('=====================================\n')
