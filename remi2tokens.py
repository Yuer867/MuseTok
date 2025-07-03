import sys, os
sys.path.append('./model')
import yaml
import math
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset

from model.musetok import TransformerResidualVQ
from utils import pickle_load, pickle_dump, numpy_to_tensor, tensor_to_numpy

config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
ckpt_path = sys.argv[2]

def convert_event(event_seq, event2idx, to_ndarr=True):
    if isinstance(event_seq[0], dict):
        event_seq = [event2idx['{}_{}'.format(e['name'], e['value'])] for e in event_seq]
    else:
        event_seq = [event2idx[e] for e in event_seq]

    if to_ndarr:
        return np.array(event_seq)
    else:
        return event_seq

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
        else:
            transposed_raw_events.append(ev)
    
    assert len(transposed_raw_events) == len(raw_events)
    return transposed_raw_events

class REMIWholeSequenceDataset(Dataset):
    def __init__(self, data_dir, vocab_file, pieces=[], 
                model_enc_seqlen=128, model_max_bars=16,
                do_augment=False, augment_range=range(-6, 7), 
                min_pitch=21, max_pitch=108):
        self.vocab_file = vocab_file
        self.read_vocab()

        self.data_dir = data_dir
        self.pieces = pieces
        self.build_dataset()

        self.model_enc_seqlen = model_enc_seqlen
        self.model_max_bars = model_max_bars
        
        # augment music sequences by pitch shifting
        self.do_augment = do_augment
        self.augment_range = augment_range
        self.min_pitch, self.max_pitch = min_pitch, max_pitch

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
            self.pieces = sorted(glob(os.path.join(self.data_dir, '*.pkl')))
        else:
            self.pieces = sorted([os.path.join(self.data_dir, p) for p in self.pieces])
        self.piece2idx = {self.pieces[idx]:idx for idx in range(len(self.pieces))}

        self.piece_bar_pos = []

        for i, p in enumerate(self.pieces):
            bar_pos, p_evs = pickle_load(p)
            if not i % 10000:
                print ('[preparing data] now at #{}'.format(i))
            if bar_pos[-1] == len(p_evs):
                print ('piece {}, got appended bar markers'.format(p))
                bar_pos = bar_pos[:-1]
            if len(p_evs) - bar_pos[-1] == 2:
                # got empty trailing bar
                bar_pos = bar_pos[:-1]

            bar_pos.append(len(p_evs))
            self.piece_bar_pos.append(bar_pos)

    def get_sample_from_file(self, piece_evs, piece_bar_pos, picked_st_bar):
        if len(piece_bar_pos) > picked_st_bar + self.model_max_bars:
            picked_evs = piece_evs[ piece_bar_pos[picked_st_bar] : piece_bar_pos[picked_st_bar + self.model_max_bars] ]
            picked_bar_pos = np.array(piece_bar_pos[ picked_st_bar : picked_st_bar + self.model_max_bars ]) - piece_bar_pos[picked_st_bar]
        else:
            picked_evs = piece_evs[piece_bar_pos[picked_st_bar]:]
            picked_bar_pos = np.array(piece_bar_pos[picked_st_bar:] + [piece_bar_pos[-1]] * 
                                        (self.model_max_bars - len(piece_bar_pos) + picked_st_bar)
                                        ) - piece_bar_pos[picked_st_bar]
        assert len(picked_bar_pos) == self.model_max_bars

        return picked_evs, picked_bar_pos

    def pad_sequence(self, seq, maxlen, pad_value=None):
        if pad_value is None:
            pad_value = self.pad_token

        seq.extend([pad_value for _ in range(maxlen - len(seq))])

        return seq

    def pitch_augment(self, bar_events):
        bar_min_pitch, bar_max_pitch = check_extreme_pitch(bar_events)
        
        n_samples = 0
        for n_keys in self.augment_range:
            if bar_min_pitch + n_keys < self.min_pitch or bar_max_pitch + n_keys > self.max_pitch:
                continue
            else:
                augmented_bar_events = transpose_events(bar_events, n_keys)
                n_samples += 1
        return augmented_bar_events

    def get_padded_enc_data(self, bar_positions, bar_events):
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

        return padded_enc_input, enc_padding_mask
    
    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        piece_evs = pickle_load(self.pieces[idx])[1]
        piece_bar_pos = self.piece_bar_pos[idx]
        
        if self.do_augment:
            piece_min_pitch, piece_max_pitch = check_extreme_pitch(piece_evs)
            piece_augment_range = []
            for n_keys in self.augment_range:
                if piece_min_pitch + n_keys < self.min_pitch or piece_max_pitch + n_keys > self.max_pitch:
                    continue
                else:
                    piece_augment_range.append(n_keys)
            if len(piece_augment_range) == 0:
                piece_augment_range = [0]
        else:
            piece_augment_range = [0]
        
        n_segment = math.ceil((len(piece_bar_pos)-1) / self.model_max_bars)
        piece_enc_inp = np.zeros((len(piece_augment_range), n_segment, self.model_max_bars, self.model_enc_seqlen))
        piece_enc_padding_mask = np.zeros((len(piece_augment_range), n_segment, self.model_max_bars, self.model_enc_seqlen))
        
        for i in range(n_segment):
            st_bar = i * self.model_max_bars
            bar_events, bar_pos = self.get_sample_from_file(piece_evs, piece_bar_pos, st_bar)
            bar_pos = bar_pos.tolist() + [len(bar_events)]
            
            for j in range(len(piece_augment_range)):
                n_keys = piece_augment_range[j]
                augment_bar_events = transpose_events(bar_events, n_keys)
                augment_bar_tokens = convert_event(augment_bar_events, self.event2idx, to_ndarr=False)
                enc_inp, enc_padding_mask = self.get_padded_enc_data(bar_pos, augment_bar_tokens)
                
                piece_enc_inp[j][i] = enc_inp
                piece_enc_padding_mask[j][i] = enc_padding_mask
        
        if 'PDMX' in self.pieces[idx]:
            piece_id = '/'.join(self.pieces[idx].split('/')[-3:])[:-4]
            dataset = 'PDMX'
        else:
            piece_id = os.path.basename(self.pieces[idx])[:-4]
            dataset = self.pieces[idx].split('/')[-3]
            assert dataset in ['EMOPIA', 'POP909', 'Pop1k7', 'Multipianomide-Classic', 'Hymnal-Folk', 'Ragtime-perfect-Jazz']
        
        return {
            'id': idx,
            'piece_id': piece_id,
            'dataset': dataset,
            'n_bar': len(piece_bar_pos)-1,
            'n_segment': n_segment,
            'n_augment': len(piece_augment_range),
            'enc_inp': piece_enc_inp,
            'enc_padding_mask': piece_enc_padding_mask
        }


def dump_token_sequence(dset, model, data_dir, device, dump_latents=False,
                        do_augment=False, num_quantizers=8, codebook_size=1024):
    dump_pieces = []
    
    for p in tqdm(range(len(dset))):
        p_data = dset[p]
        n_segment = p_data['n_segment']
        n_augment = p_data['n_augment']
        n_bar = p_data['n_bar']
        
        # data_path = os.path.join(data_dir, p_data['dataset'], 'data_events_timeLast_strict', p_data['piece_id'] + '.pkl')
        # if os.path.exists(data_path):
        #     continue
        
        # for i in range(n_augment):
        #     data_path = os.path.join(data_dir, p_data['piece_id'] + '_{}.pkl'.format(i))
        #     if os.path.exists(data_path):
        #         print('exist')
        #         dump_pieces.append(p_data['piece_id'] + '_{}.pkl'.format(i))
        #         continue

        enc_inp = p_data['enc_inp']
        enc_padding_mask = p_data['enc_padding_mask']
        max_bar, seq_len = enc_inp.shape[2], enc_inp.shape[3]
        
        if do_augment:
            p_indices = np.zeros((n_augment, n_segment * max_bar, num_quantizers))
            
            k = 4
            for s in range(int(np.ceil(n_segment / k))):
                enc_inp_ = enc_inp[:, s*k:(s+1)*k, :, :]
                n_segment_ = enc_inp_.shape[1]
                enc_padding_mask_ = enc_padding_mask[:, s*k:(s+1)*k, :, :]
                enc_inp_ = numpy_to_tensor(enc_inp_.reshape(-1, max_bar, seq_len), device=device).permute(2, 0, 1).long()
                enc_padding_mask_ = numpy_to_tensor(enc_padding_mask_.reshape(-1, max_bar, seq_len), device=device).bool()
                
                with torch.no_grad():
                    _, indices = model.get_batch_latent(enc_inp_, enc_padding_mask_, latent_from_encoder=False)
                
                indices = tensor_to_numpy(indices).reshape(n_augment, n_segment_ * max_bar, -1)
                p_indices[:, s*k * max_bar :(s*k+n_segment_) * max_bar, :] = indices
                
            p_indices = p_indices[:, :n_bar, :]
            p_indices += np.arange(num_quantizers) * codebook_size
            p_indices = p_indices.reshape(n_augment, -1).astype(int).tolist()
            assert len(p_indices) == n_augment
            assert len(p_indices[0]) == n_bar * num_quantizers
            
            for i in range(n_augment):
                data_path = os.path.join(data_dir, p_data['dataset'], 
                                        'data_tokens', p_data['piece_id'] + '_{}.pkl'.format(i))
                dump_pieces.append(os.path.join(p_data['dataset'], 
                                        'data_tokens', p_data['piece_id'] + '_{}.pkl'.format(i)))
                os.makedirs('/'.join(data_path.split('/')[:-1]), exist_ok=True)
                pickle_dump(p_indices[i], data_path)
        
        else:
            enc_inp = numpy_to_tensor(enc_inp.reshape(-1, max_bar, seq_len), device=device).permute(2, 0, 1).long()
            enc_padding_mask = numpy_to_tensor(enc_padding_mask.reshape(-1, max_bar, seq_len), device=device).bool()
            
            with torch.no_grad():
                latents, indices = model.get_batch_latent(enc_inp, enc_padding_mask, latent_from_encoder=False)

            indices = tensor_to_numpy(indices).reshape(n_augment, n_segment * max_bar, -1)[:, :n_bar, :]
            indices += np.arange(num_quantizers) * codebook_size
            indices = indices.reshape(n_augment, -1).astype(int).tolist()
            latents = tensor_to_numpy(latents).reshape(n_augment, n_segment * max_bar, -1)[:, :n_bar, :]
            assert len(indices) == n_augment
            assert len(indices[0]) == n_bar * num_quantizers
            assert n_bar == latents[0].shape[0]
            
            data_path = os.path.join(data_dir, p_data['dataset'], 'data_tokens', p_data['piece_id'] + '.pkl')
            dump_pieces.append(os.path.join(p_data['dataset'], 'data_tokens', p_data['piece_id'] + '.pkl'))
            os.makedirs('/'.join(data_path.split('/')[:-1]), exist_ok=True)

            if dump_latents:
                pickle_dump(latents[0], data_path)
            else:
                pickle_dump(indices[0], data_path)

    return dump_pieces
    
    
def dump_density(files, split):
    density2pieces = pickle_load('data/data_splits_events/all/density2pieces_{}.pkl'.format(split))
    piece2density = {}
    for i in density2pieces['monophonic']:
        piece2density[os.path.basename(i)[:-4]] = 'monophonic'
    for i in density2pieces['contrapuntal']:
        piece2density[os.path.basename(i)[:-4]] = 'contrapuntal'
    for i in density2pieces['polyphonic']:
        piece2density[os.path.basename(i)[:-4]] = 'polyphonic'
    
    density2pieces_new = {'monophonic':[], 'contrapuntal':[], 'polyphonic':[]}
    for file in files:
        if split in ['train']:
            filename = '_'.join(os.path.basename(file)[:-4].split('_')[:-1])
        elif split in ['valid', 'test']:
            filename = os.path.basename(file)[:-4]
        if filename in piece2density:
            density2pieces_new[piece2density[filename]].append(file)
        else:
            print(filename)
    
    assert len(files) == len(density2pieces_new['monophonic']) + \
                        len(density2pieces_new['contrapuntal']) + \
                        len(density2pieces_new['polyphonic'])
    pickle_dump(density2pieces_new, 'data/data_splits_tokens/density2pieces_{}.pkl'.format(split))


def dump_small_set():
    small_set = pickle_load('data/data_splits_events/all/small_valid.pkl')
    new_small_set = []
    for file in small_set:
        new_small_set.append(file.replace('events', 'tokens'))
    pickle_dump(new_small_set, 'data/data_splits_tokens/small_valid.pkl')


if __name__ == "__main__":
    # dataset
    train_pieces = pickle_load(config['data']['train_split'])
    valid_pieces = pickle_load(config['data']['val_split'])
    test_pieces = pickle_load(config['data']['test_split'])
    print ('[info]', '# training samples: {}'.format(len(train_pieces)))
    print('[info]', '# validation samples: {}'.format(len(valid_pieces)))
    print('[info]', '# test samples: {}'.format(len(test_pieces)))
    
    train_dset = REMIWholeSequenceDataset(
        data_dir=config['data']['data_dir'], 
        vocab_file=config['data']['vocab_path'], 
        pieces=train_pieces,
        model_enc_seqlen=config['data']['enc_seqlen'], 
        model_max_bars=config['data']['max_bars'],
        do_augment=True
    )
    val_dset = REMIWholeSequenceDataset(
        data_dir=config['data']['data_dir'], 
        vocab_file=config['data']['vocab_path'], 
        pieces=valid_pieces,
        model_enc_seqlen=config['data']['enc_seqlen'], 
        model_max_bars=config['data']['max_bars'],
        do_augment=False
    )
    test_dset = REMIWholeSequenceDataset(
        data_dir=config['data']['data_dir'], 
        vocab_file=config['data']['vocab_path'], 
        pieces=test_pieces,
        model_enc_seqlen=config['data']['enc_seqlen'], 
        model_max_bars=config['data']['max_bars'],
        do_augment=False
    )
    
    # tokenizer
    device = config['training']['device']
    mconf = config['model']
    model = TransformerResidualVQ(
        mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
        mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
        mconf['d_latent'], mconf['d_embed'], train_dset.vocab_size,
        mconf['num_quantizers'], mconf['codebook_size'],
        rotation_trick=mconf['rotation_trick'], rvq_type=mconf['rvq_type']
    ).to(device)
    model.eval()
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    print('[info] successfully load tokenizer')

    num_quantizers = mconf['num_quantizers']
    codebook_size = mconf['codebook_size']
    data_dir = 'data'
    os.makedirs(os.path.join(data_dir, 'data_splits_tokens'), exist_ok=True)
    
    # training data
    print('[info] dump learned tokens of training set to {}'.format(data_dir))
    print('[info] augmentation = {}'.format(train_dset.do_augment))
    train_pieces = dump_token_sequence(train_dset, model, data_dir, device, 
                                        dump_latents=False,
                                        do_augment=True,
                                        num_quantizers=num_quantizers, 
                                        codebook_size=codebook_size)
    print('num of training samples:', len(train_pieces))
    pickle_dump(train_pieces, os.path.join(data_dir, 'data_splits_tokens/all_train.pkl'))
    dump_density(train_pieces, split='train')
    
    # validation data
    print('dump learned tokens of validation set to {}'.format(data_dir))
    print('[info] augmentation = {}'.format(val_dset.do_augment))
    val_pieces = dump_token_sequence(val_dset, model, data_dir, device, 
                                        dump_latents=False,
                                        do_augment=False,
                                        num_quantizers=num_quantizers, 
                                        codebook_size=codebook_size)
    print('num of validation samples:', len(val_pieces))
    pickle_dump(val_pieces, os.path.join(data_dir, 'data_splits_tokens/all_valid.pkl'))
    dump_density(val_pieces, split='valid')
    dump_small_set()
    
    # test data
    print('dump learned tokens of test set to {}'.format(data_dir))
    print('[info] augmentation = {}'.format(test_dset.do_augment))
    test_pieces = dump_token_sequence(test_dset, model, data_dir, device, 
                                        dump_latents=False,
                                        do_augment=False,
                                        num_quantizers=num_quantizers, 
                                        codebook_size=codebook_size)
    print('num of test samples:', len(test_pieces))
    pickle_dump(test_pieces, os.path.join(data_dir, 'data_splits_tokens/all_test.pkl'))
    dump_density(test_pieces, split='test')
    