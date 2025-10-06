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

def convert_event(event_seq, event2idx, to_ndarr=True):
    if isinstance(event_seq[0], dict):
        event_seq = [event2idx['{}_{}'.format(e['name'], e['value'])] for e in event_seq]
    else:
        event_seq = [event2idx[e] for e in event_seq]

    if to_ndarr:
        return np.array(event_seq)
    else:
        return event_seq


class MuseTokEncoder:
    def __init__(self, model, device='cuda', vocab_file='data/dictionary.pkl', 
                model_enc_seqlen=128, model_max_bars=16):
        self.device = device
        self.model = model
        
        self.vocab_file = vocab_file
        self.read_vocab()

        self.model_enc_seqlen = model_enc_seqlen
        self.model_max_bars = model_max_bars

    def read_vocab(self):
        vocab = pickle_load(self.vocab_file)[0]
        self.idx2event = pickle_load(self.vocab_file)[1]
        orig_vocab_size = len(vocab)
        self.event2idx = vocab
        self.bar_token = self.event2idx['Bar_None']
        self.eos_token = self.event2idx['EOS_None']
        self.pad_token = orig_vocab_size
        self.vocab_size = self.pad_token + 1

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

    def get_segments(self, piece_evs, piece_bar_pos):
        piece_bar_pos.append(len(piece_evs))
        n_segment = math.ceil((len(piece_bar_pos)-1) / self.model_max_bars)
        piece_enc_inp = np.zeros((n_segment, self.model_max_bars, self.model_enc_seqlen))
        piece_enc_padding_mask = np.zeros((n_segment, self.model_max_bars, self.model_enc_seqlen))
        
        for i in range(n_segment):
            st_bar = i * self.model_max_bars
            bar_events, bar_pos = self.get_sample_from_file(piece_evs, piece_bar_pos, st_bar)
            bar_pos = bar_pos.tolist() + [len(bar_events)]
            
            augment_bar_tokens = convert_event(bar_events, self.event2idx, to_ndarr=False)
            enc_inp, enc_padding_mask = self.get_padded_enc_data(bar_pos, augment_bar_tokens)
            
            piece_enc_inp[i] = enc_inp
            piece_enc_padding_mask[i] = enc_padding_mask

        return {
            'n_bar': len(piece_bar_pos)-1,
            'n_segment': n_segment,
            'enc_inp': piece_enc_inp,
            'enc_padding_mask': piece_enc_padding_mask
        }
        
    def encoding(self, p_data, return_latents=False):
        n_bar = p_data['n_bar']
        n_segment = p_data['n_segment']
        enc_inp = p_data['enc_inp']
        enc_padding_mask = p_data['enc_padding_mask']
        
        enc_inp = numpy_to_tensor(enc_inp, device=self.device).permute(2, 0, 1).long()
        enc_padding_mask = numpy_to_tensor(enc_padding_mask, device=self.device).bool()
        
        with torch.no_grad():
            latents, indices = self.model.get_batch_latent(enc_inp, enc_padding_mask, latent_from_encoder=False)

        indices = tensor_to_numpy(indices).reshape(n_segment * self.model_max_bars, -1)[:n_bar, :].astype(int)
        latents = tensor_to_numpy(latents).reshape(n_segment * self.model_max_bars, -1)[:n_bar, :]
        assert indices.shape[0] == n_bar and latents.shape[0] == n_bar

        if return_latents:
            return indices, latents
        else:
            return indices