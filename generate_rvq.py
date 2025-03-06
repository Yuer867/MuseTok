import sys, os, random, time
from copy import deepcopy
sys.path.append('./model')

from dataloader import REMIFullSongTransformerDataset
from model.residualVQ import TransformerResidualVQ

from utils import pickle_load, numpy_to_tensor, tensor_to_numpy, pickle_dump
# Pop1k7
from remi2midi import remi2midi
from tqdm import tqdm

# PDMX
# from remiplus.input_representation import remi2midi  # PDMX-TODO

# HookTheory
# from convert2midi import event_to_midi

import torch
import yaml
import numpy as np
from scipy.stats import entropy

DEFAULT_TEMPO = 110

config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
data_dir = config['data']['data_dir']
vocab_path = config['data']['vocab_path']
data_split = config['data']['test_split']

ckpt_path = sys.argv[2]
out_dir = sys.argv[3]
n_pieces = int(sys.argv[4])
n_samples_per_piece = int(sys.argv[5])

###########################################
# little helpers
###########################################
def word2event(word_seq, idx2event):
    return [ idx2event[w] for w in word_seq ]

def get_beat_idx(event):
    return int(event.split('_')[-1])

class TempoEvent(object):
    def __init__(self, tempo, bar, position, bar_resol, fraction):
        self.tempo = tempo
        self.start_tick = bar * bar_resol + position * (bar_resol // fraction)

###########################################
# sampling utilities
###########################################
def temperatured_softmax(logits, temperature):
    try:
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        assert np.count_nonzero(np.isnan(probs)) == 0
    except:
        print ('overflow detected, use 128-bit')
        logits = logits.astype(np.float128)
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        probs = probs.astype(float)
    return probs

def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def top(probs):
    top_index = np.argmax(probs)
    top_prob = np.max(probs)
    return top_index, top_prob

########################################
# generation
########################################
def get_latent_embedding(model, piece_data):
    # reshape
    batch_inp = piece_data['enc_input'].permute(1, 0).long().to(device)
    batch_padding_mask = piece_data['enc_padding_mask'].bool().to(device)

    # get latent conditioning vectors
    with torch.no_grad():
        piece_latents, _ = model.get_sampled_latent(
            batch_inp, padding_mask=batch_padding_mask
        )

    return piece_latents

def get_latent_indices(model, piece_data):
    # reshape
    batch_inp = piece_data['enc_input'].permute(1, 0).long().to(device)
    batch_padding_mask = piece_data['enc_padding_mask'].bool().to(device)

    # get latent conditioning vectors
    with torch.no_grad():
        _, indices = model.get_sampled_latent(
            batch_inp, padding_mask=batch_padding_mask
        )

    return indices

def add_pos(song):
    new_song = []
    current_pos = None
    for i in range(len(song)):
        if 'Position' in song[i]:
            current_pos = song[i]
        if 'Instrument' in song[i] and 'Position' not in song[i-1]:
            new_song.append(current_pos)
            
        new_song.append(song[i])
    return new_song
            

def generate_on_latent_ctrl_vanilla_truncate(
        model, latents, event2idx, idx2event, 
        max_events=12800, primer=None,
        max_input_len=1280, truncate_len=512,
        time_signature=None
    ):
    latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)

    if primer is None:
        generated = [event2idx['Bar_None']]
    else:
        generated = [event2idx[e] for e in primer]
        latent_placeholder[:len(generated), 0, :] = latents[0].squeeze(0)
    
    target_bars, generated_bars = latents.size(0), 0

    steps = 0
    time_st = time.time()
    cur_pos = 0
    failed_cnt = 0

    cur_input_len = len(generated)
    generated_final = deepcopy(generated)
    entropies = []
    perplexities = []

    while generated_bars < target_bars:
        if len(generated) == 1:
            dec_input = numpy_to_tensor([generated], device=device).long()
        else:
            dec_input = numpy_to_tensor([generated], device=device).permute(1, 0).long()

        latent_placeholder[len(generated)-1, 0, :] = latents[generated_bars]
        dec_seg_emb = latent_placeholder[:len(generated), :]

        # sampling
        with torch.no_grad():
            logits = model.generate(dec_input, dec_seg_emb)
        logits = tensor_to_numpy(logits[0])
        probs = temperatured_softmax(logits, temperature=1.0)
        # word = nucleus(probs, nucleus_p)
        word, prob = top(probs)
        word_event = idx2event[word]

        if 'Beat' in word_event:
            event_pos = get_beat_idx(word_event)
            if not event_pos >= cur_pos:
                print(event_pos, cur_pos)
                print([idx2event[word] for word in generated][-10:])
                input()
                failed_cnt += 1
                print('[info] position not increasing, failed cnt:', failed_cnt)
                if failed_cnt >= 10:
                    print('[FATAL] model stuck, exiting ...')
                    # return generated
                    break
                continue
            else:
                cur_pos = event_pos
                failed_cnt = 0

        if 'Bar' in word_event:
            generated_bars += 1
            cur_pos = 0
            print('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated_final)))
            
        if word_event == 'PAD_None':
            print('[info] generated padding token')
            continue
            # break
        
        if 'Time_Signature' in word_event:
            if word_event != time_signature:
                print('[info] wrong time signature {}, enforce to {}'.format(word_event.split('_')[2], time_signature.split('_')[2]))
                word = event2idx[time_signature]
                
        if word_event == 'EOS_None':
            generated_bars += 1
            if generated_bars == target_bars - 1:
                generated.append(event2idx['Bar_None'])
                print('[info] gotten eos')
                break
            else:
                cur_pos = 0
                word = event2idx['Bar_None']
                print('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated_final)))
                print('[info] gotten eos before generating all bars, continue')
            
        if len(generated) > max_events:
            generated_bars += 1
            generated.append(event2idx['Bar_None'])
            print('[info] gotten eos')
            break

        generated.append(word)
        generated_final.append(word)
        entropies.append(entropy(probs))
        perplexities.append(prob)

        cur_input_len += 1
        steps += 1

        assert cur_input_len == len(generated)
        if cur_input_len == max_input_len:
            generated = generated[-truncate_len:]
            latent_placeholder[:len(generated)-1, 0, :] = latent_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0, :]

            print('[info] reset context length: cur_len: {}, accumulated_len: {}, truncate_range: {} ~ {}'.format(
                cur_input_len, len(generated_final), cur_input_len-truncate_len, cur_input_len-1
            ))
            cur_input_len = len(generated)
            break

    # assert generated_bars == target_bars
    print('-- generated events:', len(generated_final))
    print('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))
    # return generated_final[:-1], time.time() - time_st, np.array(entropies), 
    perplexity = np.exp(-np.sum(np.log(perplexities)) / len(generated_final))
    return generated_final[:-1], time.time() - time_st, perplexity


###########################################
# model evaluation
###########################################
def compute_perplexity(model, latents, inp, target, idx2event):
    assert len(inp) == len(target)
    num_events = len(inp)
    latent_placeholder = torch.zeros(num_events, 1, latents.size(-1)).to(device)
    
    bar_pos = 0
    perplexities = []
    for i in tqdm(range(num_events)):
        events = tensor_to_numpy(inp[:(i+1)])
        if len(events) == 1:
            dec_input = numpy_to_tensor([events], device=device).long()
        else:
            dec_input = numpy_to_tensor([events], device=device).permute(1, 0).long()
        
        latent_placeholder[len(events)-1, 0, :] = latents[bar_pos]
        dec_seg_emb = latent_placeholder[:len(events), :]
        
        with torch.no_grad():
            logits = model.generate(dec_input, dec_seg_emb)
        logits = tensor_to_numpy(logits[0])
        probs = np.exp(logits) / np.sum(np.exp(logits))
        dec_target = int(tensor_to_numpy(target)[i])
        perplexities.append(probs[dec_target])
        
        if dec_target < len(idx2event) and idx2event[dec_target] == 'Bar_None':
            bar_pos += 1
    
    assert len(perplexities) == num_events
    perplexity = np.exp(-np.sum(np.log(perplexities)) / num_events)
    return perplexity


if __name__ == "__main__":
    dset = REMIFullSongTransformerDataset(
        data_dir, vocab_path, 
        do_augment=True,
        model_enc_seqlen=config['data']['enc_seqlen'], 
        model_dec_seqlen=config['generate']['dec_seqlen'],
        model_max_bars=config['generate']['max_bars'],
        pieces=pickle_load(data_split),
        pad_to_same=False, use_attr_cls=config['model']['use_attr_cls'],
        shuffle=False, appoint_st_bar=0
    )
    random.seed(42)
    pieces = random.sample(range(len(dset)), n_pieces)
    selected_pieces = []
    for p in pieces:
        p_data = dset[p]
        p_path = p_data['piece_path']
        if 'PDMX' not in p_path:
            selected_pieces.append(p)
    # pieces = pickle_load('data/data_splits_timeLast/PDMX/test_poly.pkl')
    pieces = pickle_load('data/data_splits_timeLast_strict/all/density2pieces_test.pkl')['monophonic'][:n_pieces]
    # pieces = random.sample(pieces, n_pieces)
    pieces = [dset.piece2idx[os.path.join(config['data']['data_dir'], piece)] for piece in pieces]
    print('[sampled pieces]', pieces)
    
    mconf = config['model']
    model = TransformerResidualVQ(
        mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
        mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
        mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
        mconf['num_quantizers'], mconf['codebook_size'],
        rvq_type=mconf['rvq_type']
    ).to(device)
    model.eval()
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    times = []
    idx = 0
    orig_piece_perplexities = []
    gene_piece_perplexities = []
    for p in pieces:
        idx += 1
        print('[info] generate {}/{} pieces'.format(idx, len(pieces)))
        
        # fetch test sample
        p_data = dset[p]
        p_id = p_data['piece_id']
        p_bar_id = p_data['st_bar_id']
        p_data['enc_input'] = p_data['enc_input'][:p_data['enc_n_bars']]
        p_data['enc_padding_mask'] = p_data['enc_padding_mask'][:p_data['enc_n_bars']]

        # get the original sample
        orig_song_path = p_data['piece_path']
        print(orig_song_path)
        orig_song_pos = pickle_load(orig_song_path)[0]
        orig_song_events = ['{}_{}'.format(evs['name'], evs['value']) for evs in pickle_load(orig_song_path)[1]]
        orig_song_times = ['{}_{}'.format(evs['name'], evs['value']) for evs in pickle_load(orig_song_path)[1] if evs['name'] == 'Time_Signature']
        if len(list(set(orig_song_times))) != 1:
            print('invalid pieces {} with multiple time signatures'.format(orig_song_path))
        orig_song_time = orig_song_times[0]
        orig_song = orig_song_events[orig_song_pos[p_bar_id]:][:p_data['length']]
        orig_out_file = os.path.join(out_dir, 'id{}_bar{}_orig'.format(p, p_bar_id))
        print('[info] writing to ...', orig_out_file)
        
        # output reference song's MIDI
        print(*orig_song, sep='\n', file=open(orig_out_file + '.txt', 'a'))
        remi2midi(orig_song, orig_out_file + '.mid', enforce_tempo=True, enforce_tempo_val=[TempoEvent(DEFAULT_TEMPO, 0, 0, 4, 4)])
        
        for k in p_data.keys():
            if k == 'piece_id' or k == 'piece_path':
                continue
            if not torch.is_tensor(p_data[k]):
                p_data[k] = numpy_to_tensor(p_data[k], device=device)
            else:
                p_data[k] = p_data[k].to(device)

        p_latents = get_latent_embedding(model, p_data)
        # p_indices = get_latent_indices(model, p_data)
        print(p_latents[0])
        
        orig_perplexity = compute_perplexity(model, p_latents, p_data['dec_input'], p_data['dec_target'], dset.idx2event)
        orig_piece_perplexities.append(orig_perplexity)
        print('[info] orig piece perplexity: {:.4f}'.format(orig_perplexity))

        # piece_entropies = []
        for samp in range(n_samples_per_piece):
            print('[info] piece: {}, bar: {}'.format(p_id, p_bar_id))

            out_file = os.path.join(out_dir, 'id{}_bar{}_sample{:02d}'.format(
                p, p_bar_id, samp + 1))
            
            print('[info] writing to ...', out_file)
            if os.path.exists(out_file + '.txt'):
                print('[info] file exists, skipping ...')
                continue

            song, t_sec, perplexity = generate_on_latent_ctrl_vanilla_truncate(
                                        model, p_latents, 
                                        dset.event2idx, dset.idx2event,
                                        max_events=12800,
                                        max_input_len=config['generate']['max_input_dec_seqlen'], 
                                        # max_input_len=2500,
                                        truncate_len=min(512, config['generate']['max_input_dec_seqlen'] - 32), 
                                        # nucleus_p=config['generate']['nucleus_p'], 
                                        # temperature=config['generate']['temperature'],
                                        time_signature=orig_song_time
                                    )
            times.append(t_sec)

            song = word2event(song, dset.idx2event)
            
            print(*song, sep='\n', file=open(out_file + '.txt', 'a'))
            remi2midi(song, out_file + '.mid', enforce_tempo=True, enforce_tempo_val=[TempoEvent(DEFAULT_TEMPO, 0, 0, 4, 4)])
            
            # print ('[info] piece entropy: {:.4f} (+/- {:.4f})'.format(
            #     entropies.mean(), entropies.std()
            # ))
            # piece_entropies.append(entropies.mean())
            print ('[info] piece perplexity: {:.4f}'.format(perplexity))
            gene_piece_perplexities.append(perplexity)

    print ('[time stats] {} songs, generation time: {:.2f} secs (+/- {:.2f})'.format(
        n_pieces * n_samples_per_piece, np.mean(times), np.std(times)
    ))
    # print ('[entropy] {:.4f} (+/- {:.4f})'.format(
    #     np.mean(piece_entropies), np.std(piece_entropies)
    # ))
    
    print ('[orig perplexity] {:.4f} (+/- {:.4f})'.format(
        np.mean(orig_piece_perplexities), np.std(orig_piece_perplexities)
    ))
    print ('[gene perplexity] {:.4f} (+/- {:.4f})'.format(
        np.mean(gene_piece_perplexities), np.std(gene_piece_perplexities)
    ))