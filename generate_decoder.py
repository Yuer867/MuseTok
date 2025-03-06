import sys, os, random, time
from copy import deepcopy
sys.path.append('./model')

from dataloader import REMIFullSongTransformerDataset
from model.residualVQ import TransformerResidualVQ, TransformerGenerator

from utils import pickle_load, numpy_to_tensor, tensor_to_numpy, pickle_dump
# Pop1k7
# from remi2midi import remi2midi

# PDMX
from remiplus.input_representation import remi2midi  # PDMX-TODO

# HookTheory
# from convert2midi import event_to_midi

import torch
import yaml
import numpy as np
from scipy.stats import entropy

config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
data_dir = config['data']['data_dir']
vocab_path = config['data']['vocab_path']
data_split = config['data']['test_split']
tokenizer_path = config['tokenizer']['pretrained_tokenizer_path']
if tokenizer_path is None:
    raise ValueError('please provide tokenizer path')
else:
    print(tokenizer_path)

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
    # print(cusum_sorted_probs[:10])
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    # print(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def top(probs):
    top_index = np.argmax(probs)
    return top_index

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
        nucleus_p=0.9, temperature=1.2,
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
        probs = temperatured_softmax(logits, temperature)
        word = nucleus(probs, nucleus_p)
        # input()
        # word = top(probs)
        word_event = idx2event[word]

        if 'Beat' in word_event:
            event_pos = get_beat_idx(word_event)
            if not event_pos >= cur_pos:
                failed_cnt += 1
                print('[info] position not increasing, failed cnt:', failed_cnt)
                if failed_cnt >= 128:
                    print('[FATAL] model stuck, exiting ...')
                    return generated
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
        
        if 'Time Signature' in word_event:
            if word_event != time_signature:
                print('[info] wrong time signature {}, enforce to {}'.format(word_event.split('_')[1], time_signature.split('_')[1]))
                word = event2idx[time_signature]
            
        # if len(generated) > max_events or (word_event == 'EOS_None' and generated_bars == target_bars - 1):
        if len(generated) > max_events or (word_event == '<eos>' and generated_bars == target_bars - 1):
            generated_bars += 1
            generated.append(event2idx['Bar_None'])
            print('[info] gotten eos')
            break

        generated.append(word)
        generated_final.append(word)
        entropies.append(entropy(probs))

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
            # break

    assert generated_bars == target_bars
    print('-- generated events:', len(generated_final))
    print('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))
    return generated_final[:-1], time.time() - time_st, np.array(entropies)


if __name__ == "__main__":
    dset = REMIFullSongTransformerDataset(
        data_dir, vocab_path, 
        do_augment=False,
        model_enc_seqlen=config['data']['enc_seqlen'], 
        model_dec_seqlen=config['generate']['dec_seqlen'],
        model_max_bars=config['generate']['max_bars'],
        pieces=pickle_load(data_split),
        pad_to_same=False, use_attr_cls=config['model']['use_attr_cls'],
        shuffle=False, appoint_st_bar=0
    )
    pieces = random.sample(range(len(dset)), n_pieces)
    # pieces = [14075, 6889, 3467, 13952, 3610, 126, 12477, 7459, 16037, 4588, 7604, 7314, 11696, 4694, 8625, 1100, 15471, 10176, 39, 2615]
    pieces = [2615, 10143, 10176, 12133, 1559, 8625, 11675, 15471]
    # pieces = pickle_load('test_monophonic.pkl')[:n_pieces]
    pieces = pickle_load('test_polyphonic.pkl')
    print('[sampled pieces]', pieces)
    
    tconf = config['tokenizer']
    tokenizer = TransformerResidualVQ(
        tconf['enc_n_layer'], tconf['enc_n_head'], tconf['enc_d_model'], tconf['enc_d_ff'],
        tconf['dec_n_layer'], tconf['dec_n_head'], tconf['dec_d_model'], tconf['dec_d_ff'],
        tconf['d_latent'], tconf['d_embed'], dset.vocab_size,
        tconf['num_quantizers'], tconf['codebook_size'],
        rvq_type=tconf['rvq_type']
    ).to(device)
    tokenizer.eval()
    tokenizer.load_state_dict(torch.load(tokenizer_path, map_location='cpu'))
    
    mconf = config['model']
    model = TransformerGenerator(
        mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
        mconf['d_latent'], mconf['d_embed'], dset.vocab_size
    ).to(device)
    model.eval()
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    times = []
    for p in pieces:
        # fetch test sample
        p_data = dset[p]
        p_id = p_data['piece_id']
        p_bar_id = p_data['st_bar_id']
        p_data['enc_input'] = p_data['enc_input'][:p_data['enc_n_bars']]
        p_data['enc_padding_mask'] = p_data['enc_padding_mask'][:p_data['enc_n_bars']]

        # get the sample with time signature information
        orig_song_path = os.path.join('../data_events/data_pdmx_remi+_balanced', p_id[2], p_id[3], p_id + '.pkl')
        orig_song_pos = pickle_load(orig_song_path)[0]
        orig_song_events = ['{}_{}'.format(evs['name'], evs['value']) for evs in pickle_load(orig_song_path)[1]]
        orig_song = orig_song_events[orig_song_pos[p_bar_id]:][:p_data['length']+16]

        # orig_song = p_data['dec_input'].tolist()[:p_data['length']]
        # orig_song = word2event(orig_song, dset.idx2event)
        orig_out_file = os.path.join(out_dir, 'id{}_bar{}_orig'.format(p, p_bar_id))
        print('[info] writing to ...', orig_out_file)
        # output reference song's MIDI

        #### Pop1k7 ####
        # _, orig_tempo = remi2midi(orig_song, orig_out_file + '.mid', return_first_tempo=True, enforce_tempo=False)
        
        #### PDMX ####
        print(*orig_song, sep='\n', file=open(orig_out_file + '.txt', 'a'))
        pm = remi2midi(orig_song, bpm=120, has_velocity=False)  # PDMX-TODO
        pm.write(orig_out_file + '.mid')

        orig_time_sig = orig_song[1]
        print('[info] time signature {}'.format(orig_time_sig.split('_')[1]))
        
        #### HookTheory ####
        # _, orig_tempo = event_to_midi(key='Key_C', events=orig_song, mode='lead_sheet',
        #                 output_midi_path=orig_out_file + '.mid',
        #                 play_chords=True, enforce_tempo=False, return_tempos=True)
        
        for k in p_data.keys():
            if k == 'piece_id' or k == 'time_sig':
                continue
            if not torch.is_tensor(p_data[k]):
                p_data[k] = numpy_to_tensor(p_data[k], device=device)
            else:
                p_data[k] = p_data[k].to(device)

        p_latents = get_latent_embedding(tokenizer, p_data)
        # p_indices = get_latent_indices(model, p_data)
        print(p_latents.shape)
        input()

        piece_entropies = []
        for samp in range(n_samples_per_piece):
            print('[info] piece: {}, bar: {}'.format(p_id, p_bar_id))

            out_file = os.path.join(out_dir, 'id{}_bar{}_sample{:02d}_temp{}_p{}'.format(
                p, p_bar_id, samp + 1, config['generate']['temperature'], config['generate']['nucleus_p']))
            
            print('[info] writing to ...', out_file)
            if os.path.exists(out_file + '.txt'):
                print('[info] file exists, skipping ...')
                continue

            song, t_sec, entropies = generate_on_latent_ctrl_vanilla_truncate(
                                        # model, p_latents, 
                                        tokenizer, p_latents,
                                        dset.event2idx, dset.idx2event,
                                        max_events=12800,
                                        max_input_len=config['generate']['max_input_dec_seqlen'], 
                                        truncate_len=min(512, config['generate']['max_input_dec_seqlen'] - 32), 
                                        nucleus_p=config['generate']['nucleus_p'], 
                                        temperature=config['generate']['temperature'],
                                        time_signature=orig_time_sig
                                    )
            times.append(t_sec)

            song = word2event(song, dset.idx2event)
            
            #### Pop1k7 ####
            # print (*song, sep='\n', file=open(out_file + '.txt', 'a'))
            # remi2midi(song, out_file + '.mid', enforce_tempo=True, enforce_tempo_val=orig_tempo)
            
            #### PDMX ####
            print (*song, sep='\n', file=open(out_file + '.txt', 'a'))
            time_sig = orig_time_sig.split('_')[1]
            # song = add_pos(song)
            pm = remi2midi(song, bpm=120, time_signature=(int(time_sig[0]), int(time_sig[2])), has_velocity=False) # PDMX-TODO
            pm.write(out_file + '.mid')
            
            #### HookTheory ####
            # event_to_midi(key='Key_C', events=song, mode='lead_sheet',
            #               output_midi_path=out_file + '.mid',
            #               play_chords=True, enforce_tempo=True, enforce_tempo_evs=orig_tempo)
            
            print ('[info] piece entropy: {:.4f} (+/- {:.4f})'.format(
                entropies.mean(), entropies.std()
            ))
            piece_entropies.append(entropies.mean())

    print ('[time stats] {} songs, generation time: {:.2f} secs (+/- {:.2f})'.format(
        n_pieces * n_samples_per_piece, np.mean(times), np.std(times)
    ))
    print ('[entropy] {:.4f} (+/- {:.4f})'.format(
        np.mean(piece_entropies), np.std(piece_entropies)
    ))