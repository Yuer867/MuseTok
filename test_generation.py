import sys, os, random, time
from copy import deepcopy
sys.path.append('./model')

from remi2midi import remi2midi
from dataloader import RVQTokensDataset, REMIFullSongTransformerDataset
from model.musetok import GPT2TokensGenerator, TransformerGenerator, TransformerResidualVQ
from utils import pickle_load, numpy_to_tensor, tensor_to_numpy, pickle_dump

import torch
import yaml
import numpy as np
from scipy.stats import entropy

DEFAULT_TEMPO = 110

config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
device = config['training']['device']

tokenizer_path = config['tokenizer']['pretrained_tokenizer_path']
if tokenizer_path is None:
    raise ValueError('please provide tokenizer path')
else:
    print('tokenizer:', tokenizer_path)
    
decoder_path = config['decoder']['pretrained_decoder_path']
if decoder_path is None:
    raise ValueError('please provide decoder path')
else:
    print('decider:', decoder_path)

generator_path = sys.argv[2]
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
        # word = nucleus(probs, nucleus_p)
        # input()
        word = top(probs)
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
        
        # if 'Time Signature' in word_event:
        #     if word_event != time_signature:
        #         print('[info] wrong time signature {}, enforce to {}'.format(word_event.split('_')[1], time_signature.split('_')[1]))
        #         word = event2idx[time_signature]
            
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

    # assert generated_bars == target_bars
    print('-- generated events:', len(generated_final))
    print('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))
    return generated_final[:-1], time.time() - time_st, np.array(entropies)


def generate_tokens(model, primer, primer_n_bar=0, max_bars=16, num_tokens=8, codebook_size=1024, temp=1.2, top_p=0.9, eos=8193):
    generated = [t for t in primer]
    bar_pos = np.concatenate(([max_bars], np.repeat(np.arange(max_bars), num_tokens)))
    target_bars, generated_bars = max_bars, primer_n_bar

    device = next(model.parameters()).device
    steps = 0
    time_st = time.time()
    while generated_bars < target_bars:
        dec_input = numpy_to_tensor([generated], device=device).long()
        dec_bar = numpy_to_tensor([bar_pos[:len(generated)]], device=device).long()

        # sampling
        with torch.no_grad():
            logits = model.generate(dec_input, dec_bar)
        logits = tensor_to_numpy(logits[0])
        probs = temperatured_softmax(logits, temp)
        token = nucleus(probs, top_p)
        
        cur_pos = steps % num_tokens
        if cur_pos * codebook_size <= token < (cur_pos + 1) * codebook_size:
            generated.append(token)
            steps += 1
        elif token == eos:
            print('[info] gotten eos')
            break
        else:
            print('[info] invalid token')
            continue
        
        if len(generated) % num_tokens == 1:
            generated_bars += 1
            print('[info] generated {} bars'.format(len(generated) // num_tokens))

    # assert len(generated) == num_tokens * max_bars + 1
    # assert generated_bars == target_bars
    print('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))

    return generated[1:], time.time() - time_st


if __name__ == "__main__":
    dset_tokens = RVQTokensDataset(
        data_dir=config['data']['data_dir'],
        pieces=pickle_load(config['data']['test_split']),
        # pieces=pickle_load(config['data']['train_split']),
        model_max_bars=config['generate']['max_bars'],
        num_tokens=config['data']['num_quantizers'],
        codebook_size=config['data']['codebook_size'],
        shuffle=False,
        first_token_only=config['data']['first_token_only'],
        first_token_first=config['data']['first_token_first']
    )
    
    dset_music = REMIFullSongTransformerDataset(
        data_dir=config['data_music']['data_dir'], 
        pieces=pickle_load(config['data_music']['test_split']),
        vocab_file=config['data_music']['vocab_path'], 
        model_enc_seqlen=config['data_music']['enc_seqlen'], 
        model_dec_seqlen=config['generate']['dec_seqlen'],
        model_max_bars=config['generate']['max_bars'],
        shuffle=False, do_augment=False
    )
    
    mconf = config['model']
    generator = GPT2TokensGenerator(
        mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
        mconf['d_embed'], dset_tokens.vocab_size, config['data']['max_bars'], use_bar_emb=mconf['use_bar_emb']
    ).to(device)
    generator.eval()
    generator.load_state_dict(torch.load(generator_path, map_location='cpu'))
    print('[info] load generator')
    
    tconf = config['tokenizer']
    tokenizer = TransformerResidualVQ(
        tconf['enc_n_layer'], tconf['enc_n_head'], tconf['enc_d_model'], tconf['enc_d_ff'],
        tconf['dec_n_layer'], tconf['dec_n_head'], tconf['dec_d_model'], tconf['dec_d_ff'],
        tconf['d_latent'], tconf['d_embed'], dset_music.vocab_size,
        tconf['num_quantizers'], tconf['codebook_size'],
        rvq_type=tconf['rvq_type']
    ).to(device)
    tokenizer.eval()
    tokenizer.load_state_dict(torch.load(tokenizer_path, map_location='cpu'))
    print('[info] load tokenizer')
    
    dconf = config['decoder']
    decoder = TransformerGenerator(
        dconf['dec_n_layer'], dconf['dec_n_head'], dconf['dec_d_model'], dconf['dec_d_ff'],
        dconf['d_latent'], dconf['d_embed'], dset_music.vocab_size
    ).to(device)
    decoder.eval()
    # decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    print('[info] load decoder')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    max_bars = config['generate']['max_bars']
    num_tokens = config['generate']['num_quantizers']
    codebook_size = config['generate']['codebook_size']
    pieces = pickle_load('data/data_splits_timeLast_strict/all/density2pieces_test.pkl')['monophonic'][:n_pieces]
    pieces = [dset_tokens.piece2idx[os.path.join(config['data']['data_dir'], piece)] for piece in pieces]
    # pieces = pickle_load('test_polyphonic.pkl')[:n_pieces]
    print(pieces)
    primer_n_bar = 8
    use_prompt = False
    
    times = []
    print('[number of pieces]', n_pieces)
    for p in range(n_pieces):
        p_data = dset_tokens[pieces[p]]
        # orig_song_path = os.path.join('../data_events/data_pdmx_remi+_balanced', p_data['piece_id'])
        # orig_time_sig = pickle_load(orig_song_path)[1][1]['value']
        
        # generate semantic tokens    
        if use_prompt:
            p_prompt = p_data['dec_input'][:(1 + primer_n_bar * num_tokens)]
            print(p_prompt)
            # print(p_data['dec_input'])
        else:
            p_prompt = [dset_tokens.bos_token]
            primer_n_bar = 0
        gen_tokens, t_sec = generate_tokens(
                            generator,
                            max_bars=config['generate']['max_bars'],
                            num_tokens=config['generate']['num_quantizers'],
                            primer=p_prompt,
                            primer_n_bar=primer_n_bar,
                            top_p=config['generate']['nucleus_p'], 
                            temp=config['generate']['temperature'],
                            eos=dset_tokens.eos_token
                            )
        num_bars = len(gen_tokens) // num_tokens
        gen_tokens = np.array(gen_tokens) - np.tile(np.arange(num_tokens), num_bars) * codebook_size
        gen_indices = numpy_to_tensor(gen_tokens, device=device).view(-1, num_tokens).long()
        gen_latents = tokenizer.residual_sim_vq.get_output_from_indices(gen_indices)

        # decode generated tokens to music
        piece_entropies = []
        for samp in range(n_samples_per_piece):
            if use_prompt:
                out_file = os.path.join(out_dir, 'id{}_sample{:02d}_temp{}_p{}_primer{}_test'.format(
                    p, samp + 1, config['generate']['temperature'], config['generate']['nucleus_p'], primer_n_bar))
            else:
                out_file = os.path.join(out_dir, 'id{}_sample{:02d}_temp{}_p{}'.format(
                    p, samp + 1, config['generate']['temperature'], config['generate']['nucleus_p'], primer_n_bar))
            
            print('[info] writing to ...', out_file)
            if os.path.exists(out_file + '.txt'):
                print('[info] file exists, skipping ...')
                continue

            song, t_sec, entropies = generate_on_latent_ctrl_vanilla_truncate(
                                        tokenizer, gen_latents,
                                        dset_music.event2idx, dset_music.idx2event,
                                        max_events=12800,
                                        max_input_len=config['generate']['max_input_dec_seqlen'], 
                                        truncate_len=min(512, config['generate']['max_input_dec_seqlen'] - 32), 
                                        nucleus_p=config['generate']['nucleus_p'], 
                                        temperature=config['generate']['temperature']
                                    )
            times.append(t_sec)

            song = word2event(song, dset_music.idx2event)
            
            print(*song, sep='\n', file=open(out_file + '.txt', 'a'))
            remi2midi(song, out_file + '.mid', enforce_tempo=True, enforce_tempo_val=[TempoEvent(DEFAULT_TEMPO, 0, 0, 4, 4)])
            
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