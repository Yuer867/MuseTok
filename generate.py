import sys, os, random, time
from copy import deepcopy
sys.path.append('./model')

from dataloader import REMIFullSongTransformerDataset
from model.musemorphose import MuseMorphose

from utils import pickle_load, numpy_to_tensor, tensor_to_numpy
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
    return top_index

########################################
# generation
########################################
def get_latent_embedding_fast(model, piece_data, use_sampling=False, sampling_var=0.):
  # reshape
  batch_inp = piece_data['enc_input'].permute(1, 0).long().to(device)
  batch_padding_mask = piece_data['enc_padding_mask'].bool().to(device)

  # get latent conditioning vectors
  with torch.no_grad():
    piece_latents = model.get_sampled_latent(
      batch_inp, padding_mask=batch_padding_mask, 
      use_sampling=use_sampling, sampling_var=sampling_var
    )

  return piece_latents

def generate_on_latent_ctrl_vanilla_truncate(
        model, latents, rfreq_cls, polyph_cls, event2idx, idx2event, 
        max_events=12800, primer=None,
        max_input_len=1280, truncate_len=512, 
        nucleus_p=0.9, temperature=1.2, use_attr_cls=False,
        time_signature=None
      ):
  latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)
  rfreq_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  polyph_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  print ('[info] rhythm cls: {} | polyph_cls: {}'.format(rfreq_cls, polyph_cls))

  if primer is None:
    generated = [event2idx['Bar_None']]
  else:
    generated = [event2idx[e] for e in primer]
    latent_placeholder[:len(generated), 0, :] = latents[0].squeeze(0)
    rfreq_placeholder[:len(generated), 0] = rfreq_cls[0]
    polyph_placeholder[:len(generated), 0] = polyph_cls[0]
    
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

    latent_placeholder[len(generated)-1, 0, :] = latents[ generated_bars ]
    rfreq_placeholder[len(generated)-1, 0] = rfreq_cls[ generated_bars ]
    polyph_placeholder[len(generated)-1, 0] = polyph_cls[ generated_bars ]

    dec_seg_emb = latent_placeholder[:len(generated), :]
    
    if use_attr_cls:
      dec_rfreq_cls = rfreq_placeholder[:len(generated), :]
      dec_polyph_cls = polyph_placeholder[:len(generated), :]
    else:
      dec_rfreq_cls, dec_polyph_cls = None, None

    # sampling
    with torch.no_grad():
      logits = model.generate(dec_input, dec_seg_emb, dec_rfreq_cls, dec_polyph_cls)
    logits = tensor_to_numpy(logits[0])
    probs = temperatured_softmax(logits, temperature)
    # word = nucleus(probs, nucleus_p)
    word = top(probs)
    word_event = idx2event[word]

    if 'Beat' in word_event:
      event_pos = get_beat_idx(word_event)
      if not event_pos >= cur_pos:
        failed_cnt += 1
        print ('[info] position not increasing, failed cnt:', failed_cnt)
        if failed_cnt >= 128:
          print ('[FATAL] model stuck, exiting ...')
          return generated
        continue
      else:
        cur_pos = event_pos
        failed_cnt = 0

    if 'Bar' in word_event:
      generated_bars += 1
      cur_pos = 0
      print ('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated_final)))
    if word_event == 'PAD_None':
      print ('[info] generated padding token')
      # continue
      break
    
    if 'Time Signature' in word_event:
      if word_event != time_signature:
        print('[info] wrong time signature {}, enforce to {}'.format(word_event.split('_')[1], time_signature.split('_')[1]))
        word = event2idx[time_signature]
        
    # if len(generated) > max_events or (word_event == 'EOS_None' and generated_bars == target_bars - 1):
    if len(generated) > max_events or (word_event == '<eos>' and generated_bars == target_bars - 1):
      generated_bars += 1
      generated.append(event2idx['Bar_None'])
      print ('[info] gotten eos')
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
      rfreq_placeholder[:len(generated)-1, 0] = rfreq_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]
      polyph_placeholder[:len(generated)-1, 0] = polyph_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]

      print ('[info] reset context length: cur_len: {}, accumulated_len: {}, truncate_range: {} ~ {}'.format(
        cur_input_len, len(generated_final), cur_input_len-truncate_len, cur_input_len-1
      ))
      cur_input_len = len(generated)
      break

  # assert generated_bars == target_bars
  print ('-- generated events:', len(generated_final))
  print ('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))
  return generated_final[:-1], time.time() - time_st, np.array(entropies)


########################################
# change attribute classes
########################################
def random_shift_attr_cls(n_samples, upper=4, lower=-3):
  return np.random.randint(lower, upper, (n_samples,))


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
  pieces = [14075, 6889, 3467, 13952, 3610, 126, 12477, 7459, 16037, 4588, 7604, 7314, 11696, 4694, 8625, 1100, 15471, 10176, 39, 2615]
  print ('[sampled pieces]', pieces)
  
  mconf = config['model']
  model = MuseMorphose(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
    mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
    d_polyph_emb=mconf['d_polyph_emb'], d_rfreq_emb=mconf['d_rfreq_emb'],
    cond_mode=mconf['cond_mode'], use_attr_cls=mconf['use_attr_cls']
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
    p_data['enc_input'] = p_data['enc_input'][ : p_data['enc_n_bars'] ]
    p_data['enc_padding_mask'] = p_data['enc_padding_mask'][ : p_data['enc_n_bars'] ]

    if mconf['use_attr_cls']:
      orig_p_cls_str = ''.join(str(c) for c in p_data['polyph_cls_bar'])
      orig_r_cls_str = ''.join(str(c) for c in p_data['rhymfreq_cls_bar'])
      orig_p_cls = p_data['polyph_cls_bar']
      orig_r_cls = p_data['rhymfreq_cls_bar']
      print ('[info] orig - rhythm cls: {} | polyph_cls: {}'.format(orig_r_cls, orig_p_cls))
    else:
      orig_p_cls = np.zeros(config['generate']['max_bars'])
      orig_r_cls = np.zeros(config['generate']['max_bars'])

    # get the sample with time signature information
    orig_song_path = os.path.join('../data_events/data_pdmx_remi+_balanced', p_id[2], p_id[3], p_id + '.pkl')
    orig_song_pos = pickle_load(orig_song_path)[0]
    orig_song_events = ['{}_{}'.format(evs['name'], evs['value']) for evs in pickle_load(orig_song_path)[1]]
    orig_song = orig_song_events[orig_song_pos[p_bar_id]:][:p_data['length']+16]

    # orig_song = p_data['dec_input'].tolist()[:p_data['length']]
    # orig_song = word2event(orig_song, dset.idx2event)
    orig_out_file = os.path.join(out_dir, 'id{}_bar{}_orig'.format(
        p, p_bar_id
    ))
    print ('[info] writing to ...', orig_out_file)
    # output reference song's MIDI
    
    #### Pop1k7 ####
    # _, orig_tempo = remi2midi(orig_song, orig_out_file + '.mid', return_first_tempo=True, enforce_tempo=False)
    
    #### PDMX ####
    print (*orig_song, sep='\n', file=open(orig_out_file + '.txt', 'a'))
    pm = remi2midi(orig_song, bpm=120, has_velocity=False)  # PDMX-TODO
    pm.write(orig_out_file + '.mid')
    
    orig_time_sig = orig_song[1]
    print('[info] time signature {}'.format(orig_time_sig.split('_')[1]))
    
    #### HookTheory ####
    # _, orig_tempo = event_to_midi(key='Key_C', events=orig_song, mode='lead_sheet',
    #                 output_midi_path=orig_out_file + '.mid',
    #                 play_chords=True, enforce_tempo=False, return_tempos=True)

    # save metadata of reference song (events & attr classes)
    if mconf['use_attr_cls']:
      print (*orig_song, sep='\n', file=open(orig_out_file + '.txt', 'a'))
      np.save(orig_out_file + '-POLYCLS.npy', p_data['polyph_cls_bar'])
      np.save(orig_out_file + '-RHYMCLS.npy', p_data['rhymfreq_cls_bar'])
    
    for k in p_data.keys():
      if k == 'piece_id' or k == 'time_sig':
        continue
      if not torch.is_tensor(p_data[k]):
        p_data[k] = numpy_to_tensor(p_data[k], device=device)
      else:
        p_data[k] = p_data[k].to(device)

    p_latents = get_latent_embedding_fast(
                  model, p_data, 
                  use_sampling=config['generate']['use_latent_sampling'],
                  sampling_var=config['generate']['latent_sampling_var']
                )
    
    if mconf['use_attr_cls']:
      p_cls_diff = random_shift_attr_cls(n_samples_per_piece)
      r_cls_diff = random_shift_attr_cls(n_samples_per_piece)

    piece_entropies = []
    for samp in range(n_samples_per_piece):
      print ('[info] piece: {}, bar: {}'.format(p_id, p_bar_id))
      
      if mconf['use_attr_cls']:
        p_polyph_cls = (p_data['polyph_cls_bar'] + p_cls_diff[samp]).clamp(0, 7).long()
        p_rfreq_cls = (p_data['rhymfreq_cls_bar'] + r_cls_diff[samp]).clamp(0, 7).long()

        out_file = os.path.join(out_dir, 'id{}_bar{}_sample{:02d}_poly{}_rhym{}'.format(
          p, p_bar_id, samp + 1,
          '+{}'.format(p_cls_diff[samp]) if p_cls_diff[samp] >= 0 else p_cls_diff[samp], 
          '+{}'.format(r_cls_diff[samp]) if r_cls_diff[samp] >= 0 else r_cls_diff[samp]
        ))      
      else:
        out_file = os.path.join(out_dir, 'id{}_bar{}_sample{:02d}'.format(
          p, p_bar_id, samp + 1))
        
      print ('[info] writing to ...', out_file)
      if os.path.exists(out_file + '.txt'):
        print ('[info] file exists, skipping ...')
        continue

      # print (p_polyph_cls, p_rfreq_cls)

      # generate
      # song, t_sec, entropies = generate_on_latent_ctrl_vanilla_truncate(
      #                             model, p_latents, p_rfreq_cls, p_polyph_cls, dset.event2idx, dset.idx2event,
      #                             max_input_len=config['generate']['max_input_dec_seqlen'], 
      #                             truncate_len=min(512, config['generate']['max_input_dec_seqlen'] - 32), 
      #                             nucleus_p=config['generate']['nucleus_p'], 
      #                             temperature=config['generate']['temperature'],
                                  
      #                           )
      # reconstruction
      song, t_sec, entropies = generate_on_latent_ctrl_vanilla_truncate(
                                  model, p_latents, orig_r_cls, orig_p_cls, dset.event2idx, dset.idx2event,
                                  max_events=12800,
                                  max_input_len=config['generate']['max_input_dec_seqlen'], 
                                  truncate_len=min(512, config['generate']['max_input_dec_seqlen'] - 32), 
                                  nucleus_p=config['generate']['nucleus_p'], 
                                  temperature=config['generate']['temperature'],
                                  use_attr_cls=mconf['use_attr_cls'],
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
      pm = remi2midi(song, bpm=120, time_signature=(int(time_sig[0]), int(time_sig[2])), has_velocity=False) # PDMX-TODO
      pm.write(out_file + '.mid')
      
      #### HookTheory ####
      # event_to_midi(key='Key_C', events=song, mode='lead_sheet',
      #               output_midi_path=out_file + '.mid',
      #               play_chords=True, enforce_tempo=True, enforce_tempo_evs=orig_tempo)

      # save metadata of the generation
      if mconf['use_attr_cls']:
        np.save(out_file + '-POLYCLS.npy', tensor_to_numpy(p_polyph_cls))
        np.save(out_file + '-RHYMCLS.npy', tensor_to_numpy(p_rfreq_cls))
        
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