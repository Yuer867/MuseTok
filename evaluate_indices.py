import torch
import yaml
import sys, os, random, time
from tqdm import tqdm
from collections import defaultdict, Counter
sys.path.append('./model')
import matplotlib.pyplot as plt

from utils import load_txt, pickle_load, numpy_to_tensor, tensor_to_numpy
from dataloader import REMIFullSongTransformerDataset
from model.residualVQ import TransformerResidualVQ

def get_latent_indices(model, piece_data, device):
    # reshape
    batch_inp = piece_data['enc_input'].permute(1, 0).long().to(device)
    batch_padding_mask = piece_data['enc_padding_mask'].bool().to(device)

    # get latent conditioning vectors
    with torch.no_grad():
        vae_latent, indices = model.get_sampled_latent(
            batch_inp, padding_mask=batch_padding_mask
        )

    return vae_latent, indices


config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
ckpt_path = sys.argv[2]
n_pieces = int(sys.argv[3])
threshold = int(sys.argv[4])

data_split = config['data']['train_split']
dset = REMIFullSongTransformerDataset(
    config['data']['data_dir'], 
    config['data']['vocab_path'], 
    do_augment=False,
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['generate']['dec_seqlen'],
    model_max_bars=config['generate']['max_bars'],
    pieces=pickle_load(data_split)[:n_pieces],
    pad_to_same=False, use_attr_cls=config['model']['use_attr_cls'],
    shuffle=False, # appoint_st_bar=0
)

device = config['training']['device']
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

pieces = random.sample(range(len(dset)), n_pieces)

num_quantizers = mconf['num_quantizers']
indices_record = {i:Counter() for i in range(num_quantizers)}

for p in tqdm(pieces):
    p_data = dset[p]
    p_data['enc_input'] = p_data['enc_input'][:p_data['enc_n_bars']]
    p_data['enc_padding_mask'] = p_data['enc_padding_mask'][:p_data['enc_n_bars']]
    
    for k in p_data.keys():
        if k == 'piece_id' or k == 'time_sig':
            continue
        if not torch.is_tensor(p_data[k]):
            p_data[k] = numpy_to_tensor(p_data[k], device=device)
        else:
            p_data[k] = p_data[k].to(device)

    p_latent, p_indices = get_latent_indices(model, p_data, device)
    # p_code = model.residual_fsq.get_codes_from_indices(p_indices)
    # print(p_code.shape)
    # print(p_code[:, 0, :])
    # print(p_latent[0, :5])
    p_indices = tensor_to_numpy(p_indices)
    # print(p_indices)
    # input()
    for i in range(num_quantizers):
        indices_record[i].update(p_indices[:, i].astype(int))
        
print(indices_record[0].total())
n = 5
for i in range(num_quantizers):
    print('Layer {}'.format(i))
    print('most common indices:', indices_record[i].most_common(n))
    print('least common indices:', indices_record[i].most_common()[:-n-1:-1])
    
    num_utility = len(list(indices_record[i]))
    num_high_utility = len([1 for j in indices_record[i].most_common() if j[1] >= threshold])
    num_none_utility = mconf['codebook_size'] - len(list(indices_record[i]))
    print('utility rate: {}, no utility rate: {} \nhigh utility rate: {} (frequency >= {}), low utility rate: {}'.format(
        num_utility / mconf['codebook_size'], 
        num_none_utility / mconf['codebook_size'],
        num_high_utility / mconf['codebook_size'], 
        threshold,
        (num_utility - num_high_utility) / mconf['codebook_size']))
    print()
    
    most_common = indices_record[i].most_common(100)
    ids = [str(i[0]) for i in most_common]
    counts = [i[1] for i in most_common]
    
    plt.bar(ids, counts)
    plt.title('Layer {}'.format(i))
    plt.savefig('figures/layer{}_ids_dist_10k_density_s512.png'.format(i))
    plt.clf()
    plt.show()
        
    

