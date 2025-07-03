import sys, os, time
sys.path.append('./model')

import wandb
import shutil
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

from model.musetok import GPT2TokenGenerator
from dataloader import RVQTokenDataset
from utils import pickle_load

import yaml
config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

mconf = config['model']
tconf = config['tokenizer']
device = config['training']['device']
trained_steps = config['training']['trained_steps']
lr_decay_steps = config['training']['lr_decay_steps']
lr_warmup_steps = config['training']['lr_warmup_steps']
max_lr, min_lr = config['training']['max_lr'], config['training']['min_lr']

ckpt_dir = './ckpt/gen_{}L_{}H_{}D_{}E-16_bars-n{}-s{}-d{}'.format(
            mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['d_embed'], 
            tconf['num_quantizers'], tconf['codebook_size'], tconf['d_latent'])
params_dir = os.path.join(ckpt_dir, 'params/')
optim_dir = os.path.join(ckpt_dir, 'optim/')
pretrained_params_path = config['model']['pretrained_params_path']
pretrained_optim_path = config['model']['pretrained_optim_path']
ckpt_interval = config['training']['ckpt_interval']
val_interval = config['training']['val_interval']


def train_model(epoch, model, dloader, dloader_val, optim, sched):
    model.train()

    print ('[epoch {:03d}] training ...'.format(epoch))
    print ('[epoch {:03d}] # batches = {}'.format(epoch, len(dloader)))
    st = time.time()

    for batch_idx, batch_samples in enumerate(dloader):
        model.zero_grad()
        batch_dec_inp = batch_samples['dec_input'].to(device)
        batch_dec_tgt = batch_samples['dec_target'].to(device)
        batch_dec_bar = batch_samples['dec_bar_pos'].to(device)
        batch_dec_lens = batch_samples['length']

        global trained_steps
        trained_steps += 1
        
        dec_logits = model(batch_dec_inp, batch_dec_bar)
        losses = model.compute_loss(dec_logits, batch_dec_tgt)
        
        # anneal learning rate
        if trained_steps < lr_warmup_steps:
            curr_lr = max_lr * trained_steps / lr_warmup_steps
            optim.param_groups[0]['lr'] = curr_lr
        else:
            sched.step(trained_steps - lr_warmup_steps)

        # clip gradient & update model
        losses['nll_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        
        nll_loss = losses['nll_loss'].item()
        
        # generation accuracy
        pred = torch.argmax(dec_logits, dim=-1) == batch_dec_tgt
        length = batch_dec_lens.numpy()
        acc = np.sum([torch.sum(pred[i, :length[i]]).item() for i in range(len(length))]) / np.sum(length)

        print (' -- epoch {:03d} | batch {:03d}: \n\t * loss = (NLL: {:.4f}), acc = {}, step = {}, time_elapsed = {:.2f} secs'.format(
            epoch, batch_idx, nll_loss, acc, trained_steps, time.time() - st
        ))
        
        wandb.log({f"nll_loss": nll_loss}, step=trained_steps)
        wandb.log({f"acc": acc}, step=trained_steps)

        if not trained_steps % val_interval:
            vallosses = validate(model, dloader_val)
            
            wandb.log({f"nll_loss_val": np.mean(vallosses[0])}, step=trained_steps)
            wandb.log({f"acc_val": np.mean(vallosses[1])}, step=trained_steps)
            
            model.train()

        if not trained_steps % ckpt_interval:
            torch.save(model.state_dict(),
                os.path.join(params_dir, 'step_{:d}-NLL_{:.3f}-model.pt'.format(
                    trained_steps,
                    nll_loss
                ))
            )
            torch.save(optim.state_dict(),
                os.path.join(optim_dir, 'step_{:d}-NLL_{:.3f}-optim.pt'.format(
                    trained_steps,
                    nll_loss, 
                ))
            )

    print ('[epoch {:03d}] training completed\n  -- loss = (NLL: {:.4f})\n  -- time elapsed = {:.2f} secs.'.format(
        epoch, nll_loss, time.time() - st
    ))

def validate(model, dloader):
    model.eval()
    nll_loss = []
    acc_all = []

    print ('[info] validating ...')
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(dloader):
            model.zero_grad()

            batch_dec_inp = batch_samples['dec_input'].to(device)
            batch_dec_tgt = batch_samples['dec_target'].to(device)
            batch_dec_bar = batch_samples['dec_bar_pos'].to(device)
            batch_dec_lens = batch_samples['length']

            dec_logits = model(batch_dec_inp, batch_dec_bar)
            losses = model.compute_loss(dec_logits, batch_dec_tgt)
            
            # generation accuracy
            pred = torch.argmax(dec_logits, dim=-1) == batch_dec_tgt
            length = batch_dec_lens.numpy()
            acc = np.sum([torch.sum(pred[i, :length[i]]).item() for i in range(len(length))]) / np.sum(length)
            
            if not (batch_idx + 1) % 10:
                print ('batch #{}: NLL {}, acc {}'.format(batch_idx + 1, round(losses['nll_loss'].item(), 3), round(acc, 3)))

            nll_loss.append(losses['nll_loss'].item())
            acc_all.append(acc)
    
    return nll_loss, acc_all

if __name__ == "__main__":
    dset = RVQTokenDataset(
        data_dir=config['data']['data_dir'],
        pieces=pickle_load(config['data']['train_split']),
        model_max_bars=config['data']['max_bars'],
        num_tokens=config['data']['num_quantizers'],
        codebook_size=config['data']['codebook_size'],
        balanced_density=config['data']['balanced_density'],
        density2pieces=pickle_load(config['data']['density_path'])
    )
    dset_val = RVQTokenDataset(
        data_dir=config['data']['data_dir'],
        pieces=pickle_load(config['data']['val_split']),
        model_max_bars=config['data']['max_bars'],
        num_tokens=config['data']['num_quantizers'],
        codebook_size=config['data']['codebook_size'],
        shuffle=False
    )
    print ('[info]', '# training samples: {}, # validation samples: {}'.format(len(dset.pieces), len(dset_val)))

    dloader = DataLoader(dset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)
    dloader_val = DataLoader(dset_val, batch_size=config['data']['batch_size'], shuffle=False, num_workers=8)

    model = GPT2TokenGenerator(
        mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
        mconf['d_embed'], dset.vocab_size, config['data']['max_bars'], use_bar_emb=mconf['use_bar_emb']
    ).to(device)
    if pretrained_params_path:
        model.load_state_dict(torch.load(pretrained_params_path))
        print('[info] load pretrained params')
        
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print ('[info] model # params:', n_params)

    opt_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(opt_params, lr=max_lr)
    if pretrained_optim_path:
        optimizer.load_state_dict( torch.load(pretrained_optim_path) )
        print('[info] load pretrained optim')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, lr_decay_steps, eta_min=min_lr
    )

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    if not os.path.exists(optim_dir):
        os.makedirs(optim_dir)
    
    shutil.copy(config_path, os.path.join(ckpt_dir, 'config.yaml'))
    
    run = wandb.init(
        config=dict(**{"n_parameters": n_params}),
        resume="allow", project='MuseTok', group='', name='', id='generator-{}L-{}H-{}D-{}E-16_bars-n{}-s{}-d{}'.format(
            mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['d_embed'], 
            tconf['num_quantizers'], tconf['codebook_size'], tconf['d_latent']))

    for ep in range(config['training']['max_epochs']):
        train_model(ep+1, model, dloader, dloader_val, optimizer, scheduler)
    
    wandb.finish()