import sys, os, time
sys.path.append('./model')

from model.residualVQ import TransformerResidualVQ, TransformerGenerator
from dataloader import REMIFullSongTransformerDataset
from torch.utils.data import DataLoader

from utils import pickle_load, remove
from torch import nn, optim
import torch
import numpy as np
import wandb
import shutil

import yaml
config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

tconf = config['tokenizer']
mconf = config['model']

device = config['training']['device']
trained_steps = config['training']['trained_steps']
lr_decay_steps = config['training']['lr_decay_steps']
lr_warmup_steps = config['training']['lr_warmup_steps']
max_lr, min_lr = config['training']['max_lr'], config['training']['min_lr']

# ckpt_dir = config['training']['ckpt_dir']
ckpt_dir = './ckpt/dec_{}L_{}H_{}D-16_bars-seqlen_1280-pdmx-n{}-s{}-d{}-density'.format(
            mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'],
            tconf['num_quantizers'], tconf['codebook_size'], tconf['d_latent'])
params_dir = os.path.join(ckpt_dir, 'params/')
optim_dir = os.path.join(ckpt_dir, 'optim/')
pretrained_tokenizer_path = tconf['pretrained_tokenizer_path']
pretrained_params_path = config['model']['pretrained_params_path']
pretrained_optim_path = config['model']['pretrained_optim_path']
ckpt_interval = config['training']['ckpt_interval']
log_interval = config['training']['log_interval']
val_interval = config['training']['val_interval']

recons_loss_ema = 0.

def log_epoch(log_file, log_data, is_init=False):
    if is_init:
        with open(log_file, 'w') as f:
            f.write('{:4} {:8} {:12} {:12}\n'.format('ep', 'steps', 'recons_loss', 'ep_time'))

    with open(log_file, 'a') as f:
        f.write('{:<4} {:<8} {:<12} {:<12}\n'.format(
            log_data['ep'], log_data['steps'], round(log_data['recons_loss'], 5), round(log_data['time'], 2)
        ))

def compute_loss_ema(ema, batch_loss, decay=0.95):
    if ema == 0.:
        return batch_loss
    else:
        return batch_loss * (1 - decay) + ema * decay

def train_model(epoch, tokenizer, model, dloader, dloader_val, optim, sched, val_rounds):
    model.train()

    print ('[epoch {:03d}] training ...'.format(epoch))
    print ('[epoch {:03d}] # batches = {}'.format(epoch, len(dloader)))
    st = time.time()

    for batch_idx, batch_samples in enumerate(dloader):
        model.zero_grad()
        batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
        batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
        batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)
        batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
        batch_inp_lens = batch_samples['length']
        batch_padding_mask = batch_samples['enc_padding_mask'].to(device)

        global trained_steps
        trained_steps += 1
        
        vae_latent, _ = tokenizer.get_batch_latent(batch_enc_inp, batch_padding_mask)
        dec_logits = model(vae_latent, batch_dec_inp, batch_inp_bar_pos)
        losses = model.compute_loss(dec_logits, batch_dec_tgt)

        # reconstruction accuracy
        pred = torch.argmax(dec_logits, dim=-1) == batch_dec_tgt
        length = batch_inp_lens.numpy()
        acc = np.sum([torch.sum(pred[:(length[i]-1), i]).item() for i in range(len(length))]) / np.sum(length)
        
        # anneal learning rate
        if trained_steps < lr_warmup_steps:
            curr_lr = max_lr * trained_steps / lr_warmup_steps
            optim.param_groups[0]['lr'] = curr_lr
        else:
            sched.step(trained_steps - lr_warmup_steps)

        # clip gradient & update model
        losses['recons_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()

        global recons_loss_ema
        recons_loss_ema = compute_loss_ema(recons_loss_ema, losses['recons_loss'].item())

        print (' -- epoch {:03d} | batch {:03d}: \n\t * loss = (RC: {:.4f}), acc = {}, step = {}, time_elapsed = {:.2f} secs'.format(
            epoch, batch_idx, recons_loss_ema, acc, trained_steps, time.time() - st
        ))
        
        wandb.log({f"recons_loss": recons_loss_ema}, step=trained_steps)
        wandb.log({f"acc": acc}, step=trained_steps)

        if not trained_steps % log_interval:
            log_data = {
                'ep': epoch,
                'steps': trained_steps,
                'recons_loss': recons_loss_ema,
                'acc': acc,
                'time': time.time() - st
            }
            log_epoch(
                os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
            )

        if not trained_steps % val_interval:
            vallosses = validate(tokenizer, model, dloader_val, n_rounds=val_rounds)
            with open(os.path.join(ckpt_dir, 'valloss.txt'), 'a') as f:
                f.write('[step {}] RC: {:.4f} | ACC: {:.4f} [val] | RC: {:.4f} | ACC: {:.4f}\n'.format(
                    trained_steps, 
                    recons_loss_ema, 
                    acc,
                    np.mean(vallosses[0]),
                    np.mean(vallosses[1])
                ))
            
            wandb.log({f"recons_loss_val": np.mean(vallosses[0])}, step=trained_steps)
            wandb.log({f"acc_val": np.mean(vallosses[1])}, step=trained_steps)
            
            model.train()

        if not trained_steps % ckpt_interval:
            torch.save(model.state_dict(),
                os.path.join(params_dir, 'step_{:d}-RC_{:.3f}-model.pt'.format(
                    trained_steps,
                    recons_loss_ema
                ))
            )
            torch.save(optim.state_dict(),
                os.path.join(optim_dir, 'step_{:d}-RC_{:.3f}-optim.pt'.format(
                    trained_steps,
                    recons_loss_ema, 
                ))
            )

    print ('[epoch {:03d}] training completed\n  -- loss = (RC: {:.4f})\n  -- time elapsed = {:.2f} secs.'.format(
        epoch, recons_loss_ema, time.time() - st
    ))
    log_data = {
        'ep': epoch,
        'steps': trained_steps,
        'recons_loss': recons_loss_ema,
        'acc': acc,
        'time': time.time() - st
    }
    log_epoch(
        os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
    )

def validate(tokenizer, model, dloader, n_rounds=8):
    model.eval()
    loss_rec = []
    acc_all = []

    print ('[info] validating ...')
    with torch.no_grad():
        for i in range(n_rounds):
            print ('[round {}]'.format(i+1))

            for batch_idx, batch_samples in enumerate(dloader):
                model.zero_grad()

                batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
                batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
                batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)
                batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
                batch_inp_lens = batch_samples['length']
                batch_padding_mask = batch_samples['enc_padding_mask'].to(device)

                vae_latent, _ = tokenizer.get_batch_latent(batch_enc_inp, batch_padding_mask)
                dec_logits = model(vae_latent, batch_dec_inp, batch_inp_bar_pos)
                losses = model.compute_loss(dec_logits, batch_dec_tgt)
                
                # reconstruction accuracy
                pred = torch.argmax(dec_logits, dim=-1) == batch_dec_tgt
                length = batch_inp_lens.numpy()
                acc = np.sum([torch.sum(pred[:(length[i]-1), i]).item() for i in range(len(length))]) / np.sum(length)
                
                if not (batch_idx + 1) % 10:
                    print ('batch #{}: RC {}, acc {}'.format(batch_idx + 1, round(losses['recons_loss'].item(), 3), round(acc, 3)))

                loss_rec.append(losses['recons_loss'].item())
                acc_all.append(acc)
    
    return loss_rec, acc_all

if __name__ == "__main__":
    dset = REMIFullSongTransformerDataset(
        config['data']['data_dir'], config['data']['vocab_path'], 
        do_augment=True,  # TODO
        model_enc_seqlen=config['data']['enc_seqlen'], 
        model_dec_seqlen=config['data']['dec_seqlen'], 
        model_max_bars=config['data']['max_bars'],
        pieces=pickle_load(config['data']['train_split']),
        pad_to_same=True, use_attr_cls=config['model']['use_attr_cls'],
        balanced_time=config['data']['balanced_time'],
        time2pieces=pickle_load(config['data']['time_path']),
        balanced_density=config['data']['balanced_density'],
        density2pieces=pickle_load(config['data']['density_path'])
    )
    dset_val = REMIFullSongTransformerDataset(
        config['data']['data_dir'], config['data']['vocab_path'], 
        do_augment=False, 
        model_enc_seqlen=config['data']['enc_seqlen'], 
        model_dec_seqlen=config['data']['dec_seqlen'], 
        model_max_bars=config['data']['max_bars'],
        pieces=pickle_load(config['data']['val_split'])[:2000],
        pad_to_same=True, use_attr_cls=config['model']['use_attr_cls']
    )
    print ('[info]', '# training samples:', len(dset.pieces))

    dloader = DataLoader(dset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=8)
    dloader_val = DataLoader(dset_val, batch_size=config['data']['batch_size'], shuffle=False, num_workers=8)

    tokenizer = TransformerResidualVQ(
        tconf['enc_n_layer'], tconf['enc_n_head'], tconf['enc_d_model'], tconf['enc_d_ff'],
        tconf['dec_n_layer'], tconf['dec_n_head'], tconf['dec_d_model'], tconf['dec_d_ff'],
        tconf['d_latent'], tconf['d_embed'], dset.vocab_size,
        tconf['num_quantizers'], tconf['codebook_size'],
        rotation_trick=tconf['rotation_trick'],
        rvq_type=tconf['rvq_type']
    ).to(device)
    if pretrained_tokenizer_path:
        tokenizer.load_state_dict(torch.load(pretrained_tokenizer_path))
        print('[info] load pretrained tokenizer')

        for param in tokenizer.parameters():
            param.requires_grad = False
        for name, param in tokenizer.named_parameters():
            assert param.requires_grad == False
        print('[info] freeze tokenizer')

    model = TransformerGenerator(
        mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
        mconf['d_latent'], mconf['d_embed'], dset.vocab_size
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
        resume="allow", project='MMPDecoderOnly', group='', name='', id='decoder-{}L-{}H-{}D-pdmx-balanced-n{}-s{}-d{}-density'.format(
            mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'],
            tconf['num_quantizers'], tconf['codebook_size'], tconf['d_latent']
            ))

    for ep in range(config['training']['max_epochs']):
        train_model(ep+1, tokenizer, model, dloader, dloader_val, optimizer, scheduler, val_rounds=config['training']['val_rounds'])
    
    wandb.finish()