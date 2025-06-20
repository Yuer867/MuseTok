import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from vector_quantize_pytorch import ResidualVQ, ResidualSimVQ, ResidualFSQ

from transformer_encoder import VAETransformerEncoder
from transformer_helpers import (
    weights_init, PositionalEncoding, TokenEmbedding, generate_causal_mask
)

class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_ff, dropout=0.1, activation='relu'):
        super(TransformerEncoder, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

        self.tr_encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, d_ff, dropout, activation
        )
        self.tr_encoder = nn.TransformerEncoder(
            self.tr_encoder_layer, n_layer
        )

    def forward(self, x, padding_mask=None):
        out = self.tr_encoder(x, src_key_padding_mask=padding_mask)
        hidden_out = out[0, :, :]

        return hidden_out


class TransformerDecoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_ff, d_seg_emb, dropout=0.1, activation='relu'):
        super(TransformerDecoder, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_seg_emb = d_seg_emb
        self.dropout = dropout
        self.activation = activation

        self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias=False)
        self.decoder_layers = nn.ModuleList()
        for i in range(n_layer):
            self.decoder_layers.append(
                nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
            )

    def forward(self, x, seg_emb):
        attn_mask = generate_causal_mask(x.size(0)).to(x.device)
        # print (attn_mask.size())
        seg_emb = self.seg_emb_proj(seg_emb)

        out = x
        for i in range(self.n_layer):
            out += seg_emb
            out = self.decoder_layers[i](out, src_mask=attn_mask)

        return out


class TransformerVAE(nn.Module):
    def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, 
                dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
                d_vae_latent, d_embed, n_token,
                enc_dropout=0.1, enc_activation='relu',
                dec_dropout=0.1, dec_activation='relu',
                is_training=True):
        super(TransformerVAE, self).__init__()
        self.enc_n_layer = enc_n_layer
        self.enc_n_head = enc_n_head
        self.enc_d_model = enc_d_model
        self.enc_d_ff = enc_d_ff
        self.enc_dropout = enc_dropout
        self.enc_activation = enc_activation

        self.dec_n_layer = dec_n_layer
        self.dec_n_head = dec_n_head
        self.dec_d_model = dec_d_model
        self.dec_d_ff = dec_d_ff
        self.dec_dropout = dec_dropout
        self.dec_activation = dec_activation  

        self.d_vae_latent = d_vae_latent
        self.d_embed = d_embed
        self.n_token = n_token
        self.is_training = is_training

        self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
        self.emb_dropout = nn.Dropout(self.enc_dropout)
        self.pe = PositionalEncoding(d_embed)
        
        self.encoder = VAETransformerEncoder(
            enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, d_vae_latent, enc_dropout, enc_activation
        )
        self.enc_out_proj = nn.Linear(enc_d_model, d_vae_latent)
        
        self.decoder = TransformerDecoder(
            dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent, dec_dropout, dec_activation
        )
        self.dec_out_proj = nn.Linear(dec_d_model, n_token)
        
        self.apply(weights_init)
        
    
    def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.):
        std = torch.exp(0.5 * logvar).to(mu.device)
        if use_sampling:
            eps = torch.randn_like(std).to(mu.device) * sampling_var
        else:
            eps = torch.zeros_like(std).to(mu.device)

        return eps * std + mu        

    def forward(self, enc_inp, dec_inp, dec_inp_bar_pos, padding_mask=None):
        # [shape of enc_inp] (seqlen_per_bar, bsize, n_bars_per_sample)
        enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
        enc_token_emb = self.token_emb(enc_inp)

        # [shape of dec_inp] (seqlen_per_sample, bsize)
        dec_token_emb = self.token_emb(dec_inp)

        enc_token_emb = enc_token_emb.reshape(enc_inp.size(0), -1, enc_token_emb.size(-1))
        enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))
        dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))

        # [shape of padding_mask] (bsize, n_bars_per_sample, seqlen_per_bar)
        # -- should be `True` for padded indices (i.e., those >= seqlen of the bar), `False` otherwise
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))
            
        _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
        vae_latent = self.reparameterize(mu, logvar)
        vae_latent_reshaped = vae_latent.reshape(enc_bt_size, enc_n_bars, -1)
        
        dec_seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_vae_latent).to(vae_latent.device)
        for n in range(dec_inp.size(1)):
        # [shape of dec_inp_bar_pos] (bsize, n_bars_per_sample + 1)
        # -- stores [[start idx of bar #1, sample #1, ..., start idx of bar #K, sample #1, seqlen of sample #1], [same for another sample], ...]
            for b, (st, ed) in enumerate(zip(dec_inp_bar_pos[n, :-1], dec_inp_bar_pos[n, 1:])):
                dec_seg_emb[st:ed, n, :] = vae_latent_reshaped[n, b, :]

        dec_out = self.decoder(dec_inp, dec_seg_emb)
        dec_logits = self.dec_out_proj(dec_out)

        return mu, logvar, dec_logits
            

    def compute_loss(self, mu, logvar, beta, fb_lambda, dec_logits, dec_tgt):
        recons_loss = F.cross_entropy(
            dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
            ignore_index=self.n_token - 1, reduction='mean'
        ).float()

        kl_raw = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=0)
        kl_before_free_bits = kl_raw.mean()
        kl_after_free_bits = kl_raw.clamp(min=fb_lambda)
        kldiv_loss = kl_after_free_bits.mean()

        return {
            'beta': beta,
            'total_loss': recons_loss + beta * kldiv_loss,
            'kldiv_loss': kldiv_loss,
            'kldiv_raw': kl_before_free_bits,
            'recons_loss': recons_loss
        }
        
    def generate(self, inp, dec_seg_emb, keep_last_only=True):
        token_emb = self.token_emb(inp)
        dec_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
        out = self.decoder(dec_inp, dec_seg_emb)
        out = self.dec_out_proj(out)

        if keep_last_only:
            out = out[-1, ...]

        return out
    
    def get_batch_latent(self, enc_inp, padding_mask=None, latent_from_encoder=False):
        # [shape of enc_inp] (seqlen_per_bar, bsize, n_bars_per_sample)
        enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
        enc_token_emb = self.token_emb(enc_inp)
        enc_token_emb = enc_token_emb.reshape(enc_inp.size(0), -1, enc_token_emb.size(-1))
        enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))

        # [shape of padding_mask] (bsize, n_bars_per_sample, seqlen_per_bar)
        # -- should be `True` for padded indices (i.e., those >= seqlen of the bar), `False` otherwise
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))

        # [shape of enc_out] (bsize, n_bars_per_sample, enc_d_model)
        enc_out_ = self.encoder(enc_inp, padding_mask=padding_mask)
        enc_out = self.enc_out_proj(enc_out_)
        enc_out = enc_out.reshape(enc_bt_size, enc_n_bars, -1)
        
        # [shape of vae_latent] (bsize, n_bars_per_sample, d_vae_latent)
        vae_latent, indices, _ = self.residual_sim_vq(enc_out)
        
        if latent_from_encoder:
            enc_out_ = enc_out_.reshape(enc_bt_size, enc_n_bars, -1)
            return enc_out_, indices
        
        return vae_latent, indices
    
    def get_sampled_latent(self, inp, padding_mask=None):
        token_emb = self.token_emb(inp)
        enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
        
        enc_out = self.encoder(enc_inp, padding_mask=padding_mask)
        enc_out = self.enc_out_proj(enc_out)
        
        if self.rvq_type == 'SimVQ':
            vae_latent, indices, _ = self.residual_sim_vq(enc_out)
        elif self.rvq_type == 'FSQ':
            enc_out = enc_out.unsqueeze(0)
            vae_latent, indices = self.residual_fsq(enc_out)
            print(vae_latent.shape)
            vae_latent = vae_latent.squeeze(0)
            indices = indices.squeeze(0)

        return vae_latent, indices
    
    def get_encoder_latent(self, inp, padding_mask=None):
        token_emb = self.token_emb(inp)
        enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
        
        enc_out = self.encoder(enc_inp, padding_mask=padding_mask)
        enc_out = self.enc_out_proj(enc_out)
        
        return enc_out


class TransformerResidualVQ(nn.Module):
    def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, 
                dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
                d_vae_latent, d_embed, n_token,
                num_quantizers, codebook_size, rotation_trick=True,
                enc_dropout=0.1, enc_activation='relu',
                dec_dropout=0.1, dec_activation='relu',
                is_training=True, rvq_type='simVQ'):
        super(TransformerResidualVQ, self).__init__()
        self.enc_n_layer = enc_n_layer
        self.enc_n_head = enc_n_head
        self.enc_d_model = enc_d_model
        self.enc_d_ff = enc_d_ff
        self.enc_dropout = enc_dropout
        self.enc_activation = enc_activation

        self.dec_n_layer = dec_n_layer
        self.dec_n_head = dec_n_head
        self.dec_d_model = dec_d_model
        self.dec_d_ff = dec_d_ff
        self.dec_dropout = dec_dropout
        self.dec_activation = dec_activation  

        self.d_vae_latent = d_vae_latent
        self.d_embed = d_embed
        self.n_token = n_token
        self.is_training = is_training

        self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
        self.emb_dropout = nn.Dropout(self.enc_dropout)
        self.pe = PositionalEncoding(d_embed)
        
        self.encoder = TransformerEncoder(
            enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, enc_dropout, enc_activation
        )
        self.enc_out_proj = nn.Linear(enc_d_model, d_vae_latent)
        
        self.rvq_type = rvq_type
        if self.rvq_type == 'SimVQ':
            self.residual_sim_vq = ResidualSimVQ(
                dim=d_vae_latent,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                rotation_trick=rotation_trick,
            )
        elif self.rvq_type == 'FSQ':
            self.residual_fsq = ResidualFSQ(
                dim=d_vae_latent,
                num_quantizers=num_quantizers,
                levels=[8, 5, 5, 5]
            )
        elif self.rvq_type == 'RVQ':
            self.residual_vq = ResidualVQ(
                dim=d_vae_latent,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                rotation_trick=rotation_trick,
                kmeans_init=True,
                kmeans_iters=10
            )
            
        # self.residual_sim_vq = ResidualSimVQ(
        #     dim=d_vae_latent,
        #     num_quantizers=num_quantizers,
        #     codebook_size=codebook_size,
        #     rotation_trick=rotation_trick,
        # )
        
        # self.residual_fsq = ResidualFSQ(
        #     dim=d_vae_latent,
        #     num_quantizers=num_quantizers,
        #     levels=[8, 5, 5, 5]
        # )
        
        self.decoder = TransformerDecoder(
            dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent, dec_dropout, dec_activation
        )
        self.dec_out_proj = nn.Linear(dec_d_model, n_token)
        
        self.apply(weights_init)
        
        
    def forward(self, enc_inp, dec_inp, dec_inp_bar_pos, padding_mask=None):
        # [shape of enc_inp] (seqlen_per_bar, bsize, n_bars_per_sample)
        enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
        enc_token_emb = self.token_emb(enc_inp)

        # [shape of dec_inp] (seqlen_per_sample, bsize)
        dec_token_emb = self.token_emb(dec_inp)

        enc_token_emb = enc_token_emb.reshape(enc_inp.size(0), -1, enc_token_emb.size(-1))
        enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))
        dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))

        # [shape of padding_mask] (bsize, n_bars_per_sample, seqlen_per_bar)
        # -- should be `True` for padded indices (i.e., those >= seqlen of the bar), `False` otherwise
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))

        # [shape of enc_out] (bsize, n_bars_per_sample, enc_d_model)
        enc_out = self.encoder(enc_inp, padding_mask=padding_mask)
        enc_out = self.enc_out_proj(enc_out)
        enc_out = enc_out.reshape(enc_bt_size, enc_n_bars, -1)
        
        # [shape of vae_latent] (bsize, n_bars_per_sample, d_vae_latent)
        if self.rvq_type == 'SimVQ':
            vae_latent, indices, commit_loss = self.residual_sim_vq(enc_out)
        elif self.rvq_type == 'RVQ':
            vae_latent, indices, commit_loss = self.residual_vq(enc_out)
        elif self.rvq_type == 'FSQ':
            vae_latent, indices = self.residual_fsq(enc_out)
            commit_loss = None
        else:
            raise ValueError("Wrong Residual Vector Quantization model {}, choose from ['SimVQ', 'FSQ', 'RVQ'].".format(self.rvq_type))

        dec_seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_vae_latent).to(vae_latent.device)
        for n in range(dec_inp.size(1)):
        # [shape of dec_inp_bar_pos] (bsize, n_bars_per_sample + 1)
        # -- stores [[start idx of bar #1, sample #1, ..., start idx of bar #K, sample #1, seqlen of sample #1], [same for another sample], ...]
            for b, (st, ed) in enumerate(zip(dec_inp_bar_pos[n, :-1], dec_inp_bar_pos[n, 1:])):
                dec_seg_emb[st:ed, n, :] = vae_latent[n, b, :]

        dec_out = self.decoder(dec_inp, dec_seg_emb)
        dec_logits = self.dec_out_proj(dec_out)

        return dec_logits, commit_loss
            

    def compute_loss(self, commit_loss, beta, dec_logits, dec_tgt, weighted_loss=False, weight=None):
        if weighted_loss and weight is not None:
            recons_loss = F.cross_entropy(
                dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
                ignore_index=self.n_token - 1, reduction='none')
            recons_loss = torch.mean(recons_loss * weight).float()
        else:
            recons_loss = F.cross_entropy(
                dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
                ignore_index=self.n_token - 1, reduction='mean'
            ).float()

        if commit_loss is not None:
            commit_loss = commit_loss.sum().float()
            
            return {
                'beta': beta,
                'total_loss': recons_loss + beta * commit_loss,
                'commit_loss': commit_loss,
                'recons_loss': recons_loss
            }
        else:
            commit_loss = torch.zeros_like(recons_loss)
            
            return {
                'beta': beta,
                'total_loss': recons_loss,
                'commit_loss': commit_loss,
                'recons_loss': recons_loss
            }
    
    def generate(self, inp, dec_seg_emb, keep_last_only=True):
        token_emb = self.token_emb(inp)
        dec_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
        out = self.decoder(dec_inp, dec_seg_emb)
        out = self.dec_out_proj(out)

        if keep_last_only:
            out = out[-1, ...]

        return out
    
    def get_batch_latent(self, enc_inp, padding_mask=None, latent_from_encoder=False):
        # [shape of enc_inp] (seqlen_per_bar, bsize, n_bars_per_sample)
        enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
        enc_token_emb = self.token_emb(enc_inp)
        enc_token_emb = enc_token_emb.reshape(enc_inp.size(0), -1, enc_token_emb.size(-1))
        enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))

        # [shape of padding_mask] (bsize, n_bars_per_sample, seqlen_per_bar)
        # -- should be `True` for padded indices (i.e., those >= seqlen of the bar), `False` otherwise
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))

        # [shape of enc_out] (bsize, n_bars_per_sample, enc_d_model)
        enc_out_ = self.encoder(enc_inp, padding_mask=padding_mask)
        enc_out = self.enc_out_proj(enc_out_)
        enc_out = enc_out.reshape(enc_bt_size, enc_n_bars, -1)
        
        # [shape of vae_latent] (bsize, n_bars_per_sample, d_vae_latent)
        vae_latent, indices, _ = self.residual_sim_vq(enc_out)
        
        if latent_from_encoder:
            enc_out_ = enc_out_.reshape(enc_bt_size, enc_n_bars, -1)
            return enc_out_, indices
        
        return vae_latent, indices
    
    def get_sampled_latent(self, inp, padding_mask=None):
        token_emb = self.token_emb(inp)
        enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
        
        enc_out = self.encoder(enc_inp, padding_mask=padding_mask)
        enc_out = self.enc_out_proj(enc_out)
        
        if self.rvq_type == 'SimVQ':
            vae_latent, indices, _ = self.residual_sim_vq(enc_out)
        elif self.rvq_type == 'FSQ':
            enc_out = enc_out.unsqueeze(0)
            vae_latent, indices = self.residual_fsq(enc_out)
            print(vae_latent.shape)
            vae_latent = vae_latent.squeeze(0)
            indices = indices.squeeze(0)

        return vae_latent, indices
    
    def get_encoder_latent(self, inp, padding_mask=None):
        token_emb = self.token_emb(inp)
        enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
        
        enc_out = self.encoder(enc_inp, padding_mask=padding_mask)
        enc_out = self.enc_out_proj(enc_out)
        
        return enc_out


class TransformerGenerator(nn.Module):
    def __init__(self, dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
                d_vae_latent, d_embed, n_token,
                dec_dropout=0.1, dec_activation='relu',
                is_training=True):
        super(TransformerGenerator, self).__init__()
        self.dec_n_layer = dec_n_layer
        self.dec_n_head = dec_n_head
        self.dec_d_model = dec_d_model
        self.dec_d_ff = dec_d_ff
        self.dec_dropout = dec_dropout
        self.dec_activation = dec_activation

        self.d_vae_latent = d_vae_latent
        self.d_embed = d_embed
        self.n_token = n_token
        self.is_training = is_training

        self.token_emb = TokenEmbedding(n_token, d_embed, dec_d_model)
        self.emb_dropout = nn.Dropout(self.dec_dropout)
        self.pe = PositionalEncoding(d_embed)

        self.decoder = TransformerDecoder(
            dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent, dec_dropout, dec_activation
        )
        self.dec_out_proj = nn.Linear(dec_d_model, n_token)
        
        self.apply(weights_init)
        
        
    def forward(self, vae_latent, dec_inp, dec_inp_bar_pos):
        # [shape of dec_inp] (seqlen_per_sample, bsize)
        dec_token_emb = self.token_emb(dec_inp)
        dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))

        dec_seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_vae_latent).to(vae_latent.device)
        for n in range(dec_inp.size(1)):
        # [shape of dec_inp_bar_pos] (bsize, n_bars_per_sample + 1)
        # -- stores [[start idx of bar #1, sample #1, ..., start idx of bar #K, sample #1, seqlen of sample #1], [same for another sample], ...]
            for b, (st, ed) in enumerate(zip(dec_inp_bar_pos[n, :-1], dec_inp_bar_pos[n, 1:])):
                dec_seg_emb[st:ed, n, :] = vae_latent[n, b, :]

        dec_out = self.decoder(dec_inp, dec_seg_emb)
        dec_logits = self.dec_out_proj(dec_out)

        return dec_logits
            

    def compute_loss(self, dec_logits, dec_tgt, weighted_loss=False, weight=None):
        if weighted_loss and weight is not None:
            recons_loss = F.cross_entropy(
                dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
                ignore_index=self.n_token - 1, reduction='none')
            recons_loss = torch.mean(recons_loss * weight).float()
        else:
            recons_loss = F.cross_entropy(
                dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
                ignore_index=self.n_token - 1, reduction='mean'
            ).float()
            
        return {'recons_loss': recons_loss}
    
    def generate(self, inp, dec_seg_emb, keep_last_only=True):
        token_emb = self.token_emb(inp)
        dec_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
        out = self.decoder(dec_inp, dec_seg_emb)
        out = self.dec_out_proj(out)

        if keep_last_only:
            out = out[-1, ...]

        return out


class GPT2TokensGenerator(nn.Module):
    def __init__(self, dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
                d_embed, n_token, n_bar, dec_dropout=0.1, dec_activation='relu',
                use_bar_emb=True, is_training=True):
        super(GPT2TokensGenerator, self).__init__()
        self.dec_n_layer = dec_n_layer
        self.dec_n_head = dec_n_head
        self.dec_d_model = dec_d_model
        self.dec_d_ff = dec_d_ff
        self.dec_dropout = dec_dropout
        self.dec_activation = dec_activation

        self.d_embed = d_embed
        self.n_token = n_token
        self.is_training = is_training

        self.token_emb = TokenEmbedding(n_token, d_embed, dec_d_model)
        self.emb_dropout = nn.Dropout(self.dec_dropout)
        self.pe = PositionalEncoding(dec_d_model)
        
        self.n_bar = n_bar
        self.use_bar_emb = use_bar_emb
        if self.use_bar_emb:
            self.bar_emb = TokenEmbedding(n_bar+1, d_embed, dec_d_model)
        else:
            self.bar_emb = None
        
        gpt_config = GPT2Config(
            n_layer = dec_n_layer,
            n_head = dec_n_head,
            n_embd = dec_d_model,
            n_inner = dec_d_ff,
            resid_pdrop = dec_dropout,
            attn_pdrop = dec_dropout,
            max_position_embeddings = 4096,
        )
        self.decoder = nn.ModuleList([GPT2Block(gpt_config, layer_idx=i) for i in range(dec_n_layer)])
        self.dec_out_proj = nn.Linear(dec_d_model, n_token)
        
        self.apply(weights_init)
        
        
    def forward(self, dec_inp, bar_inp=None):
        dec_token_emb = self.token_emb(dec_inp)
        
        if self.use_bar_emb and bar_inp is not None:
            dec_token_emb += self.bar_emb(bar_inp)

        dec_out = self.emb_dropout(dec_token_emb + self.pe(dec_inp.size(1)).permute(1, 0, 2))
        
        for i in range(self.dec_n_layer):
            dec_out = self.decoder[i].forward(dec_out)[0]
        dec_logits = self.dec_out_proj(dec_out)

        return dec_logits
        

    def compute_loss(self, dec_logits, dec_tgt):
        recons_loss = F.cross_entropy(
            dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
            ignore_index=self.n_token - 1, reduction='mean'
        ).float()
        
        return {'recons_loss': recons_loss}
    
    def generate(self, inp, bar_inp, keep_last_only=True):
        token_emb = self.token_emb(inp)
        if self.use_bar_emb and bar_inp is not None:
            token_emb += self.bar_emb(bar_inp)
        dec_out = self.emb_dropout(token_emb + self.pe(inp.size(1)).permute(1, 0, 2))
        
        for i in range(self.dec_n_layer):
            dec_out = self.decoder[i].forward(dec_out)[0]
        out = self.dec_out_proj(dec_out)

        if keep_last_only:
            out = out[:, -1, :]

        return out


if __name__ == "__main__":
    device = 'cuda:3'
    model = TransformerResidualVQ(
        enc_n_layer=12, enc_n_head=8, enc_d_model=512, enc_d_ff=2048, 
        dec_n_layer=12, dec_n_head=8, dec_d_model=512, dec_d_ff=2048,
        num_quantizers=8, codebook_size=128,
        d_vae_latent=128, d_embed=512, n_token=300,
        is_training=True).to(device)
    model.train()
    
    enc_inp = torch.randint(0, 100, (128, 16, 16)).to(device)
    dec_inp = torch.randint(0, 100, (1280, 16)).to(device)
    dec_inp_bar_pos = torch.randint(0, 100, (16, 17)).to(device)
    padding_mask = torch.rand((16, 16, 128)).to(device)
    logits, commit_loss = model(enc_inp, dec_inp, dec_inp_bar_pos, padding_mask=padding_mask)
    print(logits.shape)
    print(commit_loss.shape)
    print(commit_loss)
    print(commit_loss.sum())
