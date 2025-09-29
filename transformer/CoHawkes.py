import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, SANNetwork


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    return padding_mask.unsqueeze(1).expand(-1, len_q, -1)


def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    return subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)


class Encoder(nn.Module):
    def __init__(
            self,
            num_types, d_covar, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_covar = d_covar

        self.position_vec = torch.tensor(
            [math.pow(1000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)
        self.cov_emb = nn.Linear(d_covar, d_model)

        self.cov_intra_att = SANNetwork(input_size=d_covar,
                                        num_classes=4,
                                        hidden_layer_size=int(d_model),
                                        num_heads=8)

        self.cov_his_outer_att = SANNetwork(input_size=2*d_model,
                                            num_classes=4,
                                            hidden_layer_size=2*d_model,
                                            num_heads=4)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=True)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def tensorized_cov(self, covariate, non_pad_mask):
        batch, seq_len = len(covariate), len(covariate[0])
        results = torch.zeros(size=(batch, seq_len, self.d_covar), device=covariate[0][0].device)
        for b_id, seq in enumerate(covariate):
            len_seq = len(seq)
            results[b_id, :, :] = torch.cat(seq).reshape(len_seq, self.d_covar)
        return results * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask, covariates):
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask = (slf_attn_mask_keypad.type_as(slf_attn_mask_subseq) + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        event_enc = self.event_emb(event_type)
        covar_enc = self.cov_emb(covariates)

        assert not torch.isnan(tem_enc).any()
        if torch.isnan(covar_enc).any():
            assert not torch.isnan(covariates).any()
            print(torch.mean(covar_enc))
            print('IS any weight NAN?', torch.isnan(self.cov_emb.weight))
            assert False

        enc_output = covar_enc + event_enc
        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
        return enc_output


class Predictor(nn.Module):
    def __init__(self, dim, num_types):
        super().__init__()
        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        return self.linear(data) * non_pad_mask


class marker_predictor(nn.Module):
    def __init__(self, dim, num_types):
        super().__init__()
        self.ts = nn.Sequential(nn.Linear(dim, num_types))

    def forward(self, data, non_pad_mask):
        return self.ts(data) * non_pad_mask


class Transformer(nn.Module):
    def __init__(
            self,
            num_types, d_covar, d_model=256, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, mixture_dim=1):
        super().__init__()
        self.encoder = Encoder(num_types, d_covar, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout)

        self.linear = nn.Linear(d_model, num_types)
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.time_predictor = Predictor(d_model, 1)
        self.type_predictor = Predictor(int(2 * d_model), num_types)

        self.d_model = d_model
        self.d_covar = d_covar

        self.w_network = nn.Sequential(
            nn.Linear(d_model, mixture_dim),
            nn.Softmax(dim=2)
        )
        self.std_network = nn.Sequential(
            nn.Linear(d_model, mixture_dim),
            nn.Softplus()
        )
        self.mean_network = nn.Sequential(
            nn.Linear(d_model, mixture_dim)
        )

        self.a = nn.Parameter(torch.tensor(-1.0))
        self.b = nn.Parameter(torch.tensor(-1e-3))

    def log_prob_in_mixtureLogNormal(self, weight, covariance, mean, time, pred_mask):
        assert (torch.sum(weight, axis=-1) - 1.0 < 1e-5).all()
        assert (covariance > 0.0).all()

        time[time <= 0] = 1.0
        log_time = torch.log(time)

        prob = weight / ((2 * torch.pi * covariance) ** 0.5 * time.unsqueeze(-1)) * \
               torch.exp(- (log_time.unsqueeze(-1) - mean) ** 2 / (2 * covariance))

        log_prob = torch.sum(torch.log(prob + 1e-5), dim=-1) * pred_mask
        return log_prob

    def forward(self, event_type, event_time, covariates):
        batch, seq_len = event_type.shape
        non_pad_mask = get_non_pad_mask(event_type)
        covariates = self.encoder.tensorized_cov(covariates, non_pad_mask)

        event_time -= event_time.clone()[:, 0].unsqueeze(-1)
        event_time *= non_pad_mask.squeeze(-1)

        enc_output = self.encoder(event_type, event_time, non_pad_mask, covariates)

        assert not torch.isnan(enc_output).any()

        w = self.w_network(enc_output)
        covar = self.std_network(enc_output) / self.d_model
        mu = self.mean_network(enc_output) / self.d_model

        next_gap = torch.cat((event_time[:, 1:] - event_time[:, :-1],
                              torch.zeros((batch, 1), device='cuda')), axis=-1)
        next_gap[next_gap < 0] = 0
        pred_mask = next_gap.ne(0).type(torch.float)

        log_prob = self.log_prob_in_mixtureLogNormal(w, covar, mu, next_gap, pred_mask)

        time_prediction = w * torch.exp(mu + covar / 2)
        time_prediction = torch.sum(time_prediction, axis=-1) * pred_mask.squeeze(-1)

        if torch.isnan(time_prediction).any() or torch.isinf(time_prediction).any():
            print('Time prediction error detected.')
            print(time_prediction)
            assert False

        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
            print('Log-prob error detected.')
            print(log_prob)
            assert False

        H2 = self.encoder.cov_intra_att(covariates)
        concat = torch.cat((enc_output, H2), dim=-1)
        temp = self.encoder.cov_his_outer_att(concat)
        type_prediction = self.type_predictor(temp, non_pad_mask)

        return enc_output, (type_prediction, time_prediction, log_prob), concat
