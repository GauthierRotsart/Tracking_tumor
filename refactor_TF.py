# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 13:24:55 2022

@author: grotsartdehe
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math


# classes utilitaires
class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.len = max_len
        # Créer une matrice de taille (max_len, d_model) avec des valeurs de zéro
        pe = torch.zeros(max_len, d_model)

        # Créer un vecteur représentant les positions (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Calculer les angles pour chaque position et chaque dimension
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # Appliquer les formules de sinus et cosinus aux positions pour chaque dimension
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Ajouter une dimension supplémentaire pour l'utiliser avec les batchs (batch_size, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Enregistre `pe` comme Variable pour les anciennes versions de PyTorch
        self.register_buffer('pe', Variable(pe, requires_grad=False))  # permet de gerer le device

    def forward(self, x):
        # Ajouter l'encodage positionnel aux embeddings des tokens en entrée
        x = x + self.pe[:, :x.shape[1], :]
        return x


class LayerNorm(nn.Module):  # couche Add and norm
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FC(nn.Module):  # fully connected layer du MLP
    def __init__(self, in_size, out_size, dropout_r, use_relu):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU(inplace=True)  # inplace permet de gagner de la memoire en "overwrite" l input
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r, use_relu):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dimension, multi_head, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.mha = multi_head
        self.hidden_dimension = hidden_dimension

        self.linear_v = nn.Linear(hidden_dimension, hidden_dimension)
        self.linear_k = nn.Linear(hidden_dimension, hidden_dimension)
        self.linear_q = nn.Linear(hidden_dimension, hidden_dimension)
        self.linear_merge = nn.Linear(hidden_dimension, hidden_dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        d_k = self.hidden_dimension // self.mha

        # 1: Split V, K, Q into B x N_head x Seq length x D / N_head
        v = self.linear_v(v).view(n_batches, -1, self.mha, d_k).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.mha, d_k).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.mha, d_k).transpose(1, 2)

        # 2: Compute the attention
        atted = self.att(v, k, q, mask)

        # 3: Concat each head to form B x Seq length x D
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, d_k * self.mha)

        # 4: Apply a final layer
        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)


class MultiHeadAttention_lora(nn.Module):
    def __init__(self, embed_size, heads, rank, dropout_rate):
        super(MultiHeadAttention_lora, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.rank = rank  # Rang pour LoRA

        # L'originale MultiHeadAttention
        self.attention = MultiHeadAttention(embed_size, heads, dropout_rate)

        # LORA : Ajout de matrices A et B pour chaque matrice de poids avec le préfixe 'lora_'
        self.lora_A_q = nn.Parameter(torch.randn(embed_size, self.rank) * 0.02)  # Paramètre de basse-rang pour Q
        self.lora_B_q = nn.Parameter(torch.randn(self.rank, embed_size))  # Paramètre de basse-rang pour Q

        #self.lora_A_k = nn.Parameter(torch.randn(embed_size, self.rank))  # Paramètre de basse-rang pour K
        #self.lora_B_k = nn.Parameter(torch.randn(self.rank, embed_size))  # Paramètre de basse-rang pour K

        self.lora_A_v = nn.Parameter(torch.randn(embed_size, self.rank) * 0.02)  # Paramètre de basse-rang pour V
        self.lora_B_v = nn.Parameter(torch.randn(self.rank, embed_size))  # Paramètre de basse-rang pour V

        # Initialisation des matrices LoRA
        self.reset_parameters()

    def reset_parameters(self):
        # Initialisation avec Xavier pour les matrices LoRA
        """
        nn.init.xavier_uniform_(self.lora_A_q)
        nn.init.xavier_uniform_(self.lora_B_q)
        #nn.init.xavier_uniform_(self.lora_A_k)
        #nn.init.xavier_uniform_(self.lora_B_k)
        nn.init.xavier_uniform_(self.lora_A_v)
        nn.init.xavier_uniform_(self.lora_B_v)
        """
        nn.init.normal_(self.lora_A_q, mean=0, std=0.02)
        nn.init.zeros_(self.lora_B_q)
        nn.init.normal_(self.lora_A_v, mean=0, std=0.02)
        nn.init.zeros_(self.lora_B_v)
    def forward(self, query, key, value, mask=None):
        # Appliquer LoRA aux poids Q, K, V
        Q = query + torch.matmul(query, self.lora_A_q) @ self.lora_B_q
        K = key #torch.matmul(key, self.lora_A_k) @ self.lora_B_k
        V = value + torch.matmul(value, self.lora_A_v) @ self.lora_B_v

        # On applique les nouvelles valeurs de Q, K, V modifiées par LoRA
        return self.attention(Q, K, V, mask)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, hidden_dimension, ff_dimension, dropout_rate):
        super(FFN, self).__init__()
        self.mlp = MLP(
            in_size=hidden_dimension,
            mid_size=ff_dimension,
            out_size=hidden_dimension,
            dropout_r=dropout_rate,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, hidden_size, ff_size, multi_head, dropout_rate, rang):
        super(Encoder, self).__init__()
        if rang == 0:
            self.mhatt = MultiHeadAttention(hidden_dimension=hidden_size,
                                            multi_head=multi_head,
                                            dropout_rate=dropout_rate)
        else:
            self.mhatt = MultiHeadAttention_lora(embed_size=hidden_size,
                                                 heads=multi_head,
                                                 rank=rang,
                                                 dropout_rate=dropout_rate)

        self.ffn = FFN(hidden_dimension=hidden_size,
                       ff_dimension=ff_size,
                       dropout_rate=dropout_rate)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, x_mask):
        y = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


class Decoder(nn.Module):
    def __init__(self, hidden_size, ff_size, multi_head, dropout_rate, rang):
        super(Decoder, self).__init__()
        if rang == 0:
            self.mhatt1 = MultiHeadAttention(hidden_dimension=hidden_size,
                                             multi_head=multi_head,
                                             dropout_rate=dropout_rate)
            self.mhatt2 = MultiHeadAttention(hidden_dimension=hidden_size,
                                             multi_head=multi_head,
                                             dropout_rate=dropout_rate)
        else:
            self.mhatt1 = MultiHeadAttention_lora(embed_size=hidden_size,
                                                  heads=multi_head,
                                                  rank=rang,
                                                  dropout_rate=dropout_rate)
            self.mhatt2 = MultiHeadAttention_lora(embed_size=hidden_size,
                                                  heads=multi_head,
                                                  rank=rang,
                                                  dropout_rate=dropout_rate)        
        self.ffn = FFN(hidden_dimension=hidden_size,
                       ff_dimension=ff_size,
                       dropout_rate=dropout_rate)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout_rate)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, tgt, tgt_mask, x, x_mask):
        y = self.norm1(tgt + self.dropout1(
            self.mhatt1(tgt, tgt, tgt, tgt_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.mhatt2(x, x, y, x_mask)
        ))

        y = self.norm3(y + self.dropout3(
            self.ffn(y)
        ))

        return y


# -------------------------
# ---- Main Net Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, cfg, rank=0):
        super(Net, self).__init__()
        self.batch_size = cfg.training.params.batch_size
        self.patch_size = cfg.training.params.patch_size
        self.hidden_dimension = cfg.model.params.embedding_size
        self.output_dimension = cfg.training.params.output_dimension
        self.img_height = cfg.training.params.height
        self.img_width = cfg.training.params.width
        self.horizon = cfg.training.params.horizon
        self.device = cfg.device

        self.embed_enc = LinearEmbedding(self.patch_size ** 2, self.hidden_dimension)
        self.embed_dec = LinearEmbedding(self.output_dimension + 1, self.hidden_dimension)

        self.img_sequence = cfg.training.params.img_sequence
        self.patches_sequence = (self.img_height // self.patch_size) * (self.img_width // self.patch_size)
        self.spatial_pos_enc = PositionalEncoding(d_model=self.hidden_dimension,
                                                  max_len=self.patches_sequence)
        self.spatio_temp_enc = PositionalEncoding(d_model=self.hidden_dimension,
                                                  max_len=self.img_sequence * self.patches_sequence)

        self.pos_dec = PositionalEncoding(d_model=cfg.model.params.embedding_size, max_len=self.horizon)

        self.enc = nn.ModuleList([Encoder(hidden_size=self.hidden_dimension,
                                          ff_size=cfg.model.params.neuron,
                                          multi_head=cfg.model.params.mha,
                                          dropout_rate=cfg.model.params.dropout,
                                          rang=rank)
                                  for _ in range(cfg.model.params.layer)])
        self.dec = nn.ModuleList([Decoder(hidden_size=self.hidden_dimension,
                                          ff_size=cfg.model.params.neuron,
                                          multi_head=cfg.model.params.mha,
                                          dropout_rate=cfg.model.params.dropout,
                                          rang=rank)
                                  for _ in range(cfg.model.params.layer)])
        self.final_layer = nn.Linear(self.hidden_dimension, self.output_dimension + 1)

    def forward(self, x, x_mask, tgt, tgt_mask):
        batch_size = x.shape[0]
        # Initially, inputs are B x N_im x N_p x P²
        # 1: Apply embedding of each patch
        x = x.view(batch_size * self.img_sequence, self.patches_sequence, self.patch_size ** 2)
        x = self.embed_enc(x)  # (B * N_im, N_patch, embedding_size)

        # 2: Apply PE spatio-temporal
        x = x.view(batch_size, self.img_sequence * self.patches_sequence, self.hidden_dimension)
        x = self.spatio_temp_enc(x)

        # 3: Apply PE spatio
        x = x.view(batch_size * self.img_sequence, self.patches_sequence, self.hidden_dimension)
        x = self.spatial_pos_enc(x)

        # 4: Feed the sequence to the encoder block
        x = x.view(batch_size, -1, self.hidden_dimension)
        for enc in self.enc:
            y = enc(x, x_mask)
        encoder_output = y

        # 5: Embed and apply PE on decoder input
        tgt = self.embed_dec(tgt)
        tgt = self.pos_dec(tgt)

        # 6: Feed the decoder sequence to the decoder block
        for dec in self.dec:
            y = dec(tgt, tgt_mask, encoder_output, x_mask)

        out = self.final_layer(y).float()  # besoin du float?
        return out


class Net_tracking_CLS(nn.Module):
    def __init__(self, cfg, rank=0):
        super(Net_tracking_CLS, self).__init__()
        self.batch_size = cfg.training.params.batch_size
        self.patch_size = cfg.training.params.patch_size
        self.hidden_dimension = cfg.model.params.embedding_size
        self.output_dimension = cfg.training.params.output_dimension
        self.img_height = cfg.training.params.height
        self.img_width = cfg.training.params.width
        self.device = cfg.device

        # CLS token (learnable parameter)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.model.params.embedding_size))

        self.embed_enc = LinearEmbedding(self.patch_size ** 2, self.hidden_dimension)
        self.patches_sequence = (self.img_height // self.patch_size) * (self.img_width // self.patch_size)
        self.spatial_pos_enc = PositionalEncoding(d_model=self.hidden_dimension,
                                                  max_len=self.patches_sequence + 1)

        self.enc = nn.ModuleList([Encoder(hidden_size=self.hidden_dimension,
                                          ff_size=cfg.model.params.neuron,
                                          multi_head=cfg.model.params.mha,
                                          dropout_rate=cfg.model.params.dropout,
                                          rang=rank)
                                  for _ in range(cfg.model.params.layer)])
        self.final_layer = nn.Linear(self.hidden_dimension, self.output_dimension)

    def forward(self, x, x_mask):
        batch_size = x.shape[0]
        # Initially, inputs are B x N_p x P²
        # 1: Apply embedding of each patch
        x = self.embed_enc(x)  # (B, N_patch, embedding_size)

        # 2: Append the CLS token to the patch sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # B x N_patch + 1 x embedding_size

        # 3: Apply PE on the sequence
        x = self.spatial_pos_enc(x)

        for enc in self.enc:
            y = enc(x, x_mask)
        encoder_output = y
        # Use CLS token output for the final regression (CLS token is at index 0)
        cls_output = encoder_output[:, 0]  # Shape: [B, HIDDEN_SIZE]
        out = self.final_layer(cls_output)  # Final regression layer
        return out


class Net_tracking_noCLS(nn.Module):
    def __init__(self, cfg, rank=0):
        super(Net_tracking_noCLS, self).__init__()
        self.batch_size = cfg.training.params.batch_size
        self.patch_size = cfg.training.params.patch_size
        self.hidden_dimension = cfg.model.params.embedding_size
        self.output_dimension = cfg.training.params.output_dimension
        self.img_height = cfg.training.params.height
        self.img_width = cfg.training.params.width
        self.device = cfg.device

        self.embed_enc = LinearEmbedding(self.patch_size ** 2, self.hidden_dimension)
        self.patches_sequence = (self.img_height // self.patch_size) * (self.img_width // self.patch_size)
        self.spatial_pos_enc = PositionalEncoding(d_model=self.hidden_dimension,
                                                  max_len=self.patches_sequence + 1)

        self.enc = nn.ModuleList([Encoder(hidden_size=self.hidden_dimension,
                                          ff_size=cfg.model.params.neuron,
                                          multi_head=cfg.model.params.mha,
                                          dropout_rate=cfg.model.params.dropout,
                                          rang=rank)
                                  for _ in range(cfg.model.params.layer)])
        self.final_layer = nn.Linear(self.hidden_dimension, self.output_dimension)

    def forward(self, x, x_mask):
        batch_size = x.shape[0]
        # Initially, inputs are B x N_p x P²
        # 1: Apply embedding of each patch
        x = self.embed_enc(x)  # (B, N_patch, embedding_size)

        # 2: Apply PE on the sequence
        x = self.spatial_pos_enc(x)

        for enc in self.enc:
            y = enc(x, x_mask)
        encoder_output = y
        pooled_output, _ = torch.max(encoder_output, dim=1)
        out = self.final_layer(pooled_output)
        # Final regression layer
        return out


class Net_tracking_meanPool(nn.Module):
    def __init__(self, cfg, rank=0):
        super(Net_tracking_meanPool, self).__init__()
        self.batch_size = cfg.training.params.batch_size
        self.patch_size = cfg.training.params.patch_size
        self.hidden_dimension = cfg.model.params.embedding_size
        self.output_dimension = cfg.training.params.output_dimension
        self.img_height = cfg.training.params.height
        self.img_width = cfg.training.params.width
        self.device = cfg.device

        self.embed_enc = LinearEmbedding(self.patch_size ** 2, self.hidden_dimension)
        self.patches_sequence = (self.img_height // self.patch_size) * (self.img_width // self.patch_size)
        self.spatial_pos_enc = PositionalEncoding(d_model=self.hidden_dimension,
                                                  max_len=self.patches_sequence + 1)

        self.enc = nn.ModuleList([Encoder(hidden_size=self.hidden_dimension,
                                          ff_size=cfg.model.params.neuron,
                                          multi_head=cfg.model.params.mha,
                                          dropout_rate=cfg.model.params.dropout,
                                          rang=rank)
                                  for _ in range(cfg.model.params.layer)])
        self.final_layer = nn.Linear(self.hidden_dimension, self.output_dimension)

    def forward(self, x, x_mask):
        batch_size = x.shape[0]
        # Initially, inputs are B x N_p x P²
        # 1: Apply embedding of each patch
        x = self.embed_enc(x)  # (B, N_patch, embedding_size)

        # 2: Apply PE on the sequence
        x = self.spatial_pos_enc(x)

        for enc in self.enc:
            y = enc(x, x_mask)
        encoder_output = y
        pooled_output = torch.mean(encoder_output, dim=1)
        out = self.final_layer(pooled_output)
        # Final regression layer
        return out