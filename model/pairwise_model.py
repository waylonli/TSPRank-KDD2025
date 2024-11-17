import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

class Comparisor(nn.Module):
    def __init__(self,
                 d_model, deep):
        super(Comparisor, self).__init__()

        self.out_bi = d_model if deep else 1
        self.bi = nn.Bilinear(d_model, d_model, self.out_bi)
        # self.proj = nn.Sigmoid()

        if deep:
            self.proj = nn.Sequential(nn.Tanh(), nn.Linear(d_model, 1))
        else:
            self.proj = None

    def forward(self, svecs, pvecs):
        v = self.bi(svecs, pvecs)
        if self.out_bi == 1:
            return v
        return self.proj(v)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int,
                 nb_layers: int = 4,
                 dim_emb: int = 128,
                 nb_heads: int = 8,
                 dim_ff: int = 32,
                 dropout: float = 0.15,):
        super(TransformerEncoder, self).__init__()

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_emb, nb_heads, dim_ff, dropout=dropout, batch_first=True),
            nb_layers)

    def forward(self, x: torch.Tensor, length: torch.Tensor=None):
        if length is None:
            return self.transformer_encoder(x.unsqueeze(0))

        feat_per_list = x.split(length.tolist())
        h = pad_sequence(feat_per_list, batch_first=True, padding_value=0)
        padding_mask = torch.ones((h.shape[0], h.shape[1]), dtype=torch.bool).to(h.device)
        for i, list_len in enumerate(length):
            padding_mask[i, :list_len] = False

        tf_embs = self.transformer_encoder(h, src_key_padding_mask=padding_mask)
        tf_embs = tf_embs[~padding_mask]

        return tf_embs


class PairwiseModel(nn.Module):
    def __init__(self, input_dim: int,
                 nb_layers: int = 4,
                 dim_emb: int = 128,
                 nb_heads: int = 8,
                 dim_ff: int = 32,
                 deep: bool = False,
                 separate: bool = True,
                 dropout: float = 0.15,
                 need_embedding: bool = False):

        super(PairwiseModel, self).__init__()

        self.nb_layers = nb_layers

        if need_embedding:
            self.embedding = nn.Sequential(
                nn.Linear(input_dim, dim_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, dim_emb)
            ) if nb_layers == 0 else nn.Linear(input_dim, dim_emb)
        else:
            self.embedding = nn.Identity()
            dim_emb = input_dim

        if nb_layers == 0:
            self.transformer = nn.Identity()
        else:
            self.transformer = TransformerEncoder(input_dim=input_dim,
                                                  nb_layers=nb_layers,
                                                  dim_emb=dim_emb,
                                                  nb_heads=nb_heads,
                                                  dim_ff=dim_ff,
                                                  dropout=dropout)
        self.separate = separate

        if separate:
            self.upper_comparisor = Comparisor(dim_emb, deep)
            self.lower_comparisor = Comparisor(dim_emb, deep)
        else:
            self.comparisor = Comparisor(dim_emb, deep)


        # initialize weights
        for p in self.parameters():
            p.requires_grad = True
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feats: torch.Tensor, length: torch.tensor = None):
        feats = self.embedding(feats)
        if self.nb_layers == 0:
            encoded_feats = feats
        else:
            encoded_feats = self.transformer(feats, length)

        if length is not None:
            encoded_feats = encoded_feats.split(length.tolist())

        adjcencies = []
        for i in range(len(encoded_feats)):
            batch_feats = encoded_feats[i]
            batch_size = batch_feats.shape[0]
            repeated_feats_1 = batch_feats.repeat(1, batch_size).view(batch_size**2, -1)
            repeated_feats_2 = batch_feats.repeat(batch_size, 1)

            if not self.separate:
                adjacency = self.comparisor(repeated_feats_1, repeated_feats_2).view(batch_size, batch_size)
                adjcencies.append(adjacency)
            else:
                upper_adjacency = self.upper_comparisor(repeated_feats_1, repeated_feats_2).view(batch_size, batch_size)
                lower_adjacency = self.lower_comparisor(repeated_feats_1, repeated_feats_2).view(batch_size, batch_size)
                # merge upper and lower triangular matrices
                adjacency = torch.triu(upper_adjacency) + torch.tril(lower_adjacency, -1)
                adjcencies.append(adjacency)

        return adjcencies


if __name__ == '__main__':
    model = PairwiseModel(2, nb_heads=1, need_embedding=False)
    feats = torch.randn(4, 2)
    print(model(feats, torch.tensor([4]))[0])
