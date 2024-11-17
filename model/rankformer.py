import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
class RankFormer(torch.nn.Module):
    def __init__(self, input_dim,
                 dim_emb=128,
                 tf_dim_feedforward=32,
                 tf_nhead=1,
                 tf_num_layers=2,
                 head_hidden_layers=None,
                 dropout=0.25,
                 list_pred_strength=0.1,
                 output_embedding_mode=False,
                 need_embedding: bool = True
                 ):
        """
        :param input_dim: dimensionality of item features.
        :param dim_emb: dimensionality of the Transformer embeddings.
        :param tf_dim_feedforward: dim_feedforward in the TransformerEncoderLayer.
        :param tf_nhead: nhead in the TransformerEncoderLayer.
        :param tf_num_layers: num_layers in the TransformerEncoder that combines the TransformerEncoderLayers.
        :param head_hidden_layers: Hidden layers in score heads as list of ints. Defaults to [32].
        :param dropout: Used in score heads and TransformerEncoderLayer.
        :param list_pred_strength: Strength of the listwide loss. If 0, no listwide score head is initialized.
        """

        super().__init__()
        # self.list_pred_strength = list_pred_strength
        self.input_dim = input_dim
        self.dim_emb = dim_emb
        self.transformer = None
        self.rank_score_net = None
        self.list_emb = None
        self.list_score_net = None
        self.list_loss_fn = None
        self.list_loss_adjustment_factor = None

        if need_embedding:
            self.embedding = nn.Linear(input_dim, dim_emb)
        else:
            self.embedding = nn.Identity()
            dim_emb = input_dim

        if tf_num_layers > 0:

            encoder_layer = torch.nn.TransformerEncoderLayer(dim_emb, nhead=tf_nhead,
                                                             dim_feedforward=tf_dim_feedforward, dropout=dropout,
                                                             activation='gelu', batch_first=True,
                                                             norm_first=True)
            # Note: the 'norm' parameter is set to 'None' here, because the TransformerEncoderLayer already computes it
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=tf_num_layers, norm=None)

        else:
            self.transformer = None

        # Prepare listwise scoring head
        # if self.list_pred_strength > 0.:
        #     rank_score_input_dim *= 2
        self.rank_score_net = MLP(input_dim=self.dim_emb, hidden_layers=head_hidden_layers, output_dim=1,
                                  dropout=dropout)
        self.rank_loss_fn = OrdinalLoss()
        # self.rank_loss_fn = SoftmaxLoss()
        self.output_embedding_mode = output_embedding_mode


    def forward(self, feat, length):
        """
        :param feat: Tensor of shape (N, input_dim) with N the total number of list elements.
        :param length: Tensor of shape (N,) with the length of each list.
        :return: If list_pred_strength is 0, a Tensor of shape (N,) with the predicted scores for each list element.
        Else, a tuple of: 1) a Tensor of shape (N,) with the predicted scores for each list element and 2) a single
        listwide score for each list.
        """

        # Split up the features per list
        feat_per_list = feat.split(length.tolist())

        if self.transformer is not None:
            # # Stack all lists as separate batch elements in a large tensor and add padding where needed
            feat = pad_sequence(feat_per_list, batch_first=True, padding_value=0)
            feat = self.embedding(feat)
            # Pad the input to the transformer to the maximum feature length
            feat = torch.nn.functional.pad(feat, (0, self.dim_emb - feat.shape[-1]))
            padding_mask = torch.ones((feat.shape[0], feat.shape[1]), dtype=torch.bool).to(feat.device)
            for i, list_len in enumerate(length):
                padding_mask[i, :list_len] = False

            tf_embs = self.transformer(feat, src_key_padding_mask=padding_mask)
            tf_list_emb = None

            # Only keep the non-padded list elements and concatenate all embedded list features again
            tf_embs = tf_embs[~padding_mask]
        else:
            tf_embs = feat

        if self.output_embedding_mode:
            return tf_embs

        rank_score = self.rank_score_net(tf_embs)

        return rank_score

    def compute_loss(self, score, target, length):
        """
        :param score: See output of forward().
        :param target: Tensor of shape (N,) with the target labels for each list element.
        :param length: Tensor of shape (N,) with the length of each list.
        :return: If list_pred_strength is 0, a 0-dimensional Tensor with the ranking loss. Else, a tuple of the ranking
        loss and the listwide loss.
        """

        if isinstance(score, tuple):
            rank_score, list_score = score
        else:
            rank_score = score
            list_score = None

        rank_loss = self.rank_loss_fn.forward_per_list(rank_score, target, length)

        return rank_loss

    def get_name(self):
        return "RankFormer"


class MLP(torch.nn.Module):
    def __init__(self, input_dim,
                 hidden_layers=None,
                 output_dim=1,
                 dropout=0.):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [32]

        net = []
        for h_dim in hidden_layers:
            net.append(torch.nn.Linear(input_dim, h_dim))
            net.append(torch.nn.ReLU())
            if dropout > 0.:
                net.append(torch.nn.Dropout(dropout))
            input_dim = h_dim
        net.append(torch.nn.Linear(input_dim, output_dim))

        self.net = torch.nn.Sequential(*net)

        self.rank_loss_fn = OrdinalLoss()
        # self.rank_loss_fn = OrdinalCrossEntropyLoss()

    def forward(self, feat, *_args):
        score = self.net(feat).squeeze(dim=-1)
        return score

    def compute_loss(self, score, target, length):
        loss = self.rank_loss_fn.forward_per_list(score, target, length)
        return loss

import torch
import torch.nn.functional as F

class BaseRankLoss(torch.nn.Module):
    def forward(self, score, target):
        raise NotImplementedError

    def forward_per_list(self, score, target, length):
        # Split score and target into lists
        length_per_list = length.tolist()
        score_per_list = score.split(length_per_list)
        target_per_list = target.split(length_per_list)

        # Compute loss per list, giving each list equal weight (regardless of length)
        loss_per_list = [
            self(score_of_list, target_of_list)
            for score_of_list, target_of_list in zip(score_per_list, target_per_list)
        ]
        losses = torch.stack(loss_per_list)

        # Remove losses that are zero (e.g. all item labels are zero)
        losses = losses[torch.abs(losses) > 0.]
        if len(losses) == 0:
            # If all losses were removed, take the sum (which will result in a zero gradient)
            return losses.sum()

        loss = losses.mean()
        return loss


class OrdinalLoss(BaseRankLoss):
    # See A Neural Network Approach to Ordinal Regression
    def forward(self, score, target, higher_is_better=False):
        """
        :param score: Tensor of shape [batch_size] - model's score predictions
        :param target: Tensor of shape [batch_size] - true ordinal labels
        :param higher_is_better: Boolean - if True, a higher ordinal label denotes a higher rank (e.g., relevance).
                                          If False, a lower ordinal label denotes a higher rank (e.g., racing).
        :return: scalar - ordinal ranking loss
        """
        min_rank = target.min().item()
        num_classes = int(target.max().item() - target.min().item())
        # Expand the score and target tensors
        score = score.unsqueeze(1).repeat(1, num_classes)
        target = target.unsqueeze(1).repeat(1, num_classes)

        # Create ordinal binary labels based on the ranking direction

        if higher_is_better:
            binary_labels = torch.arange(int(min_rank)+1, num_classes + int(min_rank)+1,
                                         device=score.device, dtype=target.dtype).unsqueeze(0)
            binary_labels = (binary_labels <= target).float()
        else:
            binary_labels = torch.arange(int(min_rank), num_classes+int(min_rank), device=score.device, dtype=target.dtype).unsqueeze(0)
            binary_labels = (binary_labels >= target).float()

        # Compute binary logistic loss
        loss = F.binary_cross_entropy_with_logits(score, binary_labels, reduction='none')

        return loss.sum(dim=1).mean()



class SoftmaxLoss(BaseRankLoss):
    def forward(self, score, target):
        softmax_score = torch.nn.functional.log_softmax(score, dim=-1)
        loss = -(softmax_score * target).mean()
        return loss
