import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import path_to_pred_label, discrete_gurobi_tsp_solver
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from utils import path_to_matrix

class CustomBCELoss(torch.nn.Module):
    def __init__(self):
        super(CustomBCELoss, self).__init__()

    def forward(self, pred_adj_matrix, target_adj_matrix, true_path):
        loss = F.binary_cross_entropy_with_logits(pred_adj_matrix, target_adj_matrix, reduction='mean')

        return loss

class NaiveWeightedLoss(torch.nn.Module):
    def __init__(self):
        super(NaiveWeightedLoss, self).__init__()

    def forward(self, pred_adj_matrix, target_adj_matrix, true_path):
        # The target adjacency matrix is a 2D tensor of shape (num_nodes, num_nodes), with each row with only one zero value indicating the next node starting from the current node.
        # The first row is the start node, with zero value at the k-th position indicating the k-th node is the first node to visit.
        # Then we shall go to the k-th row, with zero value at the j-th position indicating the j-th node is the second node to visit.
        # And so on.
        # Here I want to weight the loss of the first node more than the rest of the nodes.
        # The weight of the first node is 1, and the weight of the rest of the nodes is 0.5.
        # The loss is the sum of the negative log likelihood of the predicted adjacency matrix.
        # The negative log likelihood of the predicted adjacency matrix is the sum of the negative log likelihood of the predicted adjacency matrix of each row.
        weight = torch.ones_like(target_adj_matrix)
        weight[0] = 1
        weight[1:] = 0.5

        loss = F.binary_cross_entropy_with_logits(pred_adj_matrix, target_adj_matrix, weight=weight, reduction='sum')

        return loss


class MaxMarginLoss(nn.Module):
    def __init__(self):
        super(MaxMarginLoss, self).__init__()

    def forward(self, adjacency, target_rank, device='cuda'):
        target_path = torch.argsort(target_rank)
        target_matrix = path_to_matrix(target_path, adjacency)

        margin = torch.ones_like(target_matrix) - target_matrix
        adjacency += margin

        pred_path = discrete_gurobi_tsp_solver(adjacency.unsqueeze(0)).unsqueeze(1).to(device)
        pred_matrix = path_to_matrix(pred_path, adjacency)

        target_score = target_matrix * adjacency
        pred_score = pred_matrix * (adjacency)

        loss = torch.sum(pred_score - target_score) / len(target_rank)
        return loss


# # Example usage
# predicted_adj_matrix = torch.rand(10, 10)  # Random adjacency matrix for a 10-city TSP
# target_rank = torch.randperm(10)+1  # Random target rank
# print(target_rank)
#
# loss_fn = MaxMarginLoss()
# loss = loss_fn(predicted_adj_matrix, target_rank)
# print(f"Max Margin Loss: {loss.item()}")