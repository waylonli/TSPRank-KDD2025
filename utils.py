import networkx as nx
from networkx.algorithms import approximation as approx
import torch
import os
import copy
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from gurobipy import *
import gurobipy as gp
# load from .env file
from dotenv import load_dotenv
load_dotenv()

try:
    params = {
    "WLSACCESSID": os.getenv("WLSACCESSID"),
    "WLSSECRET": os.getenv("WLSSECRET"),
    "LICENSEID": os.getenv("LICENSEID"),
    }
    gp_env = gp.Env(params=params)
except:
    gp_env = gp.Env()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_mrr(pred_rank, target_rank, reverse=False):
    assert len(pred_rank) == len(target_rank)
    assert min(pred_rank) == min(target_rank)
    # assert max(pred_rank) == max(target_rank)
    adjustment = 0 if min(pred_rank) == 1 else 1
    if reverse:
        # reverse the ranking, make the last item the first
        pred_rank = [len(pred_rank) - rank + 1 for rank in pred_rank]
        target_rank = [len(target_rank) - rank + 1 for rank in target_rank]
    top_relevant_idx = np.argmin(target_rank)
    rank = list(pred_rank)[top_relevant_idx] + adjustment

    return float(1 / rank)


def discrete_greedy_tsp_solver(adjacency: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    G = nx.DiGraph()
    G.add_nodes_from(range(length[0].item() + 1))
    for i in range(length[0].item() + 1):
        for j in range(length[0].item() + 1):
            if i != j:
                G.add_edge(i, j, weight=adjacency[0][i][j].item())
    return approx.greedy_tsp(G)[1:-1]


def discrete_gurobi_tsp_solver(adjacency: torch.Tensor, length: torch.Tensor = None) -> torch.Tensor:
    n = adjacency.shape[1]
    V = list(range(0, n))
    A = [(i, j) for i in V for j in V if i != j]
    C = {(i, j): float(adjacency[0][i][j]) for i, j in A}

    mdl = Model('TSP', env=gp_env)
    mdl.setParam('OutputFlag', 0)
    x = mdl.addVars(A, vtype=GRB.BINARY)
    y = mdl.addVars(A, vtype=GRB.INTEGER)
    mdl.setObjective(quicksum(x[i, j] * C[i, j] for i, j in A), sense=GRB.MAXIMIZE)
    # mdl.setObjective(quicksum(x[i, j] * C[i, j] for i, j in A), sense=GRB.MINIMIZE)
    # set time limit
    # mdl.Params.timelimit = 60
    mdl.Params.lazyConstraints = 1
    mdl.update()

    # Degree constraints ======================================================
    for i in range(n):
        mdl.addConstr(quicksum(x[i, j] for j in range(n) if i != j) <= 1, name='leave_%s' % str(i))
        mdl.addConstr(quicksum(x[j, i] for j in range(n) if i != j) <= 1, name='enter_%s' % str(i))

        # mdl.addConstr(quicksum(x[i, j] for j in range(n) if i != j) == 1, name='leave_%s' % str(i))
        # mdl.addConstr(quicksum(x[j, i] for j in range(n) if i != j) == 1, name='enter_%s' % str(i))

    mdl.addConstr(quicksum(x[i, j] for j in range(n) for i in range(n) if i != j) == n - 1)
    # mdl.addConstr(quicksum(x[i, j] for j in range(n) for i in range(n) if i != j) == n)

    mdl._x = x

    def subtourelim(model, where):
        if (where == GRB.Callback.MIPSOL):
            x_sol = model.cbGetSolution(model._x)
            G = nx.Graph()
            for (i, j) in x.keys():
                if (x_sol[i, j] > 0.9):
                    G.add_edge(i, j, weight=C[i, j])
            components = [list(c) for c in nx.connected_components(G)]
            for component in components:
                if (len(component) < n):
                    model.cbLazy(
                        quicksum(x[i, j] for i in component for j in component if i != j) <= len(component) - 1)

    mdl.optimize(subtourelim)
    # mdl.optimize()

    active_arts = [a for a in A if x[a].x > 0.5]

    def sort_edges_to_path(edges):
        # Construct a mapping from each node to the next
        edge_map = {start: end for start, end in edges}
        start_set = set([start for start, end in edges])
        end_set = set([end for start, end in edges])
        # see which node is not in the end list
        try:
            start = (start_set - end_set).pop()
        except KeyError:
            # print(start_set, end_set)
            # print(len(edges))
            # print(n)
            # print(edges)
            # print(adjacency)
            # for edge in edges:
            #     print(edge, x[edge].x)
            # raise AssertionError
            print("No start node found")
            print(adjacency)
            start = 0
        # Determine starting point (assuming 1 or 0 as possible starts)
        # start = 0 if 0 in edge_map else 1

        # Reconstruct the path
        path = [start]
        while True:
            next_node = edge_map.get(path[-1])
            if next_node is None or next_node == start or len(path) >= n:
                break
            path.append(next_node)

        # Convert to PyTorch tensor
        # assert len(path) > 2, "Path must have at least 3 nodes"
        # print("pred_path:", path)
        # print("len:", len(path))
        path_tensor = torch.tensor(path)
        # path_tensor = torch.tensor(path[1:])
        return path_tensor

    return sort_edges_to_path(active_arts)


def path_to_pred_label(predicted_paths: torch.Tensor) -> torch.Tensor:
    """
    Transform a batch of predicted paths into pred_labels using PyTorch.

    Parameters:
    - predicted_paths: A 1D PyTorch tensor of shape [num_nodes] representing item rankings.

    Returns:
    - A 1D PyTorch tensor where each element is the pred_label for the corresponding predicted path.
    """
    num_nodes = len(predicted_paths)
    adjustment = 1 if predicted_paths.min().item() == 1 else 0
    pred_labels = torch.zeros(num_nodes, dtype=torch.long)
    for rank, item in enumerate(predicted_paths):
        pred_labels[item - adjustment] = rank + 1  # Adjust for zero-based indexing

    return pred_labels

def path_to_matrix(path, adj):
    if len(adj.shape) == 3:
        adj = adj.squeeze()
    # detect the minimum value in the path
    adjustment = 1 if path.min().item() == 1 else 0
    output_matrix = torch.zeros_like(adj)
    for i in range(len(path)-1):
        output_matrix[path[i]-adjustment, path[i+1]-adjustment] = 1
    return output_matrix



def binary_consecutive_target(target, adjacency_without_pad):
    gold_consecutive_adj = torch.zeros_like(adjacency_without_pad)
    for i in range(1, len(target)):
        current_idx = torch.where(target == i)[0]
        next_idx = torch.where(target == i + 1)[0]
        gold_consecutive_adj[current_idx, next_idx] = 1

    return gold_consecutive_adj


def binary_pairwise_target(target, adjacency_without_pad=None, maximize=True):
    if adjacency_without_pad is not None:
        length_without_padding = adjacency_without_pad.shape[0]
    else:
        length_without_padding = len(target)
    target = target[:length_without_padding]
    # gold_pairwise_adj = torch.ones_like(adjacency_without_pad)
    rank_matrix = target.unsqueeze(0).repeat(len(target), 1)
    # Create the adjacency matrix: 1 if horse i ranks before horse j, 0 otherwise
    if maximize:
        gold_pairwise_adj = (rank_matrix > rank_matrix.t()).int()
    else:
        gold_pairwise_adj = (rank_matrix <= rank_matrix.t()).int()

    gold_pairwise_adj = gold_pairwise_adj.float()

    # # Generate random values for ones and zeros
    # random_for_ones = torch.rand(gold_pairwise_adj.shape).to("cuda") * 0.3 + 0.7  # random values between 0.7 and 1
    # random_for_zeros = torch.rand(gold_pairwise_adj.shape).to("cuda") * 0.4  # random values between 0 and 0.4
    #
    # # Apply random values based on the adjacency matrix
    # gold_pairwise_adj = torch.where(gold_pairwise_adj == 1, random_for_ones, random_for_zeros)

    return gold_pairwise_adj


def rank_diff_target(target, adjacency_without_pad, division=8.0):
    length_without_padding = adjacency_without_pad.shape[0]
    target = target[:length_without_padding]

    n = target.shape[0]
    # Expand ranks to compute pairwise differences in a matrix form
    target_expanded = target.unsqueeze(1).expand(n, n)
    rank_differences = target_expanded - target_expanded.t()

    def custom_sigmoid(x, division):
        sign_shift = torch.sign(x)
        return 1 / (1 + torch.exp(-(sign_shift + (x / division))))

    gold_rank_diff_adj = rank_differences.fill_diagonal_(torch.max(target))
    # # print(gold_rank_diff_adj)
    # scaled_rank_diff = torch.sigmoid(gold_rank_diff_adj/division)
    scaled_rank_diff = custom_sigmoid(gold_rank_diff_adj, division)
    # print(scaled_rank_diff)
    # print(scaled_rank_diff)


    # pred_label_from_target = path_to_pred_label(discrete_gurobi_tsp_solver(scaled_rank_diff.unsqueeze(0))).to(device)
    # if not pred_label_from_target.equal(target.to(device)):
    #     print("Mismatch!")

    return scaled_rank_diff



def load_EOD_data(data_path, market_name, tickers, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            # remove the last day since lots of missing data
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            # print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 1], dtype=np.float32)
            masks = np.ones([len(tickers), single_EOD.shape[0]],
                            dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32)
            base_price = np.zeros([len(tickers), single_EOD.shape[0]],
                                  dtype=np.float32)
        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                masks[index][row] = 0.0
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) \
                    > 1e-8:
                ground_truth[index][row] = \
                    (single_EOD[row][-1] - single_EOD[row - steps][-1]) / \
                    single_EOD[row - steps][-1]
            for col in range(single_EOD.shape[1]):
                if abs(single_EOD[row][col] + 1234) < 1e-8:
                    single_EOD[row][col] = 1.1
        eod_data[index, :, :] = single_EOD[:, 1:]
        base_price[index, :] = single_EOD[:, -1]
    return eod_data, masks, ground_truth, base_price


def load_graph_relation_data(relation_file, lap=False):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    ajacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float),
                       np.ones(rel_shape, dtype=float))
    degree = np.sum(ajacent, axis=0)
    for i in range(len(degree)):
        degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    deg_neg_half_power = np.diag(degree)
    if lap:
        return np.identity(ajacent.shape[0], dtype=float) - np.dot(
            np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)
    else:
        return np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)


def load_relation_data(relation_file):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
    return relation_encoding, mask


def build_SFM_data(data_path, market_name, tickers):
    eod_data = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0]],
                                dtype=np.float32)

        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                # handle missing data
                if row < 3:
                    # eod_data[index, row] = 0.0
                    for i in range(row + 1, single_EOD.shape[0]):
                        if abs(single_EOD[i][-1] + 1234) > 1e-8:
                            eod_data[index][row] = single_EOD[i][-1]
                            # print(index, row, i, eod_data[index][row])
                            break
                else:
                    eod_data[index][row] = np.sum(
                        eod_data[index, row - 3:row]) / 3
                    # print(index, row, eod_data[index][row])
            else:
                eod_data[index][row] = single_EOD[row][-1]
        # print('test point')
    np.save(market_name + '_sfm_data', eod_data)


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """

    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


if __name__ == "__main__":
    # target = torch.tensor([ 8, 28, 13, 21, 19, 11, 12, 26,  6,  9, 27,  7,  5, 24, 23, 30, 14, 22,
    #      1, 10,  4,  2, 15, 16, 29, 18,  3, 20, 17, 25], dtype=torch.long).to(device)
    # target_float = target.float()
    # adj = torch.rand((30, 30)).to(device)
    # # target_adj = rank_diff_target(target, pred_adj)
    # for i in range(1, 100):
    #     target_adj = rank_diff_target(target, adj, i).cpu()
    #     # print(target_adj)
    #     pred_path = discrete_gurobi_tsp_solver(target_adj.unsqueeze(0)).cpu()
    # # print(len(pred_path))
    #
    #     pred_rank = path_to_pred_label(pred_path)
    #     # see if the pred_path is the same as target
    #     if pred_rank.equal(target.cpu()):
    #         # print(target_adj)
    #         print("Correct i:", i)
    #
    #         continue
    #     else:
    #         # rmse
    #         print(f"{i} RMSE", torch.sqrt(torch.mean((target_float.cpu() - pred_rank)**2)))
    # print("pred_path:", pred_path)
    # print("target_labels:", path_to_pred_label(pred_path))
    # print("true_labels:", target)
    print(mapk([[1, 2, 3]], [[1, 2, 4]]))