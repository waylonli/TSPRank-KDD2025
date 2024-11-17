import yaml


import torch
import wandb
from tqdm import tqdm
from utils import *
import os
from sklearn.metrics import root_mean_squared_error, accuracy_score
import networkx as nx
from networkx.algorithms import approximation as approx
from scipy.stats import kendalltau
from torch.nn.utils.rnn import pad_sequence

class BasePipeline():
    def __init__(self, input_dim: int, nb_layers: int = 4, dim_emb: int = 10, nb_heads: int = 2, dim_ff: int = 32, need_embedding: bool = False, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.config = {"input_dim": input_dim, "nb_layers": nb_layers, "dim_emb": dim_emb, "nb_heads": nb_heads, "dim_ff": dim_ff, "need_embedding": need_embedding}
        self.graph_encoder = None
        self.train_tsp_solver = None
        self.test_tsp_solver = None
        self.device = device


    def evaluate(self, test_loader: torch.utils.data.DataLoader, verbose: bool = True):
        self.graph_encoder.eval()
        pred_rankings = []
        gold_rankings = []
        reverse_pred_rankings = []
        reverse_gold_rankings = []
        mrr = 0
        kendall_tau = 0
        loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True) if verbose else enumerate(test_loader)
        valid_num = 0
        with torch.no_grad():
            for idx, batch in loop:
                feat, length, target = batch['feat'].to(self.device), batch['length'].to(self.device), batch['target'].to(self.device)
                adjacency = self.graph_encoder(feat, length)[0].unsqueeze(0)

                # if idx == 0:
                #     print("adjacency:", adjacency)

                try:
                    pred_path = torch.tensor(self.test_tsp_solver(adjacency, length)).to(self.device)
                except gurobipy.GurobiError:
                    continue
                try:
                    pred_rank = path_to_pred_label(pred_path)
                except:
                    print("Error in path_to_pred_label")
                new_kendall_tau = kendalltau(pred_rank, target.cpu().numpy().astype(int).tolist())[0]
                # check if the kendall tau is nan
                if np.isnan(new_kendall_tau):
                    continue
                new_mrr = get_mrr(pred_rank, target.cpu().numpy().astype(int).tolist())
                new_pred_rankings = [int(item) for item in list(pred_rank)]
                new_gold_rankings = target.cpu().numpy().astype(int).tolist()

                reverse_pred_rank = len(new_pred_rankings) - np.array(new_pred_rankings) + 1
                reverse_gold_rank = len(new_gold_rankings) - np.array(new_gold_rankings) + 1

                reverse_gold_rankings.append(reverse_gold_rank.tolist())
                reverse_pred_rankings.append(reverse_pred_rank.tolist())

                kendall_tau += new_kendall_tau
                mrr += new_mrr
                pred_rankings += new_pred_rankings
                gold_rankings += new_gold_rankings

                valid_num += 1


        rmse = round(root_mean_squared_error(gold_rankings, pred_rankings), 4)
        acc = round(accuracy_score(gold_rankings, pred_rankings), 4)
        mrr = round(mrr / valid_num, 4)
        kendall_tau = round(kendall_tau / valid_num, 4)

        metrics = {"rmse": rmse, "acc": acc, "mrr": mrr, "kendall_tau": kendall_tau}

        # metrics = {"rmse": rmse, "acc": acc, "mrr": mrr, "kendall_tau": kendall_tau}
        return metrics, pred_rankings, gold_rankings

    def save_checkpoint(self, path: str):
        path = os.path.join(path, "model.pth") if ".pth" not in path else path
        # make directory if it does not exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # export the config to a file
        with open(os.path.join(os.path.dirname(path), "config.yaml"), "w") as f:
            yaml.dump(self.config, f)
        # save the model
        torch.save(self.graph_encoder.state_dict(), path)

    def load_checkpoint(self, path):
        path = os.path.join(path, "model.pth") if os.path.isdir(path) else path
        self.graph_encoder.load_state_dict(torch.load(path))
