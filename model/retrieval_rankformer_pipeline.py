from model.rankformer import RankFormer
import torch
from sklearn.metrics import root_mean_squared_error, accuracy_score
import wandb
from tqdm import tqdm
import os
from sklearn.metrics import ndcg_score
import yaml
from scipy.stats import kendalltau
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RetrievalRankformerPipeline():
    def __init__(self, input_dim: int, nb_layers: int = 4, dim_emb: int = 10, nb_heads: int = 2, dim_ff: int = 32):
        self.rankformer = RankFormer(input_dim=input_dim, dim_emb=dim_emb, tf_dim_feedforward=dim_ff, tf_nhead=nb_heads, tf_num_layers=nb_layers)
        self.rankformer.to(device)
        self.config = {
            'input_dim': input_dim,
            'nb_layers': nb_layers,
            'dim_emb': dim_emb,
            'nb_heads': nb_heads,
            'dim_ff': dim_ff
        }

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            logger: wandb.run = None,
            epochs: int = 200,
            optimizer: str = "adam",
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            eval_freq: int = 10,
            eval_metric: str = "rmse",
            checkpoint_path: str = "checkpoints/marginal_pipeline",
            eval_first: bool = True,):

        self.rankformer.train()

        if logger is not None:
            logger.watch(self.rankformer, log="all", log_freq=10)

        if optimizer == "adam":
            optimizer = torch.optim.Adam(self.rankformer.parameters(), lr=1e-3, weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(self.rankformer.parameters(), lr=0.0001)

        losses = []
        loop = tqdm(range(epochs), total=epochs, leave=True)

        higher_better = False if eval_metric.lower() in ["rmse"] else True
        best_metric = 0 if higher_better else float('inf')

        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        for epoch in loop:
            self.rankformer.train()
            epoch_loss = 0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                feat, length, target = batch['feat'].to(device), batch['length'].to(device), batch['target'].to(device)
                # print(length)
                scores = self.rankformer(feat, length)
                loss = self.rankformer.compute_loss(scores, target, length)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            logger.log({"train_loss": epoch_loss})
            loop.set_description('Loss: {:.4f} | Change: {:.4f}'.format(epoch_loss,
                                                                        losses[-1] - losses[-2] if len(
                                                                            losses) > 1 else 0))

            if (epoch+1) % eval_freq == 0:
                complete_valid_metric = self.evaluate(valid_loader, verbose=False)[0]
                valid_metric = complete_valid_metric[eval_metric.lower()]
                if (higher_better and valid_metric > best_metric) or (not higher_better and valid_metric < best_metric):
                    best_metric = valid_metric

                    complete_test_metric = self.evaluate(test_loader, verbose=False)[0]

                    print(f"Saving the model at epoch {epoch+1}")
                    print("Valid:", complete_valid_metric)
                    print("Test:", complete_test_metric)
                    self.save_checkpoint(checkpoint_path)

                logger.log({f"valid_{eval_metric}": valid_metric})

        return

    def evaluate(self, test_loader: torch.utils.data.DataLoader, verbose: bool = True):
        self.rankformer.eval()
        pred_rankings = []
        gold_rankings = []
        reverse_pred_rankings = []
        reverse_gold_rankings = []
        mrr = 0
        kendall_tau = 0

        loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True) if verbose else enumerate(test_loader)

        for i, batch in loop:
            feat, length, target = batch['feat'].to(device), batch['length'].to(device), batch['target'].to(device)
            scores = self.rankformer(feat, length)
            target = target.split(length.tolist())
            scores = scores.split(length.tolist())
            for i in range(len(target)):
                # obtain the original ranking, the position should be the same as the score, argsort is giving the path, not the ranking
                sorted_idx = torch.argsort(scores[i], descending=True)
                # turn the path into ranking
                ori_pred_rank = torch.zeros_like(sorted_idx)
                ori_pred_rank[sorted_idx] = torch.arange(len(sorted_idx)).to(device)
                rank_gap = target[i].min() - ori_pred_rank.min()
                adjusted_pred_rank = ori_pred_rank + rank_gap
                new_pred_rankings = adjusted_pred_rank.cpu().numpy().astype(int).tolist()
                new_gold_rankings = target[i].cpu().numpy().astype(int).tolist()

                reverse_pred_rank = len(new_pred_rankings) - np.array(new_pred_rankings) + 1
                reverse_gold_rank = len(new_gold_rankings) - np.array(new_gold_rankings) + 1

                pred_rankings += new_pred_rankings
                gold_rankings += new_gold_rankings

                reverse_gold_rankings.append(reverse_gold_rank.tolist())
                reverse_pred_rankings.append(reverse_pred_rank.tolist())

                mrr += get_mrr(adjusted_pred_rank.cpu().numpy().astype(int).tolist(), target[i].cpu().numpy().astype(int).tolist())
                kendall_tau += kendalltau(adjusted_pred_rank.cpu().numpy().astype(int).tolist(), target[i].cpu().numpy().astype(int).tolist())[0]

        rmse = round(root_mean_squared_error(gold_rankings, pred_rankings), 4)
        acc = round(accuracy_score(gold_rankings, pred_rankings), 4)
        mrr = round(mrr / len(test_loader), 4)
        kendall_tau = round(kendall_tau / len(test_loader), 4)
        ndcg_3 = round(ndcg_score(reverse_gold_rankings, reverse_pred_rankings, k=3), 4)
        ndcg_5 = round(ndcg_score(reverse_gold_rankings, reverse_pred_rankings, k=5), 4)
        ndcg_10 = round(ndcg_score(reverse_gold_rankings, reverse_pred_rankings, k=10), 4)

        metrics = {"rmse": rmse, "acc": acc, "mrr": mrr, "kendall_tau": kendall_tau, "ndcg_3": ndcg_3, "ndcg_5": ndcg_5, "ndcg_10": ndcg_10}

        return metrics, pred_rankings, gold_rankings

    def save_checkpoint(self, path: str):
        path = os.path.join(path, "model.pth") if ".pth" not in path else path
        # make directory if it does not exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # export the config to a file
        with open(os.path.join(os.path.dirname(path), "config.yaml"), "w") as f:
            yaml.dump(self.config, f)
        # save the model
        torch.save(self.rankformer.state_dict(), path)

    def load_checkpoint(self, path):
        path = os.path.join(path, "model.pth") if os.path.isdir(path) else path
        self.rankformer.load_state_dict(torch.load(path))
