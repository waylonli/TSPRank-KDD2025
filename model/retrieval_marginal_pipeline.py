import fsspec
import torch.nn
import wandb
from tqdm import tqdm
from utils import *
import os
from torch.nn.utils.rnn import pad_sequence
from model.pairwise_model import PairwiseModel
from model.base_pipeline import BasePipeline
from model.loss import *
from torch.nn import BCELoss
import scipy.stats as sps
import random
from transformers import optimization
from sklearn.metrics import root_mean_squared_error, accuracy_score, ndcg_score
from scipy.stats import kendalltau

class RetrievalMarginalPipeline(BasePipeline):
    def __init__(self, input_dim: int,
                 nb_layers: int = 4,
                 dim_emb: int = 10,
                 nb_heads: int = 2,
                 dim_ff: int = 32,
                 train_mode: str = "hybrid",
                 need_embedding: bool = False,
                 tsp_solver: str = "gurobi",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(RetrievalMarginalPipeline, self).__init__(input_dim=input_dim, nb_layers=nb_layers, dim_emb=dim_emb, nb_heads=nb_heads, dim_ff=dim_ff, need_embedding=need_embedding, device=device)
        self.graph_encoder = PairwiseModel(input_dim=input_dim, nb_layers=nb_layers, dim_emb=dim_emb, nb_heads=nb_heads, dim_ff=dim_ff, deep=False, need_embedding=need_embedding, separate=True).to(self.device)
        self.dim_emb = dim_emb if need_embedding else input_dim
        self.train_mode = train_mode
        if tsp_solver == "greedy":
            self.test_tsp_solver = discrete_greedy_tsp_solver
        elif tsp_solver == "gurobi":
            self.test_tsp_solver = discrete_gurobi_tsp_solver
        else:
            raise NotImplementedError

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            logger: wandb.run,
            epochs: int = 200,
            optimizer: str = "adam",
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            eval_freq: int = 10,
            eval_metric: str = "rmse",
            checkpoint_path: str = "checkpoints/marginal_pipeline",
            eval_first: bool = True,):

        # update config
        self.config.update({"train_mode": self.train_mode})

        logger.watch(self.graph_encoder, log="all", log_freq=10)

        loss_fn = MaxMarginLoss()

        if optimizer == "adam":
            optimizer = torch.optim.Adam(self.graph_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(self.graph_encoder.parameters(), lr=learning_rate)

        warmup_steps = int(epochs * len(train_loader) * 0.3)
        # print(f"Using warmup steps: {warmup_steps} out of {epochs * len(train_loader)} steps")

        # scheduler = optimization.get_cosine_schedule_with_warmup(optimizer,
        #                                                         num_warmup_steps=warmup_steps,
        #                                                         num_training_steps=epochs*len(train_loader),
        #                                                         num_cycles=1)

        # scheduler = optimization.get_polynomial_decay_schedule_with_warmup(optimizer,
        #                                                         num_warmup_steps=warmup_steps,
        #                                                         num_training_steps=epochs*len(train_loader),
        #                                                         lr_end=learning_rate/100,
        #                                                         power=3)

        # scheduler = NOAMSchedule(d_model=self.dim_emb, warmup_steps=warmup_steps, optimizer=optimizer)

        losses = []
        loop = tqdm(range(epochs), total=epochs, leave=True)

        higher_better = False if eval_metric.lower() in ["rmse"] else True

        if eval_first:
            initial_metrics, pred_rankings, target_rankings = self.evaluate(valid_loader, verbose=False)
            best_metric = initial_metrics[eval_metric.lower()]
            best_kendall_tau = initial_metrics['kendall_tau']
            best_ndcg_10 = initial_metrics['ndcg_10']
            print(f"Initial performance on the test set: {initial_metrics}")
        else:
            best_metric = -float('inf') if higher_better else float('inf')
            best_kendall_tau = -float('inf')
            best_ndcg_10 = -float('inf')

        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        train_metric = -1
        test_metric = -1
        print("Training Mode:", self.train_mode)

        if self.train_mode == "local":
            do_glob = False
        elif self.train_mode == "global":
            do_glob = True

        for epoch in loop:
            steps_num = 0
            self.graph_encoder.train()
            epoch_loss = 0
            idx = epoch % 2
            for batch in train_loader:
                if self.train_mode == "hybrid":
                    do_glob = idx % 2
                optimizer.zero_grad()
                feat, length, target = batch['feat'].to(self.device), batch['length'].to(self.device), batch['target'].to(self.device)

                adjacency = self.graph_encoder(feat, length)
                # if steps_num == 0:
                #     print(adjacency[0].shape)

                # pred_tour = differentiable_tsp_solver(adjacency)
                # use argmin to get the gold tour from target, where target is the finishing position
                target = target.split(length.tolist())
                target = pad_sequence(target, batch_first=True, padding_value=0)
                # turn target into integer
                target = target.type(torch.int64)
                loss = 0

                for i in range(len(target)):
                    length_without_padding = length[i].item()

                    target_rank = torch.tensor(target[i][:length_without_padding].tolist()).to(self.device)
                    target_expanded = torch.tensor(target_rank, dtype=torch.float).unsqueeze(1).expand(
                        length_without_padding,
                        length_without_padding)
                    rank_diff = (target_expanded - target_expanded.t()).to(self.device)

                    if do_glob:
                        loss += loss_fn(adjacency[i], target_rank)

                    else:
                        if self.train_mode == "local":
                            # gold_adjacency = binary_pairwise_target(target[i], adjacency[i])
                            # loss += F.binary_cross_entropy(torch.sigmoid(adjacency[i]), gold_adjacency, weight=(F.leaky_relu(-rank_diff
                            #     ) / (0.5 * length[i])) + 0.1)
                            # loss += F.binary_cross_entropy(torch.sigmoid(adjacency[i]), gold_adjacency,
                            #                                weight=(torch.abs(-rank_diff
                            #                                                     ) / (0.5 * length[i])) + 0.1)
                            gold_adjacency = binary_consecutive_target(target[i], adjacency[i])
                            loss += F.cross_entropy(adjacency[i], gold_adjacency, weight=target_rank)
                        else:
                            gold_adjacency = binary_consecutive_target(target[i], adjacency[i])
                            loss += F.cross_entropy(adjacency[i], gold_adjacency, weight=target_rank)

                    steps_num += 1

                    #
                    # local_loss += F.binary_cross_entropy(adjacency[i], gold_adjacency, weight=(torch.abs(F.leaky_relu(-rank_diff)) / (0.5*length[i])) + 0.1)

                    # loss += loss_function(adjacency[i],
                    #                       gold_adjacency)

                loss.backward()
                optimizer.step()
                # scheduler.step()
                epoch_loss += (loss / len(target))
                idx += 1

            epoch_loss /= len(train_loader)

            logger.log({"train_loss": epoch_loss.item()}, step=epoch+1)
            losses.append(epoch_loss.item())
            loop.set_description('Loss: {:.4f} | LR: {:.5f} | Change: {:.4f} | Valid {}: {:.4f}'.format(epoch_loss.item(),
                                                                                                        # scheduler.get_last_lr()[0],
                                                                                                        optimizer.param_groups[0]['lr'],
                                                                                                        losses[-1] -
                                                                                                        losses[
                                                                                                            -2] if len(
                                                                                                            losses) > 1 else 0,
                                                                                                        eval_metric,
                                                                                                        valid_metric))
            if (epoch+1) % eval_freq == 0:
                # evaluate on 10% of the training set

                # evaluate on the validation set
                complete_valid_metric = self.evaluate(valid_loader, verbose=False)[0]
                valid_metric = complete_valid_metric[eval_metric.lower()]

                logger.log({f"valid_{eval_metric}": valid_metric}, step=epoch+1)
                complete_test_metric = self.evaluate(test_loader, verbose=False)[0]
                test_metric = complete_test_metric[eval_metric.lower()]
                logger.log({f"test_{eval_metric}": test_metric}, step=epoch + 1)


                if (higher_better and valid_metric > best_metric) or (not higher_better and valid_metric < best_metric):
                    best_metric = valid_metric
                    self.save_checkpoint(checkpoint_path)
                    print(f"Store the best model at epoch {epoch+1} with valid {eval_metric} = {round(best_metric, 4)}")
                    print(complete_valid_metric)
                    print(complete_test_metric)
                    # print all the metrics


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

        ndcg_3 = round(ndcg_score(reverse_gold_rankings, reverse_pred_rankings, k=3), 4)
        ndcg_5 = round(ndcg_score(reverse_gold_rankings, reverse_pred_rankings, k=5), 4)
        ndcg_10 = round(ndcg_score(reverse_gold_rankings, reverse_pred_rankings, k=10), 4)
        metrics = {"rmse": rmse, "acc": acc, "mrr": mrr, "kendall_tau": kendall_tau, "ndcg_3": ndcg_3, "ndcg_5": ndcg_5, "ndcg_10": ndcg_10}

        # metrics = {"rmse": rmse, "acc": acc, "mrr": mrr, "kendall_tau": kendall_tau}
        return metrics, pred_rankings, gold_rankings




