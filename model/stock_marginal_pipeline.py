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
from sklearn.metrics import root_mean_squared_error, accuracy_score
from scipy.stats import kendalltau


class StockMarginalPipeline(BasePipeline):
    def __init__(self, stocks: list,
                 market_name: str,
                 train_mode: str,
                 nb_layers: int = 2,
                 nb_heads: int = 2,
                 dim_ff: int = 32,
                 tsp_solver: str = "gurobi",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(StockMarginalPipeline, self).__init__(input_dim=128, nb_layers=nb_layers, dim_emb=128,
                                                    nb_heads=nb_heads, dim_ff=dim_ff, device=device)

        self.stock_num = len(stocks)
        self.stocks = stocks
        self.train_mode = train_mode

        if tsp_solver == "greedy":
            self.test_tsp_solver = discrete_greedy_tsp_solver
        elif tsp_solver == "gurobi":
            self.test_tsp_solver = discrete_gurobi_tsp_solver
        else:
            raise NotImplementedError
        self.data_path = "./data/stock_rank/2013-01-01"
        self.market_name = market_name
        self.tickers_name = f'{market_name}_tickers_qualify_dr-0.98_min-5_smooth.csv'
        self.tickers = np.genfromtxt(os.path.join(self.data_path, '..', self.tickers_name),
                                     dtype=str, delimiter='\t', skip_header=False)[self.stocks]

        if market_name == "NASDAQ":
            self.steps = 1
            self.sequence = 16
            self.input_dim = 128
            self.dim_emb = 128
            self.embedding = np.load("./data/stock_rank/NASDAQ_embeddings.npy")[self.stocks, :, :]
        elif market_name == "NYSE":
            self.steps = 1
            self.sequence = 8
            self.input_dim = 64
            self.dim_emb = 64
            self.embedding = np.load("./data/stock_rank/NYSE_embeddings.npy")[self.stocks, :, :]

        self.graph_encoder = PairwiseModel(input_dim=self.input_dim, nb_layers=nb_layers, dim_emb=self.dim_emb,
                                           nb_heads=nb_heads,
                                           dim_ff=dim_ff, need_embedding=False, deep=False).to(device)


        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(self.data_path, self.market_name, self.tickers, self.steps)

        self.trade_dates = self.eod_data.shape[1]
        # self.sample_k_companies(30)
        self.valid_index = 756
        self.test_index = 1008
        self.est_index = 1008
        self.device = device
        # initialize the graph encoder using xavier uniform
        # torch.nn.init.xavier_uniform_(self.graph_encoder.weight)

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.sequence
        mask_batch = torch.tensor(self.mask_data[:, offset: offset + seq_len + self.steps]).float().to(self.device)
        mask_batch = torch.min(mask_batch, axis=1).values
        return torch.tensor(self.embedding[:, offset, :]).float().to(self.device), \
            mask_batch.unsqueeze(1), \
            torch.tensor(self.price_data[:, offset + self.sequence - 1]).float().to(self.device).unsqueeze(1), \
            torch.tensor(self.gt_data[:, offset + self.sequence + self.steps - 1]).float().to(self.device).unsqueeze(1)

    def fit(self,
            epochs: int = 200,
            optimizer: str = "adam",
            replace_checkpoints: bool = False,
            eval_freq: int = 10,
            eval_metric: str = "rmse",
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            checkpoint_path: str = "checkpoints/marginal_pipeline",
            continue_from_checkpoint: bool = False,
            ):

        # logger.watch(self.graph_encoder, log="all", log_freq=10)

        if optimizer == "adam":
            optimizer = torch.optim.Adam(self.graph_encoder.parameters(), lr=learning_rate, weight_decay=0)
        else:
            optimizer = torch.optim.SGD(self.graph_encoder.parameters(), lr=learning_rate)

        total_steps = epochs * (self.valid_index - self.sequence - self.steps + 1)
        warmup_steps = int(total_steps * 0.2)
        print(f"Using warmup steps: {warmup_steps} out of {total_steps} steps")

        scheduler = optimization.get_polynomial_decay_schedule_with_warmup(optimizer,
                                                                           num_warmup_steps=warmup_steps,
                                                                           num_training_steps=total_steps,
                                                                           lr_end=learning_rate / 100,
                                                                           power=3)

        losses = []
        loop = tqdm(range(epochs), total=epochs, leave=True)

        higher_better = False if eval_metric.lower() in ["rmse"] else True
        best_metric = -float('inf') if higher_better else float('inf')


        if continue_from_checkpoint and os.path.exists(os.path.join(checkpoint_path, "model.pth")):
            self.load_checkpoint(checkpoint_path)
            initial_metrics = self.evaluate(range(self.test_index - self.sequence - self.steps + 1,
                                              self.trade_dates - self.sequence - self.steps + 1),
                                        verbose=False)[0]
            best_metric = initial_metrics[eval_metric]
            print(f"Continue from the checkpoint with {eval_metric} = {round(best_metric, 4)}")
        elif os.path.exists(checkpoint_path) and replace_checkpoints:
            try:
                os.remove(os.path.join(checkpoint_path, "model.pth"))
                os.remove(os.path.join(checkpoint_path, "config.yaml"))
            except:
                pass
        else:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        train_metric = -1
        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        # sample 20% of the training index from batch_offsets
        train_valid_index = random.sample(list(batch_offsets), int(0.2 * len(batch_offsets)))

        if self.train_mode == "local":
            do_glob = False
        elif self.train_mode == "global" or self.train_mode == "hybrid":
            do_glob = True
            glob_loss_fn = MaxMarginLoss()
        else:
            raise NotImplementedError

        for epoch in loop:
            self.graph_encoder.train()
            epoch_loss = 0
            np.random.shuffle(batch_offsets)
            idx = epoch % 2
            for j in range(self.valid_index - self.sequence -
                           self.steps + 1):
                if self.train_mode == "hybrid":
                    do_glob = idx % 2
                optimizer.zero_grad()
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(batch_offsets[j])

                adjacency = self.graph_encoder(emb_batch)[0]

                target_rank = list(mask_batch.shape[0] - torch.tensor(
                    sps.rankdata(gt_batch.cpu().detach().numpy(), method='ordinal') - 1))
                target_rank = torch.tensor(target_rank)
                target_expanded = torch.tensor(target_rank).unsqueeze(1).expand(mask_batch.shape[0],
                                                                                mask_batch.shape[0])
                rank_diff = (target_expanded - target_expanded.t()).to(self.device)

                # turn rank_diff to torch.float
                rank_diff = rank_diff.to(torch.float)

                if do_glob:
                    # target_path = torch.argsort(target_rank)
                    # target_matrix = path_to_matrix(target_path, adjacency)
                    #
                    # margin = torch.ones_like(target_matrix) - target_matrix
                    # adjacency += margin
                    #
                    # pred_path = discrete_gurobi_tsp_solver(adjacency.unsqueeze(0)).unsqueeze(1).to(self.device)
                    # pred_matrix = path_to_matrix(pred_path, adjacency)
                    #
                    # target_score = target_matrix * adjacency
                    # pred_score = pred_matrix * (adjacency)
                    #
                    # loss = torch.sum(pred_score - target_score) / self.stock_num
                    loss = glob_loss_fn(adjacency, target_rank, device=self.device)
                else:
                    if self.train_mode == "local":
                        target_adj = binary_pairwise_target(target_rank, adjacency, maximize=True).to(self.device)
                        loss = F.binary_cross_entropy(torch.sigmoid(adjacency), target_adj, weight=(torch.abs(-rank_diff)/(0.5*self.stock_num)))
                    else:
                        target_adj = binary_consecutive_target(target_rank, adjacency)
                        loss = F.cross_entropy(adjacency.squeeze(), target_adj)

                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss
            epoch_loss /= (self.valid_index - self.sequence - self.steps + 1)
            # logger.log({"train_loss": epoch_loss.item()}, step=epoch+1)
            losses.append(float(epoch_loss))
            loop.set_description('Loss: {:.4f} | LR: {:.5f} | Change: {:.4f} | Train {}: {:.4f}'.format(epoch_loss,
                                                                                                        scheduler.get_last_lr()[0],
                                                                                                        losses[-1] -
                                                                                                        losses[
                                                                                                            -2] if len(
                                                                                                            losses) > 1 else 0,
                                                                                                        eval_metric,
                                                                                                        train_metric))
            if (epoch + 1) % eval_freq == 0:

                train_complete_metric = self.evaluate(train_valid_index, verbose=False)[0]
                train_metric = train_complete_metric[eval_metric]
                # evaluate on the validation set
                valid_complete_metric = self.evaluate(range(self.valid_index - self.sequence - self.steps + 1,
                                                            self.test_index - self.sequence - self.steps + 1),
                                                      verbose=False)[0]
                valid_metric = valid_complete_metric[eval_metric]
                # evaluate on the test set
                test_complete_metric = self.evaluate(range(self.test_index - self.sequence - self.steps + 1,
                                                           self.trade_dates - self.sequence - self.steps + 1),
                                                     verbose=False)[0]

                logger.log({f"valid_{eval_metric}": metric}, step=epoch+1)

                if (higher_better and valid_metric > best_metric) or (
                        (not higher_better) and valid_metric < best_metric):
                    best_metric = valid_metric
                    print("Train:", train_complete_metric)
                    print("Valid:", valid_complete_metric)
                    print("Test:", test_complete_metric)
                    self.save_checkpoint(checkpoint_path)
                    # print(f"Store the best model at epoch {epoch+1} with {eval_metric} = {round(best_metric, 4)}")
    def evaluate(self,
                 time_steps: list = None,
                 verbose: bool = True):

        if time_steps is None:
            time_steps = range(self.test_index - self.sequence - self.steps + 1,
                                                           self.trade_dates - self.sequence - self.steps + 1)

        self.graph_encoder.eval()
        pred_rankings = []
        gold_rankings = []
        mrr = 0
        invalid_num = 0
        kendall_tau = 0
        valid_num = 1

        # map_1 = []
        # map_3 = []
        # map_5 = []
        #
        # bt_long1 = 1.0
        # sharpe_ratio1 = []
        #
        # bt_long3 = 1.0
        # sharpe_ratio3 = []
        #
        # bt_long5 = 1.0
        # sharpe_ratio5 = []
        #
        # gold_bt_long5 = 1.0
        # gold_sharpe_ratio5 = []

        top_1_pred = []
        top_1_gold = []
        top_3_pred = []
        top_3_gold = []
        top_5_pred = []
        top_5_gold = []

        gt_batches = []

        for t in time_steps:
            emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(t)

            adjacency = self.graph_encoder(emb_batch)[0].unsqueeze(0)

            target_rank = mask_batch.shape[0] - torch.tensor(sps.rankdata(gt_batch.cpu().detach().numpy(), method='ordinal') - 1)

            pred_path = self.test_tsp_solver(adjacency)
            try:
                pred_rank = path_to_pred_label(pred_path)
            except:
                # print("Mismatched length")
                # print("pred_path length:", len(pred_path))
                # print("target_rank length:", len(target_rank))
                continue

            # assert len(target_path) == gt_batch.shape[0]
            if len(pred_rank) != len(target_rank):
                print("Mismatched length")
                print("pred_rank length:", len(pred_rank))
                print("target_rank length:", len(target_rank))
                continue

            pred_rankings += list(pred_rank)
            gold_rankings += list(target_rank)

            try:
                top_1_pred.append([int(torch.argsort(pred_rank)[0])])
                top_1_gold.append([int(torch.argsort(target_rank)[0])])
                top_3_pred.append(torch.argsort(pred_rank)[:3].tolist())
                top_3_gold.append(torch.argsort(target_rank)[:3].tolist())
                top_5_pred.append(torch.argsort(pred_rank)[:5].tolist())
                top_5_gold.append(torch.argsort(target_rank)[:5].tolist())

                gt_batches.append(gt_batch)

                # top_5_pred = torch.argsort(pred_rank)[:5]
                # top_5_gold = torch.argsort(target_rank)[:5]
                #
                # true_return_top_5_pred = float(torch.mean(gt_batch[top_5_pred]))
                # true_return_top_5_gold = float(torch.mean(gt_batch[top_5_gold]))
                #
                # bt_long5 *= (1 + true_return_top_5_pred)
                # sharpe_ratio5.append(true_return_top_5_pred)
                #
                # gold_bt_long5 *= (1 + true_return_top_5_gold)
                # gold_sharpe_ratio5.append(true_return_top_5_gold)

                mrr += get_mrr(pred_rank, target_rank)
                kendall_tau += kendalltau(pred_rank, target_rank)[0]
            except AssertionError:
                # print("==================Mismatch Labels===================")
                # print("pred_rank:", pred_rank)
                # print("target_rank:", target_rank)
                pass
            valid_num += 1

        mrr /= valid_num
        kendall_tau /= valid_num
        acc = accuracy_score(gold_rankings, pred_rankings)
        rmse = root_mean_squared_error(gold_rankings, pred_rankings)

        metrics = {"stock_num": self.stock_num,
                   "rmse": rmse,
                   "acc": acc,
                   "mrr": mrr,
                   "kendall_tau": kendall_tau}

        for k, top_k_pred, top_k_gold in zip([1, 3, 5], [top_1_pred, top_3_pred, top_5_pred], [top_1_gold, top_3_gold, top_5_gold]):
            bt_long = 1.0
            sharpe_ratio = []
            for pred, gold, gt_batch in zip(top_k_pred, top_k_gold, gt_batches):
                true_return_pred = float(torch.mean(gt_batch[pred]))
                true_return_gold = float(torch.mean(gt_batch[gold]))
                bt_long *= (1 + true_return_pred)
                sharpe_ratio.append(true_return_pred)

            bt_long -= 1
            sharpe_ratio = np.array(sharpe_ratio)
            sharpe_ratio = np.mean(sharpe_ratio) / np.std(sharpe_ratio) * 15.87

            map_k = mapk(top_k_gold, top_k_pred, k)

            metrics.update({f"IRR@{k}": bt_long, f"SR@{k}": sharpe_ratio, f"MAP@{k}": map_k})



        # bt_long5 -= 1
        # sharpe_ratio5 = np.array(sharpe_ratio5)
        # sharpe_ratio5 = np.mean(sharpe_ratio5) / np.std(sharpe_ratio5) * 15.87
        # gold_bt_long5 -= 1
        # gold_sharpe_ratio5 = np.array(gold_sharpe_ratio5)
        # gold_sharpe_ratio5 = np.mean(gold_sharpe_ratio5) / np.std(gold_sharpe_ratio5) * 15.87


        # print("Pred:", metrics)
        # print("Gold:", {"bt_long5": gold_bt_long5, "sharpe_ratio5": gold_sharpe_ratio5})
        return metrics, pred_rankings, gold_rankings
