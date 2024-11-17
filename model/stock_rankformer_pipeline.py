import fsspec
import torch.nn
import wandb
import yaml
from model.rankformer import RankFormer
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


class StockRankformerPipeline():
    def __init__(self, stocks: list,
                 market_name: str,
                 nb_layers: int = 2,
                 nb_heads: int = 2,
                 dim_ff: int = 32,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):


        self.stock_num = len(stocks)
        self.stocks = stocks


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

        self.rankformer = RankFormer(input_dim=self.input_dim, dim_emb=self.dim_emb, tf_dim_feedforward=dim_ff, tf_nhead=nb_heads,
                                     tf_num_layers=nb_layers)
        self.rankformer.to(device)
        self.config = {
            'input_dim': self.input_dim,
            'nb_layers': nb_layers,
            'dim_emb': self.dim_emb,
            'nb_heads': nb_heads,
            'dim_ff': dim_ff
        }

        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(self.data_path, self.market_name, self.tickers, self.steps)

        print(self.embedding.shape)
        self.trade_dates = self.eod_data.shape[1]
        # self.sample_k_companies(30)
        self.valid_index = 756
        self.test_index = 1008
        self.est_index = 1008
        self.device = device

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
            epochs: int = 50,
            optimizer: str = "adam",
            replace_checkpoints: bool = False,
            eval_freq: int = 10,
            eval_metric: str = "rmse",
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            checkpoint_path: str = "checkpoints/marginal_pipeline",
            continue_from_checkpoint: bool = False,
            ):


        if optimizer == "adam":
            optimizer = torch.optim.Adam(self.rankformer.parameters(), lr=learning_rate, weight_decay=0)
        else:
            optimizer = torch.optim.SGD(self.rankformer.parameters(), lr=learning_rate)

        total_steps = epochs * (self.valid_index - self.sequence - self.steps + 1)
        warmup_steps = int(total_steps * 0.2)
        print(f"Using warmup steps: {warmup_steps} out of {total_steps} steps")

        # scheduler = optimization.get_polynomial_decay_schedule_with_warmup(optimizer,
        #                                                                    num_warmup_steps=warmup_steps,
        #                                                                    num_training_steps=total_steps,
        #                                                                    lr_end=learning_rate / 100,
        #                                                                    power=3)

        losses = []
        loop = tqdm(range(epochs), total=epochs, leave=True)

        higher_better = False if eval_metric.lower() in ["rmse"] else True
        best_metric = -float('inf') if higher_better else float('inf')
        best_test_metric = -float('inf') if higher_better else float('inf')

        if os.path.exists(checkpoint_path) and replace_checkpoints:
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

        for epoch in loop:
            self.rankformer.train()
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            for j in range(self.valid_index - self.sequence -
                           self.steps + 1):

                optimizer.zero_grad()
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(batch_offsets[j])

                length = torch.tensor([self.stock_num]).to(self.device)
                scores = self.rankformer(emb_batch, length)

                target_rank = (mask_batch.shape[0] - torch.tensor(
                    sps.rankdata(gt_batch.cpu().detach().numpy(), method='ordinal') - 1)).to(self.device)
                loss = self.rankformer.compute_loss(scores, target_rank, length)
                loss.backward()
                optimizer.step()
                tra_loss += loss.item()

                # scheduler.step()
            tra_loss /= (self.valid_index - self.sequence - self.steps + 1)
            # logger.log({"train_loss": epoch_loss.item()}, step=epoch+1)
            losses.append(float(tra_loss))
            loop.set_description('Loss: {:.4f} | LR: {:.5f} | Change: {:.4f} | Train {}: {:.4f}'.format(tra_loss,
                                                                                                        optimizer.param_groups[0]['lr'],
                                                                                                        losses[-1] -
                                                                                                        losses[
                                                                                                            -2] if len(
                                                                                                            losses) > 1 else 0,
                                                                                                        eval_metric,
                                                                                                        train_metric))
            if (epoch + 1) % eval_freq == 0:

                train_complete_metric = self.evaluate(train_valid_index, verbose=False)[0]
                train_metric = train_complete_metric[eval_metric.lower()]
                # evaluate on the validation set
                valid_complete_metric = self.evaluate(range(self.valid_index - self.sequence - self.steps + 1,
                                                            self.test_index - self.sequence - self.steps + 1),
                                                      verbose=False)[0]
                valid_metric = valid_complete_metric[eval_metric.lower()]
                # evaluate on the test set
                test_complete_metric = self.evaluate(range(self.test_index - self.sequence - self.steps + 1,
                                                           self.trade_dates - self.sequence - self.steps + 1),
                                                     verbose=False)[0]
                test_metric = test_complete_metric[eval_metric.lower()]
                # logger.log({f"valid_{eval_metric}": metric}, step=epoch+1)

                if (higher_better and valid_metric > best_metric) or (
                        (not higher_better) and valid_metric < best_metric):
                    best_metric = valid_metric
                    best_test_metric = test_metric
                    print(f"-Store the best model at epoch {epoch + 1}-")
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

        self.rankformer.eval()
        pred_rankings = []
        gold_rankings = []
        mrr = 0
        invalid_num = 0
        kendall_tau = 0
        valid_num = 1

        top_1_pred = []
        top_1_gold = []
        top_3_pred = []
        top_3_gold = []
        top_5_pred = []
        top_5_gold = []
        gt_batches = []

        for t in time_steps:
            emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(t)

            length = torch.tensor([self.stock_num]).to(self.device)
            target_rank = (mask_batch.shape[0] - torch.tensor(
                sps.rankdata(gt_batch.cpu().detach().numpy(), method='ordinal') - 1)).cpu()
            scores = self.rankformer(emb_batch, length)
            sorted_idx = torch.argsort(scores, descending=True)
            ori_pred_rank = torch.zeros_like(sorted_idx)
            ori_pred_rank[sorted_idx] = torch.arange(len(sorted_idx)).to(device)
            rank_gap = target_rank.min() - ori_pred_rank.min()
            pred_rank = (ori_pred_rank + rank_gap).detach().cpu()
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

        for k, top_k_pred, top_k_gold in zip([1, 3, 5], [top_1_pred, top_3_pred, top_5_pred],
                                             [top_1_gold, top_3_gold, top_5_gold]):
            bt_long = 1.0
            sharpe_ratio = []
            for pred, gold, gt_batch in zip(top_k_pred, top_k_gold, gt_batches):
                true_return_pred = float(torch.mean(gt_batch[pred]))
                bt_long *= (1 + true_return_pred)
                sharpe_ratio.append(true_return_pred)

            bt_long -= 1
            sharpe_ratio = np.array(sharpe_ratio)
            sharpe_ratio = np.mean(sharpe_ratio) / np.std(sharpe_ratio) * 15.87

            map_k = mapk(top_k_gold, top_k_pred, k)

            metrics.update({f"IRR@{k}": bt_long, f"SR@{k}": sharpe_ratio, f"MAP@{k}": map_k})

        return metrics, pred_rankings, gold_rankings

    def save_checkpoint(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.rankformer.state_dict(), os.path.join(path, "model.pth"))
        config = {"sequence": self.sequence,
                  "steps": self.steps,
                  "stock_num": self.stock_num,
                  "market_name": self.market_name}
        with open(os.path.join(path, "config.yaml"), "w") as f:
            yaml.dump(config, f)


    def load_checkpoint(self, path: str):
        path = os.path.join(path, "model.pth") if os.path.isdir(path) else path
        self.rankformer.load_state_dict(torch.load(path))

        return
