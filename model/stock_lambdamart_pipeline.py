import fsspec
import torch.nn
import wandb
import yaml
import xgboost as xgb
from tqdm import tqdm
from utils import *
import os
import pickle
from model.loss import *
from torch.nn import BCELoss
import scipy.stats as sps
import random
from sklearn.metrics import root_mean_squared_error, accuracy_score
from scipy.stats import kendalltau


class StockLambdaMARTPipeline():
    def __init__(self, stocks: list,
                 market_name: str,
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
            self.emb_dim = 128
            self.embedding = np.load("./data/stock_rank/NASDAQ_embeddings.npy")[self.stocks, :, :]

        elif market_name == "NYSE":
            self.steps = 1
            self.sequence = 8
            self.emb_dim = 64
            self.embedding = np.load("./data/stock_rank/NYSE_embeddings.npy")[self.stocks, :, :]

        self.lambdamart_ranker = xgb.XGBRanker(
                                    objective='rank:pairwise',
                                    eval_metric='ndcg',
                                    learning_rate=0.1,
                                    n_estimators=10000,
                                    n_jobs=4
                                )

        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(self.data_path, self.market_name, self.tickers, self.steps)

        print(self.embedding.shape)
        self.trade_dates = self.eod_data.shape[1]
        # self.sample_k_companies(30)
        self.valid_index = 756
        self.test_index = 1008
        self.est_index = 1008
        self.train_data, self.valid_data, self.test_data = self.convert_to_df()


    def convert_to_df(self):
        labels_data = np.ones_like(self.gt_data)
        # get ordinal ranking
        for t in range(self.gt_data.shape[1]):
            labels_data[:, t] = self.gt_data.shape[0] + 1 - sps.rankdata(self.gt_data[:, t], method='ordinal')
        labels_data = labels_data[:, self.sequence:]

        # put embeddings and labels into a dataframework
        pd_data = pd.DataFrame(columns=['qid', 'target'] + [f'f_{i}' for i in range(self.emb_dim)])
        for i in tqdm(range(labels_data.shape[1])):
            for j in range(labels_data.shape[0]):
                new_item = {
                    'qid': i + 1,
                    'target': labels_data[j, i],
                    'gt': self.gt_data[j, i],
                }
                for d in range(self.emb_dim):
                    new_item[f'f_{d}'] = self.embedding[j, i, d]
                pd_data = pd_data._append(new_item, ignore_index=True)

        train_data = pd_data[pd_data['qid'] <= self.valid_index - self.sequence - self.steps + 1]
        valid_data = pd_data[(pd_data['qid'] > self.valid_index - self.sequence - self.steps + 1) & (
                    pd_data['qid'] <= self.test_index - self.sequence - self.steps + 1)]
        test_data = pd_data[pd_data['qid'] > self.test_index - self.sequence - self.steps + 1]
        return train_data, valid_data, test_data


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

    def fit(self, **kwargs):

        self.lambdamart_ranker.fit(
            self.train_data.drop(columns=['qid', 'target']),
            self.train_data['target'],
            group=self.train_data.groupby('qid').size().values,
            eval_set=[(self.valid_data.drop(columns=['qid', 'target']), self.valid_data['target'])],
            eval_group=[self.valid_data.groupby('qid').size().values],
            early_stopping_rounds=20,
            verbose=True
        )
        os.makedirs(kwargs['checkpoint_path'], exist_ok=True)
        self.save_checkpoint(kwargs['checkpoint_path'])


    def evaluate(self,**kwargs):
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

        for qid in self.test_data['qid'].unique():
            cur_qid = self.test_data[self.test_data['qid'] == qid]
            cur_pred = self.lambdamart_ranker.predict(cur_qid.drop(columns=['qid', 'target']))
            pred_rank = torch.tensor(sps.rankdata(cur_pred, method='ordinal'))
            target_rank = torch.tensor(cur_qid['target'].tolist())
            gt_batch = torch.tensor(cur_qid['gt'].tolist())

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
        pickle.dump(self.lambdamart_ranker, open(os.path.join(path, "model.pkl"), "wb"))


    def load_checkpoint(self, path: str):
        # print("Loading checkpoint from", path)
        path = os.path.join(path, "model.pkl") if os.path.isdir(path) else path
        self.lambdamart_ranker = pickle.load(open(path, "rb"))

        return
