import xgboost as xgb
import lightgbm as lgb
from utils import *
from sklearn.metrics import root_mean_squared_error, accuracy_score
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score

class LambdaMART():
    def __init__(self, framework: str = "xgb", n_estimators: int = 1000, loss_fn: str = "rank:pairwise", reverse_ranking: bool = True):
        self.ranker = xgb.XGBRanker(n_estimators=n_estimators, objective=loss_fn, verbose=1) if framework == "xgb" \
            else lgb.LGBMRanker(n_estimators=n_estimators, objective=loss_fn)
        self.n_estimators = n_estimators
        self.loss_fn = loss_fn
        self.reverse_ranking = reverse_ranking
        self.variables = None

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, reverse_ranking: bool = False):
        # only keep the numerical features
        self.variables = train_data.columns
        # sort the data by qid
        train_data = train_data.sort_values(by=['qid'])
        valid_data = valid_data.sort_values(by=['qid'])

        X_train, y_train = train_data.drop(columns=['target']), train_data['target']
        X_valid, y_valid = valid_data.drop(columns=['target']), valid_data['target']
        # reverse the ranking label to make it higher the better
        if reverse_ranking:
            y_train = y_train.max() - y_train + 1
            y_valid = y_valid.max() - y_valid + 1
        else:
            self.reverse_ranking = False

        if self.loss_fn == "lambdarank":
            self.ranker.fit(X_train, y_train, group=train_data.groupby('qid').size().values.tolist(),
                            eval_set=[(X_valid, y_valid)], eval_group=[valid_data.groupby('qid').size().values.tolist()])
        else:
            self.ranker.fit(X_train, y_train,
                            eval_set=[(X_valid, y_valid)],
                            eval_metric='auc',
                            verbose=1,
                            early_stopping_rounds=50)
        return


    def evaluate(self, test_data: pd.DataFrame):
        test_data = test_data[self.variables]
        X_test = test_data.drop(columns=['target'])
        if self.reverse_ranking:
            test_data['lmart_label'] = test_data.groupby('qid')['target'].rank(ascending=False, method='first').astype(int)
        else:
            test_data['lmart_label'] = test_data['target']

        ranking_score = self.ranker.predict(X_test)
        test_data['ranking_score'] = ranking_score

        # turn ranking scores into rankings in each qids
        test_data['ranking'] = test_data.groupby('qid')['ranking_score'].rank(ascending=True, method='first')
        # convert into integer
        test_data['ranking'] = test_data['ranking'].astype(int)

        rmse = round(root_mean_squared_error(test_data['lmart_label'], test_data['ranking']), 4)
        acc = round(accuracy_score(test_data['lmart_label'], test_data['ranking']), 4)
        mrr = round(test_data.groupby('qid').apply(
            lambda x: get_mrr(x['ranking'].tolist(), x['lmart_label'].tolist(), reverse=True)).mean(), 4)
        ken_tau = round(test_data.groupby('qid').apply(
            lambda x: kendalltau(x['ranking'].tolist(), x['lmart_label'].tolist())[0]).mean(), 4)
        pred_rankings = test_data.groupby('qid').apply(lambda x: x['ranking'].tolist()).tolist()
        gold_rankings = test_data.groupby('qid').apply(lambda x: x['lmart_label'].tolist()).tolist()


        try:
            ndcg_3 = round(ndcg_score(gold_rankings, pred_rankings, k=3), 4)
            ndcg_5 = round(ndcg_score(gold_rankings, pred_rankings, k=5), 4)
            ndcg_10 = round(ndcg_score(gold_rankings, pred_rankings, k=10), 4)
            metrics = {"rmse": rmse, "acc": acc, "mrr": mrr, "kendall_tau": ken_tau, "ndcg_3": ndcg_3, "ndcg_5": ndcg_5,
                       "ndcg_10": ndcg_10}
        except:
            metrics = {"rmse": rmse, "acc": acc, "mrr": mrr, "kendall_tau": ken_tau}
        # print(metrics)

        return metrics, pred_rankings, gold_rankings