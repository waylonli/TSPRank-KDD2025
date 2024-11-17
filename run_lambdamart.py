import pickle
import warnings
import wandb
import argparse
from model.lambdamart import LambdaMART
from utils import *
from preprocess.dataset_process import preprocess_data
import networkx as nx
tsp = nx.approximation.traveling_salesman_problem
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_lambdamart(args):

    train_data, valid_data, test_data = preprocess_data(args, fold=args.fold, n_largest=args.top)
    print("Train data qids: ", len(train_data['qid'].unique()))
    print("Valid data qids: ", len(valid_data['qid'].unique()))
    print("Test data qids: ", len(test_data['qid'].unique()))

    reverse_rank = True if args.dataset in ["mq2008", "otd2"] else False

    future_columns = [col for col in train_data.columns if col.startswith('future_')]
    train_data = train_data.drop(columns=future_columns)
    valid_data = valid_data.drop(columns=future_columns)

    if args.dataset == "otd2":
        feature_columns = [col for col in train_data.columns if col.startswith('feature_')]
        train_data = train_data[feature_columns + ['target', 'qid']]
        valid_data = valid_data[feature_columns + ['target', 'qid']]
        test_data = test_data[feature_columns + ['target', 'qid']]

    if args.framework == "xgb":
        assert args.loss_fn in ["rank:map", "rank:pairwise", "rank:ndcg"]
        lmart_model = LambdaMART(framework=args.framework, n_estimators=args.n_estimators, loss_fn=args.loss_fn)
    elif args.framework == "lgbm":
        lmart_model = LambdaMART(framework=args.framework, n_estimators=args.n_estimators, loss_fn="lambdarank")

    lmart_model.fit(train_data=train_data, valid_data=valid_data, reverse_ranking=reverse_rank)
    # get feature importance
    feature_importance = lmart_model.ranker.feature_importances_
    feature_importance = pd.DataFrame({'feature': train_data.drop(columns=['target', 'qid']).columns,
                                       'importance': feature_importance}).sort_values(by='importance', ascending=False)
    print(feature_importance)

    metrics, _, _ = lmart_model.evaluate(test_data)
    print(metrics)

    # store the pretrained model using pickle
    # checkpoints = f"checkpoints/lmart_{args.dataset}_fold{args.fold}_{args.top}"
    # os.makedirs(checkpoints, exist_ok=True)
    # with open(f"{checkpoints}/model.pkl", "wb") as f:
    #     pickle.dump(lmart_model, f)
    # f.close()

    exp_name = f"lmart_{args.dataset}_fold{args.fold}_top{args.top}_{args.framework}_{args.n_estimators}_{args.loss_fn}"
    wandb_logger = wandb.init(project="TSPRank",
                                name=exp_name,
                                tags=["lmart", args.dataset, args.framework, f"n_estimators_{args.n_estimators}", args.loss_fn],
                                entity="uoe-turing")

    wandb_logger.log(metrics)

    wandb_logger.finish()

    return




# main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--sample_size", type=int, default=-1)
    parser.add_argument("--embed", type=str, default="openai")
    parser.add_argument("--custom_exp_name", type=str, default=None)
    parser.add_argument("--framework", type=str, required=True, choices=["xgb", "lgbm"], default=None)
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--loss_fn", type=str, default="lambdarank")
    parser.add_argument("--numerical_only", type=bool, default=True)

    args = parser.parse_args()
    print(args)
    run_lambdamart(args)
