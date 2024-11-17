import os.path
import yaml
import json

from tqdm import tqdm
from model.stock_lambdamart_pipeline import StockLambdaMARTPipeline
from preprocess.data_loader import load_data
# from model.discrete_pipeline import HRDiscretePipeline
from model.stock_marginal_pipeline import StockMarginalPipeline
from model.stock_baseline_pipeline import StockBaselinePipeline
from model.stock_rankformer_pipeline import StockRankformerPipeline
import warnings
# import wandb
import argparse
from preprocess.dataset_process import preprocess_data
from utils import *
import networkx as nx
tsp = nx.approximation.traveling_salesman_problem
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):

    with open("./data/stock_rank/{}_industry_ticker_filtered.json".format(args.market), "r") as f:
        industry_ticker = json.load(f)

    if args.sector != "all":
        train_stocks = [industry_ticker[args.sector]]
    else:
        train_stocks = [industry_ticker[sector] for sector in industry_ticker.keys()]

    start_from = 0
    idx = start_from
    for stocks in train_stocks[start_from:]:
        if idx >= len(train_stocks):
            break
        print()
        print("="*25 + f" {idx}/{len(train_stocks)-1}  " + "="*25)
        print(f"Training on {list(industry_ticker.keys())[idx]} sector")
        print(f"Number of stocks in {list(industry_ticker.keys())[idx]}: {len(stocks)}")
        if len(stocks) > 30 or len(stocks) < 5:
            idx += 1
            continue
        if args.pipeline == "marginal":
            pipeline = StockMarginalPipeline(stocks=stocks,
                                             market_name=args.market,
                                             nb_layers=args.tf_num_layers,
                                             nb_heads=args.tf_nhead,
                                             dim_ff=args.tf_dim_ff,
                                             train_mode=args.learning,
                                             device=device,
                                             tsp_solver=args.tsp_solver,)
        elif args.pipeline == "baseline":
            pipeline = StockBaselinePipeline(stocks=stocks,
                                             market_name=args.market,
                                             device=device,)
        elif args.pipeline == "rankformer":
            pipeline = StockRankformerPipeline(stocks=stocks,
                                               market_name=args.market,
                                               nb_layers=args.tf_num_layers,
                                               nb_heads=args.tf_nhead,
                                               dim_ff=args.tf_dim_ff,
                                               device=device,)
        elif args.pipeline == "lambdamart":
            pipeline = StockLambdaMARTPipeline(stocks=stocks,
                                               market_name=args.market,)
        else:
            raise NotImplementedError


        replace_checkpoints = True

        learning_str_map = {"local": "local", "global": "global", "hybrid": "global"}


        checkpoint_path_dir = os.path.join("checkpoints", f"{args.pipeline}_stock_{learning_str_map[args.learning]}", f"{args.market}_{list(industry_ticker.keys())[idx]}") if args.pipeline != "baseline" else os.path.join("checkpoints", f"{args.pipeline}_stock", f"{args.market}_{list(industry_ticker.keys())[idx]}")

        print("Starting training...")

        pipeline.fit(epochs=args.epochs,
                     optimizer=args.optimizer,
                     replace_checkpoints=replace_checkpoints,
                     learning_rate=args.lr,
                     weight_decay=args.weight_decay,
                     eval_freq=args.eval_freq,
                     eval_metric=args.eval_metric,
                     checkpoint_path=checkpoint_path_dir,
                     continue_from_checkpoint=args.from_checkpoint,
                     )
                     # eval_first=True if args.from_checkpoint is not None else False,)
        idx += 1

    return

def test(args):
    with open("./data/stock_rank/{}_industry_ticker_filtered.json".format(args.market), "r") as f:
        industry_ticker = json.load(f)

    if "local" in args.checkpoint:
        learning_type = "local"
    elif "global" in args.checkpoint:
        learning_type = "global"
    elif "hybrid" in args.checkpoint:
        learning_type = "global"
    else:
        learning_type = "unknown"

    if args.sector != "all":
        test_stocks = [industry_ticker[args.sector]]
    else:
        test_stocks = [industry_ticker[sector] for sector in industry_ticker.keys()]

    result_df = pd.DataFrame(columns=["sector", "stock_num", "rmse", "acc", "mrr", "kendall_tau", "IRR@1", "SR@1", "MAP@1", "IRR@3", "SR@3", "MAP@3", "IRR@5", "SR@5", "MAP@5"])

    start_from = 0
    idx = start_from
    for stocks in tqdm(test_stocks[start_from:]):

        if idx >= len(test_stocks):
            break
        checkpoint_path = os.path.join(args.checkpoint, f"{args.market}_{list(industry_ticker.keys())[idx]}")
        if (not os.path.exists(checkpoint_path)) or (len(stocks) < 5) or (len(stocks) > 30):
            idx += 1
            continue

        print()
        print("=" * 25 + f" {idx}/{len(test_stocks) - 1}  " + "=" * 25)
        print(f"Evaluate on {list(industry_ticker.keys())[idx]} sector")
        print(f"Number of stocks in {list(industry_ticker.keys())[idx]}: {len(stocks)}")

        # load the yaml config file
        try:
            config = yaml.load(open(os.path.join(args.checkpoint, f"{args.market}_{list(industry_ticker.keys())[idx]}", "config.yaml"), "r"), Loader=yaml.FullLoader)
        except:
            config = None
    # if args.pipeline == "discrete":
    #     pipeline = DiscretePipeline(input_dim=config['input_dim'],
    #                                 nb_layers=config['nb_layers'],
    #                                 nb_heads=config['nb_heads'],
    #                                 dim_ff=config['dim_ff'],
    #                                 batchnorm=config['batchnorm'],
    #                                 dim_emb=config['dim_emb'],
    #                                 tsp_solver=args.tsp_solver)
        if args.pipeline == "marginal":
            pipeline = StockMarginalPipeline(stocks=stocks,
                                             market_name=args.market,
                                             nb_layers=config['nb_layers'],
                                             nb_heads=config['nb_heads'],
                                             dim_ff=config['dim_ff'],
                                             train_mode=learning_type,
                                             tsp_solver=args.tsp_solver)
        elif args.pipeline == "baseline":
            pipeline = StockBaselinePipeline(stocks=stocks,
                                             market_name=args.market)
        elif args.pipeline == "rankformer":
            pipeline = StockRankformerPipeline(stocks=stocks,
                                               market_name=args.market,
                                               nb_layers=1,
                                               nb_heads=8,
                                               dim_ff=128,)
        elif args.pipeline == "lambdamart":
            pipeline = StockLambdaMARTPipeline(stocks=stocks,
                                               market_name=args.market)
        else:
            raise NotImplementedError

        pipeline.load_checkpoint(os.path.join(args.checkpoint, f"{args.market}_{list(industry_ticker.keys())[idx]}"))

        exp_name = f"test_stock_{args.pipeline}_{list(industry_ticker.keys())[idx]}"

    # wandb_logger = wandb.init(project="TSPRank",
    #                           name=exp_name,
    #                           tags=["test", dataset_name, args.pipeline, args.tsp_solver],
    #                           entity="uoe-turing")

        metrics, pred_rankings, gold_rankings = pipeline.evaluate()
        new_row = {"sector": list(industry_ticker.keys())[idx],
                   "stock_num": metrics['stock_num'],
                   "rmse": round(metrics['rmse'], 4),
                   "acc": round(metrics['acc'], 4),
                   "mrr": round(metrics['mrr'], 4),
                   "kendall_tau": round(metrics['kendall_tau'], 5),
                   "IRR@1": round(metrics['IRR@1'], 4),
                   "SR@1": round(metrics['SR@1'], 4),
                   "MAP@1": round(metrics['MAP@1'], 4),
                   "IRR@3": round(metrics['IRR@3'], 4),
                   "SR@3": round(metrics['SR@3'], 4),
                   "MAP@3": round(metrics['MAP@3'], 4),
                   "IRR@5": round(metrics['IRR@5'], 4),
                   "SR@5": round(metrics['SR@5'], 4),
                   "MAP@5": round(metrics['MAP@5'], 4)}
        result_df = result_df._append(new_row, ignore_index=True)
        print(new_row)
        idx += 1
    # wandb_logger.log(metrics)
    # wandb_logger.finish()
    # calculate average performance metrics, add one more row to the result_df
    # let stock_num col be int type
    result_df["stock_num"] = result_df["stock_num"].astype(int)
    avg_metrics = {"sector": "average",
                   "stock_num": result_df["stock_num"].mean(),
                    "rmse": result_df["rmse"].mean(),
                    "acc": result_df["acc"].mean(),
                    "mrr": result_df["mrr"].mean(),
                    "kendall_tau": result_df["kendall_tau"].mean(),
                    "IRR@1": result_df["IRR@1"].mean(),
                    "SR@1": result_df["SR@1"].mean(),
                    "MAP@1": result_df["MAP@1"].mean(),
                    "IRR@3": result_df["IRR@3"].mean(),
                    "SR@3": result_df["SR@3"].mean(),
                    "MAP@3": result_df["MAP@3"].mean(),
                    "IRR@5": result_df["IRR@5"].mean(),
                    "SR@5": result_df["SR@5"].mean(),
                    "MAP@5": result_df["MAP@5"].mean()}
    result_df = result_df._append(avg_metrics, ignore_index=True)

    print(avg_metrics)

    if args.pipeline == "baseline":
        result_df.to_csv(f"./logs/stock_baseline_{args.market}_result.csv", index=False)
    elif args.pipeline == "marginal":
        result_df.to_csv(f"./logs/stock_marginal_{args.market}_{learning_type}_result.csv", index=False)
    elif args.pipeline == "rankformer":
        result_df.to_csv(f"./logs/stock_rankformer_{args.market}_result.csv", index=False)
    elif args.pipeline == "lambdamart":
        result_df.to_csv(f"./logs/stock_lambdamart_{args.market}_result.csv", index=False)
    return

# main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument("--sector", type=str, required=True)
    train_parser.add_argument("--market", type=str, choices=["NASDAQ", "NYSE"], required=True)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--learning", type=str, choices=["local", "global", "hybrid"], default="local")
    train_parser.add_argument("--dim_emb", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--optimizer", type=str, default="adam")
    train_parser.add_argument("--tf_dim_ff", type=int, default=32)
    train_parser.add_argument("--tf_nhead", type=int, default=2)
    train_parser.add_argument("--tf_num_layers", type=int, default=4)
    train_parser.add_argument("--pipeline", type=str, choices=["baseline", "discrete", "marginal", "rankformer", "lambdamart"], default="marginal")
    train_parser.add_argument("--tsp_solver", type=str, choices=["greedy", "gurobi"], default="gurobi")
    train_parser.add_argument("--eval_freq", type=int, default=10)
    train_parser.add_argument("--eval_metric", type=str, default="rmse")
    train_parser.add_argument("--sample_size", type=int, default=-1)
    train_parser.add_argument("--from_checkpoint", type=bool, default=False)

    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument("--sector", type=str, required=True)
    test_parser.add_argument("--market", type=str, choices=["NASDAQ", "NYSE"], required=True)
    test_parser.add_argument('--checkpoint', type=str, required=True, default=None)
    test_parser.add_argument("--pipeline", type=str, choices=["baseline", "discrete", "marginal", "rankformer", "lambdamart"], default="marginal")
    test_parser.add_argument("--tsp_solver", type=str, choices=["greedy", "gurobi"], default="greedy")


    test_parser.set_defaults(func=test)
    args = parser.parse_args()
    print(args)
    args.func(args)