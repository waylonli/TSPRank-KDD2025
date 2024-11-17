import os.path
import yaml
from preprocess.data_loader import load_data
from model.hr_discrete_pipeline import HRDiscretePipeline
from model.event_marginal_pipeline import EventMarginalPipeline
from model.event_rankformer_pipeline import EventRankformerPipeline
import warnings
import wandb
import argparse
from preprocess.dataset_process import preprocess_data
from utils import *
import networkx as nx
tsp = nx.approximation.traveling_salesman_problem
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    train_data, valid_data, test_data = preprocess_data(args, n_largest=args.top, fold=args.fold)

    train_valid_qids = np.random.choice(train_data['qid'].unique(), int(0.05*len(train_data['qid'].unique())), replace=False)
    train_valid_data = train_data[train_data['qid'].isin(train_valid_qids)]

    print("Train data qids: ", len(train_data['qid'].unique()))
    print("Train valid data qids: ", len(train_valid_data['qid'].unique()))
    print("Valid data qids: ", len(valid_data['qid'].unique()))
    print("Test data qids: ", len(test_data['qid'].unique()))

    future_columns = [col for col in train_data.columns if col.startswith('future_')]

    if len(future_columns) > 0:
        train_data = train_data.drop(columns=future_columns)
        valid_data = valid_data.drop(columns=future_columns)
        test_data = test_data.drop(columns=future_columns)

    if args.learning in ["global", "hybrid"]:
        learning_str = "global"
    else:
        learning_str = "local"


    variables = [col for col in train_data.columns if col.startswith('feature_')]
    print("Num of feats:", len(variables))

    replace_checkpoints = True

    if args.pipeline.lower() not in ["rankformer", "baseline"]:
        checkpoint_path_dir = "checkpoints/{}_{}_fold{}_{}_{}".format(args.pipeline, args.dataset, args.fold, args.top, learning_str)
    else:
        checkpoint_path_dir = "checkpoints/{}_{}_fold{}_{}".format(args.pipeline, args.dataset, args.fold, args.top)

    if replace_checkpoints:
        if not os.path.exists(checkpoint_path_dir):
            os.makedirs(checkpoint_path_dir)
        # export the variable list to the checkpoint folder
        with open(os.path.join(checkpoint_path_dir, "variables.txt"), "w") as f:
            for var in variables:
                f.write(var + "\n")
        f.close()
        with open(os.path.join(checkpoint_path_dir, "dataset.txt"), "w") as f:
            f.write(args.dataset)
        f.close()
        # train_data.to_csv(os.path.join(checkpoint_path_dir, "train.csv"), index=False)
        valid_data.to_csv(os.path.join(checkpoint_path_dir, "valid.csv"), index=False)
        test_data.to_csv(os.path.join(checkpoint_path_dir, "test.csv"), index=True)

    # # concatenate the train, valid and test data
    # train_data = pd.concat([train_data, valid_data, test_data], axis=0)

    train_loader, train_input_dim, train_max_target = load_data(train_data, batch_size=args.batch_size, stage='train',
                                                                variables=variables, qid_column='qid',
                                                                label_column='target', transform=None)
    train_valid_loader, train_valid_input_dim, train_valid_max_target = load_data(train_valid_data, batch_size=1, stage='test',
                                                                variables=variables, qid_column='qid',
                                                                label_column='target', transform=None)
    valid_loader, valid_input_dim, valid_max_target = load_data(valid_data, batch_size=1, stage='test',
                                                                variables=variables, qid_column='qid',
                                                                label_column='target', transform=None)
    test_loader, test_input_dim, test_max_target = load_data(test_data, batch_size=1, stage='test',
                                                            variables=variables, qid_column='qid',
                                                            label_column='target', transform=None)

    if args.pipeline == "marginal":
        pipeline = EventMarginalPipeline(
                                    sentence_embed_dim=train_input_dim,
                                    nb_layers=args.tf_num_layers,
                                    nb_heads=args.tf_nhead,
                                    dim_ff=args.tf_dim_ff,
                                    device=device,
                                    train_mode=args.learning,
                                    tsp_solver=args.tsp_solver,)
    elif args.pipeline == "rankformer":
        pipeline = EventRankformerPipeline(
                                      sentence_embed_dim=train_input_dim,
                                      nb_layers=args.tf_num_layers,
                                      nb_heads=args.tf_nhead,
                                      dim_ff=args.tf_dim_ff,)
    else:
        raise NotImplementedError


    exp_name = f"train_{args.dataset}_fold{args.fold}_{args.pipeline}_{learning_str}"

    wandb_logger = wandb.init(project="TSPRank",
                              name=exp_name,
                              tags=["train", args.dataset, args.pipeline, args.optimizer, args.tsp_solver],
                              entity="uoe-turing")


    if args.from_checkpoint is not None:
        pipeline.load_checkpoint(args.from_checkpoint)

    print("Starting training...")

    pipeline.fit(train_loader,
                 valid_loader,
                 test_loader,
                 wandb_logger,
                 epochs=args.epochs,
                 optimizer=args.optimizer,
                 learning_rate=args.lr,
                 weight_decay=args.weight_decay,
                 eval_freq=args.eval_freq,
                 eval_metric=args.eval_metric,
                 checkpoint_path=checkpoint_path_dir,
                 eval_first=True if args.from_checkpoint is not None else False,)

    wandb_logger.finish()

    return

def test(args):
    # _, _, test_data = preprocess_data(args)
    # train_data = pd.read_csv(os.path.join(args.checkpoint, "train.csv"))
    test_data = pd.read_csv(os.path.join(args.checkpoint, "test.csv"))
    print("Test data qids: ", len(test_data['qid'].unique()))
    # test_data['temp'] = test_data['target']
    dataset_name = open(os.path.join(args.checkpoint, "dataset.txt"), "r").read()
    # read the variables from the checkpoint folder
    with open(os.path.join(args.checkpoint, "variables.txt"), "r") as f:
        variables = f.readlines()
    f.close()
    variables = [var.strip() for var in variables]
    # test_data = test_data[variables]
    print("Num of feats:", len(variables))

    # train_loader, train_input_dim, train_max_target = load_data(train_data, batch_size=1, stage='test',
    #                                                             variables=variables, qid_column='qid',
    #                                                             label_column='target', transform=None)
    test_loader, test_input_dim, test_max_target = load_data(test_data, batch_size=1, stage='test',
                                                            variables=variables, qid_column='qid',
                                                            label_column='target', transform=None)

    # load the yaml config file
    config = yaml.load(open(os.path.join(args.checkpoint, "config.yaml"), "r"), Loader=yaml.FullLoader)

    if args.pipeline == "marginal":
        pipeline = EventMarginalPipeline(
                                    sentence_embed_dim=test_input_dim,
                                    nb_layers=config['nb_layers'],
                                    nb_heads=config['nb_heads'],
                                    dim_ff=config['dim_ff'],
                                    tsp_solver=args.tsp_solver)
    elif args.pipeline == "rankformer":
        pipeline = EventRankformerPipeline(
                                      sentence_embed_dim=test_input_dim,
                                      nb_layers=config['nb_layers'],
                                      nb_heads=config['nb_heads'],
                                      dim_ff=config['dim_ff'],)
    else:
        raise NotImplementedError

    pipeline.load_checkpoint(args.checkpoint)

    if "local" in args.checkpoint:
        learning = "local"
    elif "global" in args.checkpoint:
        learning = "global"
    else:
        learning = "hybrid"

    fold = args.checkpoint.split("_")[2]
    top = args.checkpoint.split("_")[3]


    exp_name = f"test_{dataset_name}_{fold}_top{top}_{args.pipeline}_{learning}" if args.pipeline != "rankformer" else f"test_{dataset_name}_{fold}_top{top}_{args.pipeline}"

    wandb_logger = wandb.init(project="TSPRank",
                              name=exp_name,
                              tags=["test", dataset_name, args.pipeline],
                              entity="uoe-turing")

    # train_metrics, train_pred_rankings, train_gold_rankings = pipeline.evaluate(train_loader)
    test_metrics, pred_rankings, gold_rankings = pipeline.evaluate(test_loader)
    # print("Train metrics:", train_metrics)
    print("Test metrics:", test_metrics)
    wandb_logger.log(test_metrics)
    wandb_logger.finish()

    return

# main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument("--dataset", required=True, type=str, default=None)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--dim_emb", type=int, default=128)
    train_parser.add_argument("--embed", type=str, default="openai")
    train_parser.add_argument("--fold", type=int, default=2)
    train_parser.add_argument("--learning", type=str, choices=["local", "global", "hybrid"], default="local")
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--optimizer", type=str, default="adam")
    train_parser.add_argument("--tf_dim_ff", type=int, default=32)
    train_parser.add_argument("--tf_nhead", type=int, default=2)
    train_parser.add_argument("--tf_num_layers", type=int, default=4)
    train_parser.add_argument("--pipeline", type=str, choices=["discrete", "marginal", "rankformer"], default="marginal")
    train_parser.add_argument("--tsp_solver", type=str, choices=["greedy", "gurobi"], default="gurobi")
    train_parser.add_argument("--eval_freq", type=int, default=10)
    train_parser.add_argument("--eval_metric", type=str, default="rmse")
    train_parser.add_argument("--numerical_only", type=bool, default=True)
    train_parser.add_argument("--from_checkpoint", type=str, default=None)
    train_parser.add_argument("--top", type=int, default=30)

    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--checkpoint', type=str, required=True, default=None)
    test_parser.add_argument("--pipeline", type=str, choices=["discrete", "marginal", "rankformer"], default="marginal")
    test_parser.add_argument("--tsp_solver", type=str, choices=["greedy", "gurobi"], default="gurobi")
    test_parser.add_argument("--numerical_only", type=bool, default=True)


    test_parser.set_defaults(func=test)
    args = parser.parse_args()
    print(args)
    args.func(args)