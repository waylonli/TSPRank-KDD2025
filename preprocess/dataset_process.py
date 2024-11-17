import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as sps


def preprocess_data(args: dict, random_seed: int = 0, fold: int = 0, n_largest: int = 30) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Preprocess the dataset and split it into training, validation, and test set
    Args:
        fold:
        dataset_name: The name of the dataset
        random_seed:  The random seed for reproducibility

    Returns:
        train_data (pd.DataFrame): The training set
        valid_data (pd.DataFrame): The validation set
        test_data (pd.DataFrame): The test set
    """

    if args.dataset.lower() == "mq2008":
        if os.path.exists('data/MQ2008/Fold{}/train_top_{}.csv'.format(fold, n_largest)):
            train_data = pd.read_csv('data/MQ2008/Fold{}/train_top_{}.csv'.format(fold, n_largest))
            valid_data = pd.read_csv('data/MQ2008/Fold{}/valid_top_{}.csv'.format(fold, n_largest))
            test_data = pd.read_csv('data/MQ2008/Fold{}/test_top_{}.csv'.format(fold, n_largest))
        else:
            train_data = pd.read_csv('data/MQ2008/Fold{}/train.txt'.format(fold), sep=' ', header=None)
            valid_data = pd.read_csv('data/MQ2008/Fold{}/vali.txt'.format(fold), sep=' ', header=None)
            test_data = pd.read_csv('data/MQ2008/Fold{}/test.txt'.format(fold), sep=' ', header=None)
            for df, name in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
                # unlimited column display
                df = df.loc[:, :47]
                df.loc[:, 1:] = df.loc[:, 1:].apply(lambda row: [el.split(':')[1] for el in row])
                df.columns = ['target'] + ['qid'] + [f'feat_{i}' for i in range(1, 47)]
                df = df.astype(float)
                df[['target', 'qid']] = df[['target', 'qid']].astype(int)
                # Keep top 30 items per qid group
                df = df.groupby('qid', group_keys=False).apply(lambda x: x.nlargest(n_largest, 'target'))
                # reverse the target in each qid group
                df['target'] = df.groupby('qid')['target'].transform(lambda x: n_largest + 1 - sps.rankdata(x, method='ordinal'))
                # shuffle the data
                df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
                # sort the data by qid
                df = df.sort_values('qid').reset_index(drop=True)
                df.to_csv(f'data/MQ2008/Fold{fold}/{name}_top_{n_largest}.csv', index=False)
            train_data = pd.read_csv('data/MQ2008/Fold{}/train_top_{}.csv'.format(fold, n_largest))
            valid_data = pd.read_csv('data/MQ2008/Fold{}/valid_top_{}.csv'.format(fold, n_largest))
            test_data = pd.read_csv('data/MQ2008/Fold{}/test_top_{}.csv'.format(fold, n_largest))
    elif args.dataset.lower() in ["otd2", "wotd"]:
        train_data = pd.read_csv(f"data/events/{args.embed}/fold{fold}/{args.dataset}_train_{n_largest}.csv")
        valid_data = pd.read_csv(f"data/events/{args.embed}/fold{fold}/{args.dataset}_valid_{n_largest}.csv")
        test_data = pd.read_csv(f"data/events/{args.embed}/fold{fold}/{args.dataset}_test_{n_largest}.csv")

        if 'Target' in train_data.columns:
            train_data = train_data.rename(columns={'Target': 'target'})
            valid_data = valid_data.rename(columns={'Target': 'target'})
            test_data = test_data.rename(columns={'Target': 'target'})
            # replace the original csv
            train_data.to_csv(f"data/events/{args.embed}/fold{fold}/{args.dataset}_train_{n_largest}.csv", index=False)
            valid_data.to_csv(f"data/events/{args.embed}/fold{fold}/{args.dataset}_valid_{n_largest}.csv", index=False)
            test_data.to_csv(f"data/events/{args.embed}/fold{fold}/{args.dataset}_test_{n_largest}.csv", index=False)
    else:
        raise NotImplementedError

    return train_data, valid_data, test_data


def check_labels(data: pd.DataFrame, label_column: str) -> bool:
    """
    Check if the label column contains any missing values
    Args:
        data (pd.DataFrame): The dataset
        label_column (str): The name of the label column

    Returns:
        list: A list of qids with missing labels
    """
    # should check if the label range from 1 to n with step 1 where n is the number of classes
    data = data.groupby('qid')
    missing_labels = []
    for qid, group in data:
        if group[label_column].min() != 1 or group[label_column].max() != len(group[label_column].unique()):
            missing_labels.append(qid)
    return missing_labels
