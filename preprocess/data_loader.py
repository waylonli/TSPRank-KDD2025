import torch
from torch.utils.data import Dataset
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader

def load_data(df, stage, variables, batch_size=64, seed=42, num_workers=0, device='cpu', label_column='target', qid_column='qid', transform=StandardScaler()):

    if 'qid' in variables:
        variables = list(variables) if not isinstance(variables, list) else variables
        variables.remove('qid')
    if 'target' in variables:
        variables = list(variables) if not isinstance(variables, list) else variables
        variables.remove('target')
    if qid_column in variables:
        variables = list(variables) if not isinstance(variables, list) else variables
        variables.remove(qid_column)
    if label_column in variables:
        variables = list(variables) if not isinstance(variables, list) else variables
        variables.remove(label_column)
    if 'id' in variables:
        variables = list(variables) if not isinstance(variables, list) else variables
        variables.remove('id')

    ltr_data = LearningToRankDataset(df,
                                     label_column=label_column,
                                     list_id_column=qid_column,
                                     variables=variables,
                                     transform=transform,
                                     seed=seed,
                                     device=device)

    if stage == 'train':
        train_loader = DataLoader(ltr_data, batch_size=batch_size, shuffle=True, collate_fn=LearningToRankDataset.collate_fn,
                                num_workers=num_workers)
        return train_loader, ltr_data.input_dim, ltr_data.max_target
    else:
        test_loader = DataLoader(ltr_data, batch_size=batch_size, shuffle=False, collate_fn=LearningToRankDataset.collate_fn,
                             num_workers=num_workers)
        return test_loader, ltr_data.input_dim, ltr_data.max_target


class LearningToRankDataset(Dataset):
    def __init__(self, df, label_column, list_id_column, variables, transform=None, seed=None, device=None):
        # It is costly to sort before any filtering happens, but we need the groups to be together for later efficiency.
        # All later steps are expected to maintain query group order.

        df.sort_values(by=list_id_column, inplace=True)
        feat_columns = variables
        self.feat = df[feat_columns].values
        if transform is not None:
            try:
                self.feat = transform.transform(self.feat)
            except:
                self.feat = transform.fit_transform(self.feat)
        try:
            self.feat = torch.from_numpy(self.feat).float().to(device)
        except:
            self.feat = None
        self.target = torch.from_numpy(df[label_column].values).float().to(device)
        self.length = torch.from_numpy(df[list_id_column].value_counts(sort=False).values).to(device)
        self.cum_length = torch.cumsum(self.length, dim=0).to(device)

        if 'id' in df.columns:
            self.id = torch.from_numpy(df['id'].values).to(device)
        else:
            self.id = None

        if 'explicit_target' in df.columns:
            self.explicit_target = torch.from_numpy(df['explicit_target'].values).int().to(device)
        else:
            self.explicit_target = None

    def __getitem__(self, item):
        # All item features, targets and list ids are stored in a single flat array. Each list is stored back-to-back.
        # When getting a batch element (i.e. a list), we therefore need to slice the correct range in the flat array.
        # The start and end indices of each list can be inferred from the cum_length array.

        if item == 0:
            start_idx = 0
        else:
            start_idx = self.cum_length[item-1]
        end_idx = self.cum_length[item].item()

        item_dict = {
            'feat': self.feat[start_idx:end_idx] if self.feat is not None else None,
            'target': self.target[start_idx:end_idx],
            'length': self.length[item].reshape(1),
            'id': self.id[start_idx:end_idx] if self.id is not None else None,
        }


        return item_dict

    def __len__(self):
        return self.length.shape[0]

    @staticmethod
    def collate_fn(batches):
        batch_example = batches[0]
        batch = {key: torch.cat([batch_vals[key] for batch_vals in batches]) for key in batch_example.keys() if batch_example[key] is not None}
        return batch

    @property
    def input_dim(self):
        return self.feat.shape[1] if self.feat is not None else 0

    @property
    def max_target(self):
        # Used in the ordinal loss function of the RankFormer
        return self.target.max().cpu().int().item()