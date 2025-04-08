# [KDD'25] TSPRank: Bridging Pairwise and Listwise Methods with a Bilinear Travelling Salesman Model

[![ACM](https://img.shields.io/static/v1?label=ACM&message=10.1145/3690624.3709234&color=blue&logo=acm)](https://dl.acm.org/doi/10.1145/3690624.3709234)
[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=2411.12064&color=red&logo=arxiv)](https://arxiv.org/abs/2411.12064) (including complete result tables)

Weixian Waylon Li, Yftah Ziser, Yifei Xie, Shay B. Cohen, and Tiejun Ma. 2025. TSPRank: Bridging Pairwise and Listwise Methods with a Bilinear Travelling Salesman Model. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD '25). Association for Computing Machinery, New York, NY, USA, 707–718. https://doi.org/10.1145/3690624.3709234

### Citation

```
@inproceedings{10.1145/3690624.3709234,
author = {Li, Weixian Waylon and Ziser, Yftah and Xie, Yifei and Cohen, Shay B. and Ma, Tiejun},
title = {TSPRank: Bridging Pairwise and Listwise Methods with a Bilinear Travelling Salesman Model},
year = {2025},
isbn = {9798400712456},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3690624.3709234},
doi = {10.1145/3690624.3709234},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1},
pages = {707–718},
numpages = {12},
keywords = {learning-to-rank, pairwise-listwise ranking, travelling salesman problem},
location = {Toronto ON, Canada},
series = {KDD '25}
}
```

## 1. Environment Setup

- Create conda environment
```conda create -n tsprank python=3.10```
- Activate conda environment
```conda activate tsprank```
- Install required packages
```pip install -r requirements.txt```

If PyTorch does not install properly, please refer to the official PyTorch installation guide: https://pytorch.org/get-started/locally/

## 2. Download Data and Pretrained Models

Download the preprocessed data and pretrained models from the following link: 
- Data: https://bollin.inf.ed.ac.uk/public/direct/tsprank/data.zip (15.5GB)
- Pretrained model checkpoints: https://bollin.inf.ed.ac.uk/public/direct/tsprank/checkpoints.zip (13.6GB)

After downloaded, extract the "data" and "checkpoints" folders into the project's root directory for use.

## 3. Reproduce Results

To reproduce the results using the pretrained models, please follow the instructions in this section.
Note that if you want to do inference on __more than 30__ ranking entities, you will need a Gurobi license.
Details can be found at https://www.gurobi.com/academia/academic-program-and-licenses/.

After obtaining the license, please fill in the value of `WLSACCESSID`, `WLSSECRET`, and `LICENSEID` in `.env` file with your own license information.

### 3.1 Stock Ranking

- Navigate to the root directory of the project.
- Reproduce the results for NASDAQ dataset:

```bash
bash scripts/stock/run_baseline.sh test NASDAQ
bash scripts/stock/run_rankformer.sh test NASDAQ
bash scripts/stock/run_tsprank.sh test local NASDAQ
bash scripts/stock/run_tsprank.sh test global NASDAQ
```
- Reproduce the results for NYSE dataset:

```bash
bash scripts/stock/run_baseline.sh test NYSE
bash scripts/stock/run_rankformer.sh test NYSE
bash scripts/stock/run_tsprank.sh test local NYSE
bash scripts/stock/run_tsprank.sh test global NYSE
```
### 3.2 Retrieval (MQ2008)

- Navigate to the root directory of the project. 
- Reproduce the results for MQ2008 dataset:

```bash
export TOPK=10 # define the number of documents to rank, can be 10 and 30
bash scripts/mq2008/run_rankformer.sh test $TOPK
bash scripts/mq2008/run_tsprank.sh test local $TOPK
bash scripts/mq2008/run_tsprank.sh test global $TOPK
```
### 3.3 Historical Events Ordering

- Navigate to the root directory of the project.
- Reproduce the results for the OTD2 dataset:

```bash
export TOPK=10 # define the number of events in a ranking group, can be 10, 30, and 50
bash scripts/events/run_rankformer.sh test $TOPK
bash scripts/events/run_tsprank.sh test local $TOPK
bash scripts/events/run_tsprank.sh test global $TOPK
```

## 4. Train Models from Scratch

To train the models from scratch, please follow the instructions in this section.

- Navigate to the root directory of the project.
- Simply follow the aforementioned instructions in Section 3.1, 3.2, and 3.3, but replace `test` with `train`, and replace `global` with `hybrid` for the `learning` argument.
- To customise the hyperparameters, please modify the corresponding arguments in the `.sh` scripts.
- The checkpoints will be saved in the `checkpoints` directory.

Some details about the hyperparameters:

| Hyperparameter  | Description                                          |
|-----------------|------------------------------------------------------|
| `--epochs`      | Number of epochs to train the model                  |
| `--eval_freq`     | Frequency of evaluation during training              |
| `--tf_num_layers` | Number of transformer layers                         |
| `--tf_nheads`     | Number of attention heads                            |
| `--lr`            | Learning rate                                        |
| `--weight_decay`  | Weight decay                                         |
| `--tf_dim_ff`     | Feedforward dimension in the transformer             |
| `--dim_emb`       | Dimension of the embedding layer                     |
| `--eval_metric`   | Evaluation metric for determining the best model     |
| `--batch_size`    | Batch size                                           |
| `--optimizer`     | Optimizer (can be `adam` or `sgd`)                   |
| `--learning`      | Learning rate scheduler (can be `local`, `global`, `hybrid`) |
| `--tsp_solver`    | TSP solver (can be `greedy` or `gurobi`)             |

Note that the best performer, TSPRank-Global, is trained using the `hybrid` learning method.
The pretrained weights using the name `global` are also trained using the `hybrid` learning method.