# [KDD'25] TSPRank: Bridging Pairwise and Listwise Methods with a Bilinear Travelling Salesman Model

[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=2405.10800&color=red&logo=arxiv)](https://arxiv.org/abs/2411.12064) (including complete result tables)

### Citation

```
@misc{li2024tsprankbridgingpairwiselistwise,
      title={TSPRank: Bridging Pairwise and Listwise Methods with a Bilinear Travelling Salesman Model}, 
      author={Weixian Waylon Li and Yftah Ziser and Yifei Xie and Shay B. Cohen and Tiejun Ma},
      year={2024},
      eprint={2411.12064},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2411.12064}, 
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