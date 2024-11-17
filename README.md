# TSPRank

## 1. Environment Setup

- Create conda environment
```conda create -n tsprank python=3.10```
- Activate conda environment
```conda activate tsprank```
- Install required packages
```pip install -r requirements.txt```

If PyTorch does not install properly, please refer to the official PyTorch installation guide: https://pytorch.org/get-started/locally/

## 2. Download Data and Pretrained Models

Download the demo data and pretrained models for MQ2008-list from the following link: 
- Demo data: https://www.dropbox.com/scl/fi/92zt2d8gnwx27p867uzx5/demo_data.zip?rlkey=fbpawm4esxvk7ip2kac3nsrqy&st=4tzs6jfb&dl=0
- Pretrained model checkpoints: https://www.dropbox.com/scl/fi/q5wedw0wwikrnjt4py3du/demo_checkpoints.zip?rlkey=dud31gx2nktzgzkoi7otdvkje&st=4lknnc8a&dl=0

Due to the anonymous review process, we are unable to provide the complete data and models with the size limitation of free cloud storage services.
__<u>We will provide the COMPLETE data and models upon acceptance.</u>__

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