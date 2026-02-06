# From Lab to Production: Malicious URL Detection on Real-World Data
This codebase was used for the paper "From Lab to Production: Malicious URL Detection on Real-World Data"

In this paper we trained and evaluated several models on malicious URL detection task in smaller variants (4M to 29M parameters).
These models are:
- [BERT](https://aclanthology.org/N19-1423/?utm_campaign=The+Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_m9bbH_7ECE1h3lZ3D61TYg52rKpifVNjL4fvJ85uqggrXsWDBTB7YooFLJeNXHWqhvOyC)
- [CharBERT](https://arxiv.org/abs/2011.01513)
- [URLBERT](https://www.sciencedirect.com/science/article/abs/pii/S1389128625004803)
- [PMANet](https://www.sciencedirect.com/science/article/abs/pii/S1566253524004160)
- [GramBeddings](https://www.sciencedirect.com/science/article/abs/pii/S016740482200356X)


## Directory Guide:
```
data/
    |- pretraining_dataset/
        |- dataset_constructor.py           # downloader of URLs from CommonCrawl for pre-training dataset
        |- preprocessing.ipynb              # preprocessing of URLs downloaded by dataset_constructor.py
    |- public_dataset
        |- benign_commoncrawl/
            |- download_nonephemeral.ipynb  # code that downloads URLs that are long-lived over multiple years
            |- create_benign_dataset.ipynb  # use .csv file from above to create final csv dataset that will be used
        |- combiner.ipynb                   # notebook that combines multiple public datasets into one final
        |- create_unfolded_dataset.ipynb    # notebook for creating dataset with random split folds that has same properties as site-aware datasets (9:1 imbalance, same number of samples in each fold, ...)
    |- utils/
        |- common_crawl/
            |- common_crawl.py              # utils for interacting with commoncrawl website
            |- data_sources.json            # information about different commoncrawl groups
        |- external_sort.py                 # utility to sort files using external method (useful for large files)
        |- feature_extraction.py            # collection of useful features extractible from URL string
        |- fold_splitting.py                # code related to splitting data into folds and analyzing those folds
        |- general.py                       # miscellaneous utitilies
        |- remove_duplicates.cpp            # file for quick removal of duplicate URLs in large CSV files
    |- exploratory_analysis.ipynb           # analysis of datasets
finetuning/
    |- utils/
        |- focal_loss.py                    # focal loss implementation
        |- mlflow_utils.py                  # mlflow utils
        |- utils.py                         # utility functions
        |- metrics.py                           # metrics calculation
    |- data_processing/
        |- dataset.py                       # dataset definition for finetuning dataset
    |- trainer.py                           # base class of the trainer
    |- bert_trainer.py                      # trainer for BERT-based models
    |- charbert_trainer.py                  # trainer for CharBERT-based models
    |- grambeddings_trainer.py              # trainer for GramBeddings model
    |- crossvalidation.py                   # cross-validation implementation
    |- evaluation.py                        # evaluation implementation
    |- trainer_factory.py                   # trainer factory (get trainer or evaluator for a given model)
models/                                     # model definitions 
    |- charbert/                           
        |- configuration_charbert.py        # CharBERT configuration (compatible with HF transformers)
        |- modeling_charbert.py             # CharBERT architecture implementation 
    |- pmanet/                              
        |- configuration_pmanet.py          # PMANet configuration (compatible with HF transformers)
        |- modeling_pmanet.py               # PMANet architecture implementation
    |- urlbert/ 
        |- token_manipulation.py            # token manipulation for stage 1 pre-training of URLBERT
        |- contrastive_learning.py          # SCL for stage 2 pre-training of URLBERT
        |- virtual_adversarial_training.py  # VAT for stage 2 pre-training of URLBERT
    |- grambeddings/
        |- Model.py                         # GramBeddings architecture implementation (in tf and keras)
        |- NGramSequenceTransformer.py      # sequence processing for GramBeddings
    |- utils.py                             # utility functions for model and tokenizer loading
pretraining/                    
    |- data_processing/
        |- dataset.py           # dataset definition for pretraining dataset
        |- dataloader.py        # dataloader definition
        |- data_utils.py        # download wiki dataset for charbert pre-training
    |- utils/
        |- utils.py             # utility functions
        |- ddp_utils.py         # DDP utils
    |- trainer.py               # base class of the trainer
    |- urlbert_trainer.py       # trainer for URLBERT-based models
    |- bert_trainer.py          # trainer for BERT-based models
    |- charbert_trainer.py      # trainer for CharBERT-based models
tokenizer/
    |- wordpiece-url-vocab-15360/     # tokenizer files 
    # tokenizer trained on URL pre-training dataset, used for BERT-url and URLBERT
    |- charbert/
        |- char_vocab.txt     # character vocabulary
        |- term_vocab.txt     # term vocabulary
```

## Usage
Our codebase uses MLFlow for experiment tracking.
You need to create `.env` file in the root directory with paths to MLFlow experiments.

1. Copy the template environment file: `cp .env.example .env`
2. Update `.env` with your local paths to MLFlow experiments.

Follow these steps to set up and run the project:
```bash
# create virtual environment
python -m venv venv

# activate virtual environment
source venv/bin/activate
venv/Scripts/activate # on Windows

# install dependencies
pip install -r requirements.txt # on Unix and Mac
```

### Pre-training
```shell
torchrun --nnodes=1 --nproc_per_node=4 run_pretraining.py \
    --model_type [bert | urlbert | charbert] \
    --model_size [tiny | mini] \
    --worker_batch_size [int] \
    --dataset_path /path/to/dataset.csv \
    --dataset_path_val /path/to/dataset_val.csv \

    # these parameters neee to be set for CharBERT
    --num_steps [int] \
    --eval_steps [int] \
    --mlm_prob 0.1 \
```
We set the other parameters to default values to match the experiment settings. You can adjust these parameters. Just use `--help` or go to [run_pretraining.py](run_pretraining.py) for more details.

Pre-training script will invoke distributed training using `torch` DDP. Every `save_freq` steps, the model will be evaluated on the validation set and model checkpoint will be logged to MLFlow as a nested run of the training run along with its hyperparameters and training statistics.
The training run will be logged with name `model_type-model_size` (optionally with `run_name_suffix`). 

Example structure of the MLFlow runs:
```text
model_type-model_size (optionally -run_name_suffix)
├── checkpoint-0
├── checkpoint-1
├── checkpoint-2
└── ...
```

During the pre-training, we log these metrics:
|metric|BERT|CharBERT|URLBERT-stage 1|URLBERT-stage 2|description|
|---|---|---|---|---|---|
|train_loss|✅|✅|✅|✅|overall training loss|
|mlm_accuracy|✅|✅|✅|✅|MLM accuracy|
|lr|✅|✅|✅|-|Learning rate|
|grad_norm|✅|✅|✅|-|Gradient L2-norm|
|term_loss|-|✅|-|-|Term loss|
|term_accuracy|-|✅|-|-|Term accuracy|
|mlm_loss|-|✅|✅|✅|MLM loss|
|std_loss|-|-|✅|-|Shuffled Token Detection loss|
|rtd_loss|-|-|✅|-|Replaced Token Detection loss|
|contrastive_loss|-|-|-|✅|Self-supervised Contrastive Learning (SCL) loss|
|adv_loss|-|-|-|✅|Virtual Adversarial Training (VAT) loss|
|lr_vat|-|-|-|✅|VAT learning rate|
|lr_sscl|-|-|-|✅|SCL learning rate|
|grad_norm_vat|-|-|-|✅|VAT gradient L2-norm|
|grad_norm_sscl|-|-|-|✅|SCL gradient L2-norm|

During evaluation, only MLM accuracy and MLM loss are logged.

Due to the dynamic processing of the CharBERT dataset (adversarial character petrubations), we need to set the `num_steps` and `eval_steps`. The length of the CharBERTDatasetPretraining dataset (`__len__` funciton) is harcoded to a large value and we rely on early stopping to stop the training.

Training hyperparameters for our experiments that differ between models:
|param|BERT-url-tiny|BERT-url-mini|CharBERT-tiny|CharBERT-mini|URLBERT-tiny (stage 1)|URLBERT-mini  (stage 1)|URLBERT-tiny  (stage 2)|URLBERT-mini (stage 2)|
|---|---|---|---|---|---|---|---|---|
|num_steps|130,582|130,582|20,000|20,000|65,291|87,054|60,000|62,291|
|worker_batch_size|128|128|96|64|128|96|128|128|
|batch_size|512|512|384|256|512|384|512|512|
|learning_rate|5e-4|5e-4|5e-4|5e-4|5e-4|5e-4|-|-|
|weight_decay|0.01|0.01|0.01|0.01|0.01|0.01|-|-|
|mlm_prob|0.15|0.15|0.1|0.1|0.15|0.15|0.15|0.15|
|vat_learning_rate|-|-|-|-|-|-|1e-4|1e-4|
|vat_weight_decay|-|-|-|-|-|-|0.01|0.01|
|sscl_learning_rate|-|-|-|-|-|-|5e-4|5e-4|
|sscl_weight_decay|-|-|-|-|-|-|0.01|0.01|
|adv_prob|-|-|0.15|0.15|-|-|-|-|
|char_maxlen_for_word|-|-|6|6|-|-|-|-|

We used 4x Nvidia A10G (except URLBERT stage, there we used 4x Nvidia L40S).

### Cross-validation 
```shell
python3 run_crossvalidation.py \
    --model_type [bert-text | bert-url | urlbert | charbert | pmanet | grambeddings] \
    --model_size [tiny | mini | small] \
    --run_id [mlflow run ID] \
    --dataset_path /path/to/dataset.csv \ 
    --batch_size [int] \ # batch size for training (default: 128)
    --eval_folds [int] ... [int] \ # evaluation folds for cross-validation (default: [0, 1, 2])
```
We set the other parameters to default values to match the experiment settings. You can adjust these parameters. Just use `--help` or go to [run_crossvalidation.py](run_crossvalidation.py) for more details.

We used 1x Nvidia A10G.

This script will run `len(eval_folds)` cross-validation runs (each run evaluation is done on `eval_folds[i]` fold and training is done on all folds except `eval_folds[i]`). The runs will be logged into MLFlow as child runs of the parent run with name `dataset_name/model_type-model_size` (optionally with `run_name_suffix`). If the parent run does not exist, it will be created. Also if one cross-validation run is already logged or running, it will be skipped.

Example structure of the MLFlow runs:
```text
dataset_name/model_type-model_size (optionally -run_name_suffix)
├── fold_0
├── fold_1
├── fold_2
└── ...
```

During cross-validation, we log these metrics for each run:

**Binary classification metrics (overall)**

|metric|description|
|---|---|
eval/accuracy|validation accuracy|
eval/fn|number of false negatives|
eval/fp|number of false positives|
eval/tn|number of true negatives|
eval/tp|number of true positives|
eval/fpr|false positive rate|
eval/pr_auc|area under precision-recall curve|
eval/pr_auc_sklearn|area under precision-recall curve (sklearn)|
eval/roc_auc|area under ROC curve|

**Binary classification metrics (per class, macro avg, weighted avg)**

|metric|description|
|---|---|
eval/\<type>/f1-score|F1-score|
eval/\<type>/precision|precision|
eval/\<type>/recall|recall|
eval/\<type>/support|support|

**Model performance metrics**

|metric|description|
|---|---|
train_batch_average_loss|MLM loss|
train_batch_accuracy|MLM accuracy|
|time_n_per_s_total_with_tokenizer|number of samples per second inference + tokenizer|
|time_n_per_s_total_without_tokenizer|number of samples per second inference without tokenizer|


### Evaluation
```shell
python3 run_evaluation.py \
    --model_type [bert-text | bert-url | urlbert | charbert | pmanet | grambeddings] \
    --model_size [tiny | mini | small] \
    --dataset_path /path/to/test_dataset.csv \ 
    --eval_fold [int] \  # the test dataset need to include fold column (just one fold to evaluate on the whole test dataset, due to code compatibility. if it is not provided or fold column is not in the dataset, we will use 0 and add fold column to the dataset)
    --batch_size [int] \ # batch size for evaluation (default: 128)
```
Other parameters are set to default values. Use `--help` or go to [run_evaluation.py](run_evaluation.py) for more details.

We used 1x Nvidia A10G.

This part is strictly for evaluation on a test set (just inference). Execution will log the run with name `model_type-model_size` (optionally with `-run_name_suffix`) to a parent run with name `dataset_name` (if the parent run does not exist, it will be created). 

During evaluation, we log these metrics for each run:
**Binary classification metrics (overall)**

|metric|description|
|---|---|
test/accuracy|average accuracy|
test/fn|number of false negatives|
test/fp|number of false positives|
test/tn|number of true negatives|
test/tp|number of true positives|
test/fpr|false positive rate|

**Binary classification metrics (per class, macro avg, weighted avg)**

|metric|description|
|---|---|
test/\<type>/f1-score|F1-score|
test/\<type>/precision|precision|
test/\<type>/recall|recall|
test/\<type>/support|support|

# Issues

If there is any issue, please submit a GitHub Issue.