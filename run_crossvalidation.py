import argparse
from dotenv import load_dotenv
import os
import torch

from finetuning.utils.utils import setup_randomness
from finetuning.data_processing.dataset import load_dataset_source, DatasetOptions
from finetuning.crossvalidation import run_cross_validation
from finetuning.trainer_factory import get_trainer

from models.grambeddings.NGramSequenceTransformer import WeightInitializer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed for random number generator")

    # Model args
    parser.add_argument("--model_type", type=str, required=True, 
                        help="Type of model to use")
    parser.add_argument("--model_size", type=str, required=True, 
                        help="Size of model to use")
    parser.add_argument("--run_id", type=str, required=True, 
                        help="MLFlow run ID of the pretrained model")
    parser.add_argument("--run_name_suffix", type=str, default="", 
                        help="Suffix for the run name (dataset_name/model_type-model_size-run_name_suffix)")

    # Eval params
    parser.add_argument("--decision_threshold", default=0.5, type=float, 
                        help="If probability of a class 1 is higher than this, then the sample is classified as class 1")
    parser.add_argument("--max_sequence_length", default=256, type=int, 
                        help="Maximum sequence length for the model")

    # Dataset params
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the dataset (csv file containing columns: url, label, fold)")
    parser.add_argument("--eval_folds", type=int, nargs="+", default=[0, 1, 2], 
                        help="Evaluation folds for cross-validation")
    parser.add_argument("--train_folds", type=int, nargs="+", default=None, 
                        help="Training folds for cross-validation (if None, all folds are used)")
    parser.add_argument("--train_size_specifier", type=str, default=None,     
                        help="Size of the training set (for debugging purposes)")
    parser.add_argument("--eval_size_specifier", type=str, default=None,  
                        help="Size of the evaluation set (for debugging purposes)")

    # Training params
    parser.add_argument("--learning_rate", default=1e-5, type=float, 
                        help="AdamW learning rate for classification layer of model")
    parser.add_argument("--weight_decay", default=0.00, type=float, 
                        help="AdamW weight decay for both parts of the model")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size")
    parser.add_argument("--epochs", default=2, type=int, 
                        help="Number of epochs to train the model")
    parser.add_argument("--loss", default="focal", choices=["cross_entropy", "focal"], type=str, 
                        help="Loss function used")
    parser.add_argument("--focal_loss_gamma", default=2, 
                        help="Gamma parameter for focal loss. Otherwise has no effect")
    parser.add_argument("--focal_loss_alpha", default=0.7, 
                        help="Alpha parameter for focal loss. Otherwise has no effect")
    
    # Dataloader params
    parser.add_argument("--train_dataloader_prefetch_factor", default=None, type=int, 
                        help="Prefetch factor for training DataLoader")
    parser.add_argument("--train_dataloader_persistent_workers", default=False, action="store_true", 
                        help="Use persistent workers for training DataLoader")
    parser.add_argument("--train_dataloader_num_workers", default=0, type=int, 
                        help="Number of workers for training DataLoader")
    parser.add_argument("--eval_dataloader_prefetch_factor", default=None, type=int, 
                        help="Prefetch factor for eval DataLoader")
    parser.add_argument("--eval_dataloader_persistent_workers", default=False, action="store_true", 
                        help="Use persistent workers for eval DataLoader")
    parser.add_argument("--eval_dataloader_num_workers", default=0, type=int, 
                        help="Number of workers for eval DataLoader")
    
    # ================== CharBERT specific params ==================
    parser.add_argument("--char_vocab_path", type=str, default="./tokenizers/charbert/char_vocab.txt",
                        help="Path to character vocabulary file for CharBERT")
    parser.add_argument("--char_max_length_per_token", default=6, type=int, 
                        help="Maximum number of characters per token")

    # ================ GramBeddings specific params ================
    # Input ngram selections
    parser.add_argument("--ngram_1", type=int, default=4, 
                        help="Ngram value of first ngram embedding layer")
    parser.add_argument("--ngram_2", type=int, default=5, 
                        help="Ngram value of second ngram embedding layer")
    parser.add_argument("--ngram_3", type=int, default=6, 
                        help="Ngram value of third ngram embedding layer")
    # Feature Selection Parameters
    parser.add_argument("--max_features", type=int, default=160000, 
                        help="Maximum number of features")
    parser.add_argument("--max_df", type=float, default=0.7, 
                        help="Embedding dimension for Embedding Layer")
    parser.add_argument("--min_df", type=float, default=1e-06, 
                        help="Embedding dimension for Embedding Layer")
    parser.add_argument("--max_seq_len_grambeddings", type=int, default=128,
                        help="The maximum sequence length to trim our transformed sequences")
    parser.add_argument("--case_insensitive", type=bool, default=False, 
                        help="Whether to ignore case")
    parser.add_argument("--warm_start", type=bool, default=False,
                        help="Whether to warm-start the model")
    parser.add_argument("--warm_mode",
        type=WeightInitializer,
        default=WeightInitializer.randomly_initialize,
        choices=list(WeightInitializer),
        help="The selected Embedding Layer weight initializing method. Only matters when warm_start is set True",
    )

    parser.add_argument("--embed_dim", type=int, default=15, 
                        help="Embedding dimension for Embedding Layer")
    parser.add_argument("--attn_width", type=int, default=10, 
                        help="The attention layer width")
    parser.add_argument("--rnn_cell_size", type=int, default=128, 
                        help="The recurrent size")
    parser.add_argument("--char_embedding_dim", type=int, default=95, 
                        help="Character embedding dimension")
    parser.add_argument("--retain_random_percentage", type=float, default=0.14, 
                        help="Retain random percentage of the data")

    return parser.parse_args([])


def env_config():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["NCCL_SHM_DISABLE"] = "1"
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA is available. Cross-validation will be run on GPU.")
    else:
        print(f"CUDA is not available. Cross-validation will be run on CPU.")


def main(args: argparse.Namespace, experiment_name: str):
    setup_randomness(args.seed)

    dataset = load_dataset_source(args.dataset_path)

    dataset_options = DatasetOptions(
        eval_folds=args.eval_folds,
        train_folds=args.train_folds,
        train_size_specifier=args.train_size_specifier,
        eval_size_specifier=args.eval_size_specifier,
        seed=args.seed
    )

    run_cross_validation(
        args=args,
        dataset_source=dataset,
        folds_to_run=args.eval_folds,
        trainer_factory=get_trainer,
        setup_randomness=setup_randomness,
        dataset_options_default=dataset_options,
        experiment_name=experiment_name,
    )

if __name__ == "__main__":
    load_dotenv(".env")
    env_config()

    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_CROSSVALIDATION_EXPERIMENT_NAME")
    assert MLFLOW_EXPERIMENT_NAME != "", "MLFLOW_CROSSVALIDATION_EXPERIMENT_NAME is not set in .env file"

    args = parse_args()
    main(args, MLFLOW_EXPERIMENT_NAME)
