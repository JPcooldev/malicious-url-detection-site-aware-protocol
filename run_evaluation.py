import argparse
from dotenv import load_dotenv
import os
import torch

from finetuning.utils.utils import setup_randomness
from finetuning.data_processing.dataset import load_dataset_source, DatasetOptions, DatasetSource
from finetuning.evaluation import run_evaluation
from finetuning.trainer_factory import get_evaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed for random number generator")

    # Model args
    parser.add_argument("--model_type", type=str, default="model_type",
                        help="Type of model to use")
    parser.add_argument("--model_size", type=str, default="model_size",
                        help="Size of model to use")
    parser.add_argument("--run_name_suffix", type=str, default="", 
                        help="Suffix for the run name (dataset_name/model_type-model_size-run_name_suffix)")

    # Eval params
    parser.add_argument("--decision_threshold", default=0.5, type=float, 
                        help="If probability of a class 1 is higher than this, then the sample is classified as class 1")
    parser.add_argument("--max_sequence_length", default=256, type=int, 
                        help="Maximum sequence length for the model")

    # Dataset params
    parser.add_argument("--dataset_path", type=str, default="dataset_path",
                        help="Path to the dataset (csv file containing columns: url, label, fold)")
    parser.add_argument("--eval_fold", type=int, default=0, 
                        help="Evaluation fold for evaluation")
    parser.add_argument("--eval_size_specifier", type=str, default=None,  
                        help="Size of the evaluation set (for debugging purposes)")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size")
    
    # Dataloader params
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

    return parser.parse_args([])

def env_config():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["NCCL_SHM_DISABLE"] = "1"
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA is available. Cross-validation will be run on GPU.")
    else:
        print(f"CUDA is not available. Cross-validation will be run on CPU.")

def validate_dataset_eval_fold(dataset: DatasetSource, args: argparse.Namespace):
    if args.eval_fold is None:
        print("[EVALUATION] No eval fold provided, using 0")
        args.eval_fold = 0
    if not "fold" in dataset.df.columns:
        print("[EVALUATION] No fold column in dataset, adding eval fold")
        dataset.df["fold"] = args.eval_fold
    if args.eval_fold not in dataset.df["fold"].unique():
        raise ValueError(f"Eval fold {args.eval_fold} not found in dataset's folds")


def main(args: argparse.Namespace, experiment_name: str):
    setup_randomness(args.seed)

    dataset = load_dataset_source(args.dataset_path)
    validate_dataset_eval_fold(dataset, args)

    dataset_options = DatasetOptions(
        eval_folds=[args.eval_fold],
        eval_size_specifier=args.eval_size_specifier,
        seed=args.seed
    )

    run_evaluation(
        args=args,
        dataset_source=dataset,
        fold=args.eval_fold,
        trainer_factory=get_evaluator,
        setup_randomness=setup_randomness,
        dataset_options_default=dataset_options,
        experiment_name=experiment_name,
    )

if __name__ == "__main__":
    load_dotenv(".env")
    env_config()

    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EVALUATION_EXPERIMENT_NAME")
    assert MLFLOW_EXPERIMENT_NAME != "", "MLFLOW_EVALUATION_EXPERIMENT_NAME is not set in .env file"

    args = parse_args()
    main(args, MLFLOW_EXPERIMENT_NAME)
