import argparse
from dotenv import load_dotenv
import os

from pretraining.data_processing.data_utils import load_dataset
from models.utils import init_model_pretraining, load_tokenizer
from pretraining.urlbert_trainer import URLBERTTrainer
from pretraining.bert_trainer import BERTTrainer
from pretraining.charbert_trainer import CharBERTTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--random_seed", type=int, default=42, 
                        help="Random seed for weight initialization")

    # Model parameters
    parser.add_argument("--model_type", type=str, choices=["bert", "urlbert", "charbert"], 
                        required=True, help="Model type")
    parser.add_argument("--model_size", type=str, choices=["tiny", "mini", "small"], 
                        required=True, help="Model size")
    parser.add_argument("--run_name_postfix", type=str, default="", 
                        help="Run name postfix (for MLFlow logging)")

    # Training parameters
    parser.add_argument("--num_steps", type=int, default=-1, 
                        help="Number of training steps (if -1, then all data is used)")
    parser.add_argument("--num_epochs", type=int, default=1, 
                        help="Number of epochs")
    parser.add_argument("--worker_batch_size", type=int, required=True,
                        help="Batch size per worker (per GPU)")
    parser.add_argument("--max_sequence_length", default=256, type=int, 
                        help="Maximum sequence length for the model")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--clip_gradients", type=bool, default=True,
                        help="Whether to clip gradients")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm")
    parser.add_argument("--mlm_prob", type=float, default=0.15,
                        help="Masked language modeling probability")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the dataset (csv file containing column `url`")
    parser.add_argument("--dataset_path_val", type=str, required=True, 
                        help="Path to the validation dataset (csv file containing column `url`")
    
    parser.add_argument("--saving_freq", type=int, default=10_000, 
                        help="Number of steps to save the model checkpoint and evaluate")
    
    # URLBERT specific parameters
    parser.add_argument("--stage", type=int, choices=[1, 2], 
                        help="Pre-training stage for URLBERT (1 or 2)")
    parser.add_argument("--rtd_coef", type=float, default=1.0,
                        help="RTD loss coefficient for URLBERT")
    parser.add_argument("--std_coef", type=float, default=1.0,
                        help="STD loss coefficient for URLBERT")
    parser.add_argument("--vat_coef", type=float, default=10.0,
                        help="VAT loss coefficient for URLBERT")
    parser.add_argument("--sscl_coef", type=float, default=1.0,
                        help="SSCL loss coefficient for URLBERT")
    parser.add_argument("--vat_learning_rate", type=float, default=1e-4,
                        help="VAT learning rate for URLBERT")
    parser.add_argument("--vat_weight_decay", type=float, default=0.01,
                        help="VAT weight decay for URLBERT")
    parser.add_argument("--sscl_learning_rate", type=float, default=5e-4,
                        help="SSCL learning rate for URLBERT")
    parser.add_argument("--sscl_weight_decay", type=float, default=0.01,
                        help="SSCL weight decay for URLBERT")
    
    # CharBERT specific parameters
    parser.add_argument("--char_vocab_path", type=str, default="./tokenizers/charbert/char_vocab.txt",
                        help="Path to character vocabulary file for CharBERT")
    parser.add_argument("--term_vocab_path", type=str, default="./tokenizers/charbert/term_vocab.txt",
                        help="Path to term vocabulary file for CharBERT")
    parser.add_argument("--char_vocab_size", type=int, default=1001,
                        help="Character vocabulary size for CharBERT")
    parser.add_argument("--term_vocab_size", type=int, default=30_000,
                        help="Term vocabulary size for CharBERT")
    parser.add_argument("--char_embedding_size", type=int, default=128,
                        help="Character embedding size for CharBERT")
    parser.add_argument("--adv_prob", type=float, default=0.15,
                        help="Adversarial probability for CharBERT")
    parser.add_argument("--char_maxlen_for_word", type=int, default=6,
                        help="Maximum character length for word in CharBERT")
    parser.add_argument("--eval_steps", type=int, 
                        help="Evaluation steps for CharBERT")

    # ================== Databricks specific params ==================
    # only for notebook on databricks
    parser.add_argument("--db_token", type=str, default=None,
                        help="Databricks token")
    parser.add_argument("--db_host", type=str, default=None,
                        help="Databricks host")
    return parser.parse_args([])

def main(args: argparse.Namespace):
    df_train = load_dataset(args.dataset_path)
    df_val = load_dataset(args.dataset_path_val)
    tokenizer = load_tokenizer(args.model_type)

    if args.model_type == "urlbert":
        trainer = URLBERTTrainer(args)
        model = init_model_pretraining(
            args.model_type, 
            args.model_size, 
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=args.max_sequence_length, 
        )
    elif args.model_type == "bert":
        trainer = BERTTrainer(args)
        model = init_model_pretraining(
            args.model_type, 
            args.model_size, 
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=args.max_sequence_length, 
        )
    elif args.model_type == "charbert":
        trainer = CharBERTTrainer(args)
        model = init_model_pretraining(
            args.model_type, 
            args.model_size, 
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=args.max_sequence_length, 
            char_vocab_size=args.char_vocab_size,
            term_vocab_size=args.term_vocab_size,
            char_embedding_size=args.char_embedding_size,
        )
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    
    trainer.train(model, tokenizer, df_train, df_val)
    

if __name__ == "__main__":
    load_dotenv(".env")

    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_PRETRAINING_EXPERIMENT_NAME")
    assert MLFLOW_EXPERIMENT_NAME != "", "MLFLOW_PRETRAINING_EXPERIMENT_NAME is not set in .env file"

    args = parse_args()
    args.experiment_path = MLFLOW_EXPERIMENT_NAME
    main(args)