from typing import Callable
import copy
import argparse

import mlflow

from .data_processing.dataset import DatasetOptions, DatasetSource
from .utils.mlflow_utils import mlflow_find_existing_parent_run


def evaluation(
    args: argparse.Namespace,
    dataset_source: DatasetSource,
    fold: int,
    parent_run_id: str,
    trainer_factory: Callable,
    experiment_name: str,
    model_name: str,
    setup_randomness: Callable,
    dataset_options_default: DatasetOptions = None,
):
    """
    Evaluation run for a single dataset fold. This function inkokes Trainer.run_evaluation() method.
    Run is saved into MLFlow as child run (name=model_name) of a parent run (name=dataset_name).
    """
    print(f"\n[EVALUATION] Running evaluation for {dataset_source.source_name}/{model_name}")

    fold_seed = fold
    setup_randomness(fold_seed)
    args.seed = fold_seed

    # Create dataset with specific fold for evaluation
    if dataset_options_default is None:
        dataset_options = DatasetOptions(
            eval_folds=[fold], train_size_specifier=None, eval_size_specifier=None, seed=fold_seed
        )
    else:
        # Create a deep copy to avoid modifying the original
        dataset_options = copy.deepcopy(dataset_options_default)
        dataset_options.eval_folds = [fold]
        dataset_options.seed = fold_seed

    dataset = dataset_source.get_dataset(dataset_options)

    mlflow.set_experiment(experiment_name)
    run_name = f"{model_name}"
    if args.run_name_suffix is not None and args.run_name_suffix != "":
        run_name += f"-{args.run_name_suffix}"
    with mlflow.start_run(run_name=run_name, tags={"mlflow.parentRunId": parent_run_id}) as child_run:
        # Log fold-specific parameters
        mlflow.log_param("eval_fold", fold)
        mlflow.log_param("fold_seed", fold_seed)

        # Create and run trainer
        trainer = trainer_factory(args=args, dataset=dataset, is_evaluation=True)
        trainer.run_evaluation()

        print(f"[EVALUATION] Completed evaluation for {model_name}, run_id: {child_run.info.run_id}")


def run_evaluation(
    args: argparse.Namespace,
    dataset_source: DatasetSource,
    fold: int,
    trainer_factory: Callable,
    setup_randomness: Callable,
    experiment_name: str,
    dataset_options_default: DatasetOptions = None,
):
    """
    Run evaluation for a single dataset fold. No training is performed.
    Run is saved into MLFlow as child run (name=model_name) of a parent run (name=dataset_name).
    Parent run is created if it doesn't exist.
    """
    model_name = f"{args.model_type}-{args.model_size}"
    mlflow.set_experiment(experiment_name)
    parent_run_name = dataset_source.source_name
    parent_run_id = mlflow_find_existing_parent_run(parent_run_name=parent_run_name, experiment_name=experiment_name)
    if parent_run_id:
        print(f"[EVALUATION] Found existing parent run for {dataset_source.source_name}: {parent_run_id}")
    else:
        print(f"[EVALUATION] Creating new parent run for {dataset_source.source_name}")
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            mlflow.log_param("dataset_name", dataset_source.source_name)
            parent_run_id = parent_run.info.run_id
    
    evaluation(
        args=args,
        dataset_source=dataset_source,
        fold=fold,
        parent_run_id=parent_run_id,
        experiment_name=experiment_name,
        trainer_factory=trainer_factory,
        model_name=model_name,
        setup_randomness=setup_randomness,
        dataset_options_default=dataset_options_default,
    )