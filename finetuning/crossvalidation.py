from typing import List, Dict, Callable
import copy

import mlflow
import pandas as pd
import numpy as np

from .utils.mlflow_utils import mlflow_find_existing_parent_run, mlflow_check_child_run_exists
from .data_processing.dataset import DatasetOptions, DatasetSource


def get_parent_run_name(dataset_name: str, model_name: str) -> str:
    """Generate parent run name for dataset/model combination"""
    return f"{dataset_name}/{model_name}"


def run_single_fold(
    args,
    dataset_source: DatasetSource,
    fold: int,
    parent_run_id: str,
    trainer_factory,
    experiment_name: str,
    model_name: str,
    setup_randomness,
    dataset_options_default: DatasetOptions = None,
):
    if mlflow_check_child_run_exists(parent_run_id=parent_run_id, fold=fold, experiment_name=experiment_name):
        print(f"[CV] Fold {fold} already exists for {dataset_source.source_name}/{model_name}. Skipping.")
        return

    print(f"\n[CV] Running fold {fold} for {dataset_source.source_name}/{model_name}")

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

    # if not is_local_development():
    #     mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"fold_{fold}", tags={"mlflow.parentRunId": parent_run_id}) as child_run:
        # Log fold-specific parameters
        mlflow.log_param("eval_fold", fold)
        mlflow.log_param("fold_seed", fold_seed)

        # Create and run trainer
        trainer = trainer_factory(args=args, dataset=dataset, is_evaluation=False)
        trainer.run_training_and_evaluation()

        print(f"[CV] Completed fold {fold}, run_id: {child_run.info.run_id}")


def run_cross_validation(
    args,
    dataset_source: DatasetSource,
    folds_to_run: List[int],
    trainer_factory: Callable,
    setup_randomness: Callable,
    experiment_name: str,
    dataset_options_default: DatasetOptions = None,
):
    model_name = f"{args.model_type}-{args.model_size}"

    mlflow.set_experiment(experiment_name)
    parent_run_name = get_parent_run_name(dataset_name=dataset_source.source_name, model_name=model_name)
    if args.run_name_suffix is not None and args.run_name_suffix != "":
        parent_run_name += f"-{args.run_name_suffix}"
    parent_run_id = mlflow_find_existing_parent_run(parent_run_name=parent_run_name, experiment_name=experiment_name)
    if parent_run_id:
        print(f"[CV] Found existing parent run for {dataset_source.source_name}/{model_name}: {parent_run_id}")
    else:
        print(f"[CV] Creating new parent run for {dataset_source.source_name}/{model_name}")
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset_name", dataset_source.source_name)
            parent_run_id = parent_run.info.run_id

    # Run all folds as separate runs with parent tag
    for fold in folds_to_run:
        run_single_fold(
            args=args,
            dataset_source=dataset_source,
            fold=fold,
            parent_run_id=parent_run_id,
            experiment_name=experiment_name,
            trainer_factory=trainer_factory,
            setup_randomness=setup_randomness,
            dataset_options_default=dataset_options_default,
            model_name=model_name,
        )


def compute_crossvalidation_metrics(
    metrics_to_compute: List[str],
    dataset_name: str,
    model_name: str,
    experiment_name: str,
) -> Dict[str, Dict[str, float]]:
    parent_run_name = get_parent_run_name(dataset_name=dataset_name, model_name=model_name)
    parent_run_id = mlflow_find_existing_parent_run(parent_run_name=parent_run_name, experiment_name=experiment_name)

    if not parent_run_id:
        print(f"[CV Metrics] No parent run found for {dataset_name}/{model_name}")
        return {}

    experiment = mlflow.get_experiment_by_name(experiment_name)

    child_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
    )

    if child_runs.empty:
        print(f"[CV Metrics] No child runs found for {dataset_name}/{model_name}")
        return {}

    print(f"[CV Metrics] Found {len(child_runs)} fold runs for {dataset_name}/{model_name}")

    results = {}

    for metric_name in metrics_to_compute:
        metric_values = []
        mlflow_metric_name = f"metrics.{metric_name}"

        for _, run in child_runs.iterrows():
            if mlflow_metric_name in run:
                value = run[mlflow_metric_name]
                if pd.notna(value):
                    metric_values.append(value)

        if metric_values:
            results[metric_name] = {
                "mean": np.mean(metric_values),
                "std": np.std(metric_values),
                "values": metric_values,
                "num_folds": len(metric_values),
            }
        else:
            print(f"[CV Metrics] Warning: No values found for metric {metric_name}")

    return results


def print_cv_results(results: Dict[str, Dict[str, float]], dataset_name: str, model_name: str):
    """Pretty print cross-validation results"""
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results for {dataset_name}/{model_name}")
    print(f"{'='*60}")

    for metric_name, stats in results.items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Folds: {stats['num_folds']}")
        print(f"  Values: {[f'{v:.4f}' for v in stats['values']]}")