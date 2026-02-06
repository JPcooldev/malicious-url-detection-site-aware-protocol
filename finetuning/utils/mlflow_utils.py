from typing import Dict, Any
import mlflow
import json
import matplotlib.pyplot as plt

from .metrics import EvaluationResults, create_confusion_matrix_plot, create_pr_curve_plot, create_roc_curve_plot

def log_evaluation_result_to_mlflow(evaluation_result: EvaluationResults) -> None:
    """Log evaluation result to MLflow"""
    mlflow.log_dict(evaluation_result.metrics, artifact_file="best_model_metrics.json")
    mlflow.log_dict(
        evaluation_result.alternative_metrics,
        artifact_file="best_model_metrics_with_best_decision_threshold.json",
    )

    fig = create_confusion_matrix_plot(evaluation_result.labels, evaluation_result.predictions)
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    class_1_probabilities = evaluation_result.probabilities[:, 1]

    fig = create_roc_curve_plot(evaluation_result.labels, class_1_probabilities)
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)

    fig = create_pr_curve_plot(evaluation_result.labels, class_1_probabilities)
    mlflow.log_figure(fig, "precision_recall_curve.png")
    plt.close(fig)


def log_metrics_dict_recursively(d, prefix: str = "", step: int | None = None, synchronous: bool = True) -> None:
    """
    Recursively flattens nested dict `d` into metric keys, and logs them in one batched call.
    Non-numeric values are saved as a JSON artifact.
    """
    scalars, other = {}, {}
    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, path + [str(k)])
        else:
            key = "/".join([p for p in ([prefix] + path) if p])
            try:
                scalars[key] = float(x)
            except Exception:
                other[key] = x

    walk(d, [])
    if scalars:
        if step is not None:
            mlflow.log_metrics(scalars, step=step, synchronous=synchronous)
        else:
            mlflow.log_metrics(scalars, synchronous=synchronous)

    if other:
        suffix = f"_{step}" if step is not None else ""
        mlflow.log_dict(other, artifact_file=f"{prefix or 'metrics'}_non_scalars{suffix}.json")



def log_dict_json(dict: Dict[str, Any], path: str) -> None:
    """Log dictionary to JSON file"""
    print("[logger] logging json dict")
    mlflow.log_dict(json.loads(json.dumps(dict, default=float)), artifact_file=path)


def mlflow_find_existing_parent_run(parent_run_name: str, experiment_name: str) -> str | None:
    """
    Find existing MLFlow parent run by name. 
    If run is found, return its run_id. If not, return None.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{parent_run_name}'",
        max_results=1,
    )
    if not runs.empty:
        return runs.iloc[0].run_id
    return None


def mlflow_check_child_run_exists(parent_run_id: str, fold: int, experiment_name: str) -> bool:
    """
    Check if child run exists for given parent run and fold (used in cross-validation).
    If run is found, return True. If not, return False.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}' and params.eval_fold = '{fold}'",
    )
    return not runs.empty