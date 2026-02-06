import mlflow
import os

def mlflow_setup(experiment_path: str, db_token: str, db_host: str):
    """Setup MLflow connection"""
    if db_token is not None and db_host is not None:
        os.environ['DATABRICKS_HOST'] = db_host
        os.environ['DATABRICKS_TOKEN'] = db_token
    mlflow.set_experiment(experiment_path)

def shorten_num_tokens(num_tokens: int) -> str:
    """Shorten number of tokens to string representation"""
    reductions = 0
    while num_tokens > 1000:
        num_tokens = round(num_tokens / 1000, 1)
        reductions += 1
    if reductions == 1:
        return f'{num_tokens}K'
    elif reductions == 2:
        return f'{num_tokens}M'
    elif reductions == 3:
        return f'{num_tokens}B'
    else:
        raise ValueError('unreasonable number of tokens')