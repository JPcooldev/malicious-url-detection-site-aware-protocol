import os
from typing import List, Optional

import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import mlflow


class URLDatasetPart(Dataset):
    """Training, validation, testing part of the dataset"""

    def __init__(self, urls, y):
        self.urls = list(urls) if not isinstance(urls, list) else urls
        self.labels = list(y) if not isinstance(y, list) else y

        assert len(self.urls) == len(self.labels), "Mismatched lengths in dataset part"

    @classmethod
    def from_pandas(cls, df):
        return cls(
            urls=df["url"],
            y=df["label"],
        )

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, i):
        return self.urls[i], self.labels[i]


class DatasetOptions:
    """Configuration and validation for dataset splitting options"""

    def __init__(
        self,
        eval_folds: List,
        seed: int,
        train_folds: Optional[List] = None,
        train_size_specifier: Optional[str] = None,
        eval_size_specifier: Optional[str] = None,
    ):
        self.train_folds = train_folds
        self.eval_folds = eval_folds
        self.train_size_specifier = train_size_specifier
        self.eval_size_specifier = eval_size_specifier
        self.seed = seed

        self._validate()

    def _validate(self):
        """Validate configuration parameters"""
        # Validate eval_folds is not None/empty
        if not self.eval_folds:
            raise ValueError("eval_folds cannot be None or empty")

        # Validate fold overlap
        if self.train_folds is not None:
            train_set = set(self.train_folds)
            eval_set = set(self.eval_folds)
            if not train_set.isdisjoint(eval_set):
                overlap = train_set.intersection(eval_set)
                raise ValueError(f"train_folds and eval_folds must not overlap. Found overlap: {overlap}")

        # Validate size specifiers
        if self.train_size_specifier is not None:
            self._validate_size_specifier(self.train_size_specifier, "train_size_specifier")

        if self.eval_size_specifier is not None:
            self._validate_size_specifier(self.eval_size_specifier, "eval_size_specifier")

    def _validate_size_specifier(self, specifier: str, param_name: str):
        """Validate size specifier format (must end with '%' or 'u')"""
        if not isinstance(specifier, str) or len(specifier) < 2:
            raise ValueError(f"{param_name} must be a string with at least 2 characters")

        if specifier[-1] not in ["%", "u"]:
            raise ValueError(f"{param_name} must end with '%' or 'u'")

        try:
            float(specifier[:-1])
        except ValueError:
            raise ValueError(f"{param_name} must have a valid number before '%' or 'u'")

    def validate_against_dataframe(self, df: pd.DataFrame):
        """Validate options against actual dataframe folds"""
        if not {"url", "fold", "label"}.issubset(df.columns):
            raise ValueError("DataFrame must include 'url', 'fold', 'label' columns")

        all_folds = set(df["fold"].unique())
        all_folds_clean = {f for f in all_folds if pd.notna(f)}

        # Validate eval_folds exist in dataframe
        eval_folds_clean = {f for f in self.eval_folds if pd.notna(f)}
        if not eval_folds_clean.issubset(all_folds_clean):
            missing = eval_folds_clean - all_folds_clean
            raise ValueError(f"eval_folds {missing} not found in dataframe folds {all_folds_clean}")

        # Validate train_folds exist in dataframe (if specified)
        if self.train_folds is not None:
            train_folds_clean = {f for f in self.train_folds if pd.notna(f)}
            if not train_folds_clean.issubset(all_folds_clean):
                missing = train_folds_clean - all_folds_clean
                raise ValueError(f"train_folds {missing} not found in dataframe folds {all_folds_clean}")

    def get_train_folds(self, all_folds: List) -> List:
        """Get train folds, auto-generating if not specified"""
        if self.train_folds is None:
            return [f for f in all_folds if f not in self.eval_folds]
        return self.train_folds


class URLDataset:
    """Main dataset class that handles splitting logic"""

    train_set: URLDatasetPart
    eval_set: URLDatasetPart
    options: DatasetOptions

    def __init__(self, df: pd.DataFrame, options: DatasetOptions, source_name: str):
        self.df = df
        self.options = options
        self.source_name = source_name

        # Validate options against dataframe
        self.options.validate_against_dataframe(self.df)

        # Perform the split
        self.train_set, self.eval_set = self._split_dataset()

    def _split_dataset(self):
        """Split dataset according to options"""
        all_folds = sorted(self.df["fold"].unique())
        print(f"[dataset]: All folds {all_folds}")

        train_folds = self.options.get_train_folds(all_folds)
        eval_folds = self.options.eval_folds

        print(f"[dataset]: Train folds {train_folds}, Eval folds {eval_folds}")

        # Split by folds
        df_train = self._retain_only_folds(self.df, train_folds, self.options.train_size_specifier)
        df_eval = self._retain_only_folds(self.df, eval_folds, self.options.eval_size_specifier)

        print(f"[dataset]: train length: {len(df_train)}, eval_length: {len(df_eval)}")

        return URLDatasetPart.from_pandas(df_train), URLDatasetPart.from_pandas(df_eval)

    def log_to_mlflow(self, should_save: bool = False):
        """Log both train and eval dataset parts to MLflow"""
        # Log train set
        mlflow.log_param("train_folds", self.options.train_folds)
        mlflow.log_param("eval_folds", self.options.eval_folds)
        mlflow.log_param("train_size_specifier", self.options.train_size_specifier)
        mlflow.log_param("eval_size_specifier", self.options.eval_size_specifier)
        train_df = pd.DataFrame({"url": self.train_set.urls, "label": self.train_set.labels})
        mlflow.log_input(
            mlflow.data.from_pandas(train_df, name="train", source=self.source_name),
            context=self.source_name,
        )
        if should_save:
            mlflow.log_text(train_df.to_csv(index=False), artifact_file="train.csv")

        # Log eval set
        eval_df = pd.DataFrame({"url": self.eval_set.urls, "label": self.eval_set.labels})
        mlflow.log_input(
            mlflow.data.from_pandas(eval_df, name="eval", source=self.source_name),
            context=self.source_name,
        )
        if should_save:
            mlflow.log_text(eval_df.to_csv(index=False), artifact_file="eval.csv")

    def _retain_only_folds(self, df: pd.DataFrame, folds: List, shorten_string: Optional[str]):
        """Retain only specified folds and optionally shorten"""
        if folds is not None:
            df_filtered = df.loc[df["fold"].isin(folds)]
        else:
            print("[dataset] Folds are none, no selection based on folds is applied")
            df_filtered = df

        shorten_num = self._convert_shortening_units_to_num(shorten_string, len(df_filtered))
        if shorten_num:
            print(f"[dataset] Shortening applied to df of length: {len(df_filtered)} by ({shorten_string})")
            df_filtered = self._shorten_df_in_smart_way(df_filtered, shorten_num)
            print(f"[dataset] New length is {len(df_filtered)}")

        return df_filtered

    def _shorten_df_in_smart_way(self, df: pd.DataFrame, target_size: int):
        folds = df["fold"].unique()
        base_quota = target_size // len(folds)
        remainder = target_size % len(folds)
        parts = []

        for i, fold in enumerate(folds):
            # Distribute remainder among first few folds
            quota = base_quota + (1 if i < remainder else 0)

            if pd.isna(fold):
                df_fold = df[df["fold"].isna()]
            else:
                df_fold = df[df["fold"] == fold]

            stratify_col = df_fold["label"] if df_fold["label"].nunique() > 1 else None
            try:
                sampled, _ = train_test_split(
                    df_fold,
                    train_size=quota,
                    stratify=stratify_col,
                    random_state=self.options.seed,
                )
            except ValueError:
                sampled = df_fold.sample(quota, replace=len(df_fold) < quota, random_state=self.options.seed)

            parts.append(sampled)

        return pd.concat(parts, ignore_index=True).reset_index(drop=True)

    def _convert_shortening_units_to_num(self, units_str: Optional[str], data_size: int) -> Optional[int]:
        """Units must either end with '%' or 'u'"""
        if units_str is None:
            return None
        if units_str[-1] == "%":
            return int(float(units_str[:-1]) / 100 * data_size)
        elif units_str[-1] == "u":
            return int(units_str[:-1])
        else:
            raise ValueError("Invalid dataset shortening units")


class DatasetSource:

    source_name: str

    def __init__(self, df: pd.DataFrame, source_name: str):
        self.source_name = source_name
        self.df = df

    def get_dataset(self, options: DatasetOptions) -> URLDataset:
        """Get URLDataset with specified options"""
        return URLDataset(self.df, options, source_name=self.source_name)


def load_dataset_source(dataset_path: str) -> DatasetSource:
    """Load dataset from path"""
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist"
    df = pd.read_csv(dataset_path)
    
    required_columns = ["url", "label", "fold"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    assert not missing_columns, f"Dataset is missing required columns: {missing_columns}. Found columns: {list(df.columns)}"
    
    df["label"] = df["label"].replace({"benign": 0, "malicious": 1})
    return DatasetSource(df, dataset_path.split("/")[-1])
