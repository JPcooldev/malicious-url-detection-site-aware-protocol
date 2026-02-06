import time
import numpy as np
import os
import gc
import mlflow

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy

# Import Grambeddings components (no changes to original files)
from models.grambeddings.Model import NBeddingModel
from models.grambeddings.NGramSequenceTransformer import (
    NBeddingTransformer,
    CharacterLevelTransformer,
)

from .data_processing.dataset import URLDataset
from .utils.metrics import compute_metrics_default, EvaluationResults
from .utils.mlflow_utils import log_evaluation_result_to_mlflow, log_metrics_dict_recursively


LABEL_PHISH = 1
LABEL_LEGIT = 0


def tf_synchronized_time():
    """Ensure GPU operations complete before measuring time in TF2 eager mode."""
    if tf.config.list_physical_devices("GPU"):
        # Force a tiny tensor onto GPU and back, synchronizing the stream
        _ = tf.convert_to_tensor(0.0).numpy()
    return time.perf_counter()


class BatchDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for batch processing to avoid loading all transformed data at once"""
    
    def __init__(self, samples, labels, transformers, batch_size, max_seq_len, shuffle=True):
        self.samples = samples
        self.labels = labels
        self.transformers = transformers  # Dictionary with transformer objects
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.samples))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get batch samples
        batch_samples = [self.samples[i][:self.max_seq_len] for i in batch_indices]
        batch_labels = self.labels[batch_indices].reshape(-1, 1)
        
        # Transform on-the-fly for this batch only
        batch_x = []
        
        # Character transformation
        char_sequences = self.transformers['char'].Transform(batch_samples)
        batch_x.append(char_sequences)
        
        # N-gram transformations
        for key in ['ngram_1', 'ngram_2', 'ngram_3']:
            if key in self.transformers:
                ngram_sequences = self.transformers[key].Transform(batch_samples)
                batch_x.append(np.array(ngram_sequences, dtype='float32'))
        
        return batch_x, batch_labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class MLflowMetricsCallback(Callback):
    """Custom Keras callback to log detailed metrics to MLflow after each epoch"""

    def __init__(self, validation_generator, log_train_loss_every=50):
        super().__init__()
        self.validation_generator = validation_generator
        self.step_count = 0
        self.log_train_loss_every = log_train_loss_every
        self.epoch_count = 0

    def on_batch_end(self, batch, logs=None):
        """Called at the end of each training batch"""
        if logs is None:
            logs = {}

        self.step_count += 1

        # Log training loss every N steps
        if self.step_count % self.log_train_loss_every == 0:
            train_loss = logs.get("loss", 0.0)
            global_step = self.epoch_count * self.params["steps"] + batch + 1

            mlflow.log_metric("train_batch_average_loss", train_loss, step=global_step, synchronous=False)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        self.epoch_count = epoch

        # Get predictions from the model using the generator
        start_time = tf_synchronized_time()
        
        print(f"Computing validation predictions...")
        val_predictions_prob = self.model.predict(self.validation_generator, verbose=1)
        
        end_time = tf_synchronized_time()
        
        # Collect all validation labels from generator
        all_val_labels = []
        for i in range(len(self.validation_generator)):
            _, batch_labels = self.validation_generator[i]
            all_val_labels.append(batch_labels)
        val_labels = np.concatenate(all_val_labels, axis=0)

        # Log prediction time per sample
        result = (end_time - start_time) / len(val_labels)
        mlflow.log_metric("eval/predictions_time", result, step=epoch)

        val_predictions_binary = (val_predictions_prob > 0.5).astype(int)

        # Flatten arrays
        val_labels = val_labels.flatten() if len(val_labels.shape) > 1 else val_labels
        val_predictions_binary = val_predictions_binary.flatten()
        val_predictions_prob = val_predictions_prob.flatten()

        # Create probabilities array for binary classification
        probabilities = np.column_stack([1 - val_predictions_prob, val_predictions_prob])

        # Compute detailed metrics
        print("Computing validation metrics...")
        metrics = compute_metrics_default(val_labels, val_predictions_binary, probabilities)

        evaluation_result = EvaluationResults(
            metrics=metrics,
            predictions=val_predictions_binary,
            labels=val_labels,
            probabilities=probabilities,
            loss=0.0,
        )
        log_metrics_dict_recursively(metrics, prefix="eval", step=epoch, synchronous=True)
        log_evaluation_result_to_mlflow(evaluation_result)


class GrambeddingsTrainer:

    def __init__(self, args, dataset: URLDataset):
        self.args = args
        self.dataset = dataset

    def run_training_and_evaluation(self):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        
        # Force garbage collection
        gc.collect()
        
        # Limit GPU memory growth if using GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        args = self.args
        print(args)
        mlflow.log_params(vars(args))
        
        print("####################################### Processing dataset  #######################################")
        # Keep only references to the data, don't create copies
        train_samples = self.dataset.train_set.urls
        train_labels = np.array(self.dataset.train_set.labels, dtype=np.int32)
        val_samples = self.dataset.eval_set.urls
        val_labels = np.array(self.dataset.eval_set.labels, dtype=np.int32)
        print("Completed")

        # Dictionary to store transformers
        transformers = {}

        ################################ Character Level Transformation ################################
        print("################################ Character Level Transformation ################################")
        transformer_char = CharacterLevelTransformer(
            args.max_seq_len_grambeddings, embedding_dim=args.char_embedding_dim, case_insensitive=args.case_insensitive
        )
        char_vocab_size, char_embedding_matrix = transformer_char.Fit()
        transformers['char'] = transformer_char
        print("Completed")

        ############################### First NGram Input Transformation ###############################
        print("############################### First NGram Input Transformation ###############################")
        transformer_1 = NBeddingTransformer(
            ngram_value=args.ngram_1,
            max_num_features=args.max_features,
            max_document_length=args.max_seq_len_grambeddings,
            min_df=args.min_df,
            max_df=args.max_df,
            embedding_dim=args.embed_dim,
            case_insensitive=args.case_insensitive,
            weight_mode=args.warm_mode.value,
        )
        print("Fitting input data in transformer to select best ngrams for n = ", args.ngram_1)
        rng = np.random.default_rng(args.seed)
        # subsample and also crop to args.max_length for feature extraction (everything else wont be considered anyway)
        idx = np.where(rng.random(len(train_samples)) <= args.retain_random_percentage)[0]
        train_samples_subsample = [train_samples[i][:args.max_seq_len_grambeddings] for i in idx]
        train_labels_subsample = train_labels[idx]
        
        selected_ngrams_1, selected_ngram_scores_1, weight_matrix_1, vocab_size_1, idf_dict_1 = transformer_1.Fit(
            train_samples_subsample, train_labels_subsample
        )
        transformers['ngram_1'] = transformer_1
        
        # Clean up unnecessary data
        del selected_ngrams_1, selected_ngram_scores_1, idf_dict_1
        gc.collect()
        print("Completed")

        ################################ 2nd NGram Input Transformation ################################
        print("################################ 2nd NGram Input Transformation ################################")
        transformer_2 = NBeddingTransformer(
            ngram_value=args.ngram_2,
            max_num_features=args.max_features,
            max_document_length=args.max_seq_len_grambeddings,
            min_df=args.min_df,
            max_df=args.max_df,
            embedding_dim=args.embed_dim,
            case_insensitive=args.case_insensitive,
            weight_mode=args.warm_mode.value,
        )
        print("Fitting input data in transformer to select best ngrams for n = ", args.ngram_2)
        selected_ngrams_2, selected_ngram_scores_2, weight_matrix_2, vocab_size_2, idf_dict_2 = transformer_2.Fit(
            train_samples_subsample, train_labels_subsample
        )
        transformers['ngram_2'] = transformer_2
        
        # Clean up unnecessary data
        del selected_ngrams_2, selected_ngram_scores_2, idf_dict_2
        gc.collect()
        print("Completed")

        ################################ 3rd NGram Input Transformation ################################
        print("################################ 3rd NGram Input Transformation ################################")
        transformer_3 = NBeddingTransformer(
            ngram_value=args.ngram_3,
            max_num_features=args.max_features,
            max_document_length=args.max_seq_len_grambeddings,
            min_df=args.min_df,
            max_df=args.max_df,
            embedding_dim=args.embed_dim,
            case_insensitive=args.case_insensitive,
            weight_mode=args.warm_mode.value,
        )
        print("Fitting input data in transformer to select best ngrams for n = ", args.ngram_3)
        selected_ngrams_3, selected_ngram_scores_3, weight_matrix_3, vocab_size_3, idf_dict_3 = transformer_3.Fit(
            train_samples_subsample, train_labels_subsample
        )
        transformers['ngram_3'] = transformer_3
        
        # Clean up unnecessary data
        del selected_ngrams_3, selected_ngram_scores_3, idf_dict_3
        gc.collect()
        print("Completed")

        ############################## Initializing the Parallel Networks ##############################
        print("############################## Initializing the Parallel Networks ##############################")
        char_model = NBeddingModel(
            vocab_size=char_vocab_size,
            embedding_dim=args.char_embedding_dim,
            max_seq_length=args.max_seq_len_grambeddings,
            embedding_matrix=char_embedding_matrix,
            rnn_cell_size=args.rnn_cell_size,
            attention_width=args.attn_width,
            warm_start=args.warm_start,
        )

        signal_model_1 = NBeddingModel(
            vocab_size=vocab_size_1,
            embedding_dim=args.embed_dim,
            max_seq_length=args.max_seq_len_grambeddings,
            embedding_matrix=weight_matrix_1,
            rnn_cell_size=args.rnn_cell_size,
            attention_width=args.attn_width,
            warm_start=args.warm_start,
        )

        signal_model_2 = NBeddingModel(
            vocab_size=vocab_size_2,
            embedding_dim=args.embed_dim,
            max_seq_length=args.max_seq_len_grambeddings,
            embedding_matrix=weight_matrix_2,
            rnn_cell_size=args.rnn_cell_size,
            attention_width=args.attn_width,
            warm_start=args.warm_start,
        )

        signal_model_3 = NBeddingModel(
            vocab_size=vocab_size_3,
            embedding_dim=args.embed_dim,
            max_seq_length=args.max_seq_len_grambeddings,
            embedding_matrix=weight_matrix_3,
            rnn_cell_size=args.rnn_cell_size,
            attention_width=args.attn_width,
            warm_start=args.warm_start,
        )

        # Clean up embedding matrices after model initialization
        del char_embedding_matrix, weight_matrix_1, weight_matrix_2, weight_matrix_3
        gc.collect()
        print("Completed")

        ################################ Merging the Parallel Networks #################################
        print("################################ Merging the Parallel Networks #################################")
        last_layer_char, input_layer_char = char_model.CreateModel(embedding_layer_name="embed_char")
        last_layer_1, input_layer_1 = signal_model_1.CreateModel(embedding_layer_name="embed_ngram_1")
        last_layer_2, input_layer_2 = signal_model_2.CreateModel(embedding_layer_name="embed_ngram_2")
        last_layer_3, input_layer_3 = signal_model_3.CreateModel(embedding_layer_name="embed_ngram_3")

        # Merging whole ngram layers
        embedded_concats = tf.keras.layers.Concatenate()([last_layer_char, last_layer_1, last_layer_2, last_layer_3])
        dense1 = Dense(2 * args.rnn_cell_size, activation="relu", name="deep_features")(embedded_concats)
        dropout = Dropout(0.2)(dense1)
        predictions = Dense(1, activation="sigmoid")(dropout)

        model = Model(inputs=[input_layer_char, input_layer_1, input_layer_2, input_layer_3], outputs=predictions)
        if args.loss == "focal":
            assert args.focal_loss_alpha is not None, "Focal loss alpha is not set"
            assert args.focal_loss_gamma is not None, "Focal loss gamma is not set"
            loss_fn = BinaryFocalCrossentropy(gamma=args.focal_loss_gamma, alpha=args.focal_loss_alpha)
        elif args.loss == "binary_crossentropy":
            loss_fn = "binary_crossentropy" # default loss function
        else:
            raise ValueError(f"Invalid loss function provided: {args.loss}")

        model.compile(
            optimizer=Adam(learning_rate=args.learning_rate, weight_decay=args.weight_decay),
            loss=loss_fn,
        )
        mlflow.log_param("all_params_count", model.count_params())
        mlflow.log_param("trainable_params_count", np.sum([np.prod(v.shape) for v in model.trainable_weights]))
        model.summary()

        # Clean up intermediate model objects
        del char_model, signal_model_1, signal_model_2, signal_model_3
        gc.collect()

        # Create data generators instead of transforming all data at once
        print("Creating training generator...")
        train_generator = BatchDataGenerator(
            train_samples, train_labels, transformers, 
            args.batch_size, args.max_seq_len_grambeddings, shuffle=True
        )
        print("Creating validation generator...")
        val_generator = BatchDataGenerator(
            val_samples, val_labels, transformers,
            args.batch_size, args.max_seq_len_grambeddings, shuffle=False
        )
        print("Starting training...")
        start = time.time()
        
        model.fit(
            train_generator,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            callbacks=[
                MLflowMetricsCallback(
                    val_generator, log_train_loss_every=50
                ),
            ],
            workers=2,
            use_multiprocessing=True,
            max_queue_size=10
        )

        print("Training is Completed")
        end = time.time()
        mlflow.log_metric("training_time", end - start)

        try:
            model.save_weights("model_weights.h5")
            mlflow.log_artifact("model_weights.h5", "model_weights")
            os.remove("model_weights.h5")
        except Exception as e:
            print(f"Failed to save model weights: {e}")
        
        # Final cleanup
        del train_generator, val_generator, transformers
        gc.collect()