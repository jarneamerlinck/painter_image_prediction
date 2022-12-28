# Load packages
## Data wrangling
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Activation
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras import layers

import re


from helpers.preprocessing_helpers import *
from helpers.training_helpers import *
from helpers.mlflow_helpers import *

## Parser arguments
import argparse


# Set our parser arguments. 
parser = argparse.ArgumentParser(
    description='Painter image prediction')

parser.add_argument('--mlflow_run', default=0, type=int,
                    help="Running as an experiment or not. Don't change this, this gets automatically changed by the MLFlow default parameters")

args = parser.parse_args()

if args.mlflow_run:
    from mlflow import log_metric, log_param, log_artifacts

class Controller(Mlflow_controller):
    def _build_model(self, input_shape, output_shape: int):
        
        inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=2)(x)

        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        outputs = layers.Dense(output_shape, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
        
        return model

    def load_features(self):
        self.batch_size = 64
        self.image_shape = (180, 180)
        self.train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            f"{PREPROCESSING_FOLDER}/train", image_size=self.image_shape, batch_size=self.batch_size, label_mode='categorical'
        )
        self.test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            f"{PREPROCESSING_FOLDER}/test", image_size=self.image_shape, batch_size=self.batch_size, label_mode='categorical' 
        )
        self.val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            f"{PREPROCESSING_FOLDER}/val", image_size=self.image_shape, batch_size=self.batch_size, label_mode='categorical' 
        )
               
    
    def _set_train_options(self):
        self.uses_datasets = True
        # self.batch_size = 64
        self.epochs = 50
        filename = "painter_baseline.keras"
        self.callback = [keras.callbacks.ModelCheckpoint(filename, save_best_only=True), MlflowCallback("accuracy")]
        self.input_shape = (180, 180, 3)
        self.labels = len(self.train_dataset.class_names)
        
    def mlflow_log(self):
        t_loss, t_accuracy = self.model.evaluate(self.test_dataset)
        mlflow.log_metrics({"test_loss": t_loss,"test_accuracy": t_accuracy})
        mlflow.log_artifact("train.py", artifact_path=self.BACKUP_FOLDER)
    def classification_report(self):
        self.plot_loss()
        self.save_to_mlflow()


# Main function for training model on split train data and evaluating on validation data
def main():
    experiment = "baseline"
    model_name = "Baseline_model"
    model_version = "001"
    mlflow_controller = Controller(experiment, model_name, model_version)
    
    mlflow_controller()

if __name__=='__main__':
    main()