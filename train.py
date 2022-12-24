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
        
        inputs = keras.Input(shape=(input_shape))        
        x = layers.Conv1D(256, 8, padding='same', activation="ReLU")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        x = layers.MaxPool1D(pool_size=8)(x)
        
        x = layers.Conv1D(128, 8, padding='same', activation="ReLU")(x)
        x = layers.Conv1D(128, 8, padding='same', activation="ReLU")(x)
        x = layers.Conv1D(128, 8, padding='same', activation="ReLU")(x)
        x = layers.Conv1D(128, 8, padding='same', activation="ReLU")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        x = layers.MaxPool1D(pool_size=8)(x)
        x = layers.Conv1D(64, 8, padding='same', activation="ReLU")(x)
        
        x = layers.Conv1D(64, 8, padding='same', activation="ReLU")(x)
        x = layers.Flatten()(x)
        
        outputs = layers.Dense(output_shape, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
        
        return model

    def load_features(self):
        data_path = pd.read_pickle(DATA_PATH_PICKLE)
        path = np.array(data_path.Path)[1]
    
        data, sample_rate = librosa.load(path)

        X, Y = [], []
        for path, emotion in zip(data_path.Path, data_path.Emotions):
            feature = get_features(path)
            for ele in feature:
                X.append(ele)
                Y.append(emotion)
                
        features = pd.DataFrame(X)
        features['labels'] = Y
        features.to_csv(f"{self.BACKUP_FOLDER}/features.csv", index=False)
        mlflow.log_artifact(f"{self.BACKUP_FOLDER}/features.csv", artifact_path=self.BACKUP_FOLDER)
        
        X = features.iloc[: ,:-1].values
        Y = features['labels'].values

        encoder = OneHotEncoder()
        Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True, test_size=1/6)
        x_train, y_train, x_val, y_val = train_test_split(x_train, y_train, random_state=0, shuffle=True, test_size=1/5)
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train = np.expand_dims(x_train, axis=2)
        x_test = np.expand_dims(x_test, axis=2)
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.set_encoder(encoder)
    
    def _set_train_options(self):
        self.batch_size = 64
        self.epochs = 100
        self.callback = rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
        self.input_shape = (self.x_train.shape[1],1)
        self.labels = self.y_train.shape[1]


# Main function for training model on split train data and evaluating on validation data
def main():
    experiment = "experimental"
    model_name = "NN_sequential"
    model_version = "001"
    mlflow_controller = Controller(experiment, model_name, model_version)
    
    mlflow_controller()

if __name__=='__main__':
    main()