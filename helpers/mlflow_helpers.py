import numpy as np
import pandas as pd
import os
import sys
import re
import pickle
from dotenv import load_dotenv

import mlflow
from mlflow import log_metric, log_param, log_artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.keras import log_model

from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report


class MlflowCallback(keras.callbacks.Callback):
    """This class is used to log training and validation loss and another metric to mlflow
    
    Needs to be passed to the model with other callbacks.
    Model fit needs to be inside of mlflow run
    In MlflowController this is possible in the _set_train_options() method
    
    Args:
            metric (str): Name of the metric mlflow needs to log
            
    Examples:
        >>> callback = [MlflowCallback("accuracy")]
        >>> model.fit(
                train_dataset, 
                batch_size=batch_size, epochs=epochs, 
                validation_data=(val_dataset), 
                callbacks=callback
                )

    """
    def __init__(self, metric:str):
        self.metric = metric
    def on_epoch_end(self, epoch, logs=None):
        """logs after each epoch

        Args:
            epoch (_type_): Numberof epoch
            logs (_type_, optional): logs, Defaults to None.
        """
        mlflow.log_metrics({
           "loss": logs["loss"],
           self.metric:
               logs[self.metric],
           "val_loss": logs["val_loss"],
           self.metric:
               logs[self.metric],
    })

class Mlflow_controller(ABC):
    """Absolute class to log model runs to mlflow
    Args:
            experiment_name (str): Name of the mlflow experiment
            model_name (str): Name of the model
            version (str): Version of the model
            
    Attributes:
        FEATURES_PATH (str): Path to feature file
        LABELS_PATH (str): Path to label file
        ENCODER_PICKLE (str): Path to encoder file
        OUTPUT_FOLDER (str): Path to output folder
        BACKUP_FOLDER (str): Path to backup folder
        PREPROCESSING_FOLDER (str): Path to preprocessing folder
        DOTENV_FILE (str): Path to dotenv file
    Examples:
        
        >>> import tensorflow as tf
        >>> from tensorflow import keras
        >>> from tensorflow.python.keras.callbacks import ModelCheckpoint
        >>> from tensorflow.keras import layers
        >>> import mlflow

        >>> from helpers.mlflow_helpers import *

        >>> class Controller(Mlflow_controller):
                def _build_model(self, input_shape, output_shape: int):
                    inputs = keras.Input(shape=input_shape)
                    x = inputs        
                    x = layers.Flatten()(x)
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
                    self.epochs = 30
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
        # Use the class
        >>> experiment = "default"
        >>> model_name = "demo_model"
        >>> model_version = "001"
        >>> mlflow_controller = Controller(experiment, model_name, model_version)

        >>> mlflow_controller()
    """
    FEATURES_PATH = "data/output/features.csv"
    LABELS_PATH = "data/output/labels.csv"
    ENCODER_PICKLE = "data/backup/encoder.pkl"
    OUTPUT_FOLDER = "data/output"
    BACKUP_FOLDER = "data/backup"
    PREPROCESSING_FOLDER = "data/preprocessed"
    DOTENV_FILE = ".env"
    
    def __init__(self, experiment_name:str, model_name:str, version:str):
        self._experiment_name = experiment_name
        self._model_name = model_name
        self._model_version = version
        self._model_filename = f"models/{self._model_name}_{self._model_version}.keras"
        self._mlflow_setup()
        self.uses_datasets=False

    def _mlflow_setup(self) -> None:
        """Set all settings for mlflow correct
        """
        load_dotenv(self.DOTENV_FILE)
        mlflow.set_experiment(experiment_name=self._experiment_name)
        mlflow.autolog() 
        mlflow.tensorflow.autolog()
        print(f"Experiment {self._experiment_name} as active")

    def log_history(self, history) -> None:
        """Saves history
        """
        self._history = history
        
    def predict(self, x_test):
        """Predicts for X

        Args:
            x_test (numpy array): features

        Returns:
           numpy array: predicted labels
        """
        return self.model.predict(x_test)
    
    def classification_report(self, x_test, y_test):
        """creates the classification_report (only without datasets)
        Is called in train()
        Override when using datasets
        ```
        def classification_report(self):
            self.plot_loss()
            self.save_to_mlflow()
        ```
        Args:
            x_test (_type_): _description_
            y_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        pred_test = self.model.predict(x_test)
        if hasattr(self,'_encoder'):
            y_pred = self._encoder.inverse_transform(pred_test)
            y_test = self._encoder.inverse_transform(y_test)
        else:
            y_pred = pred_test
        
        self.plot_loss()
        self.save_to_mlflow()
        
        if hasattr(self,'__no_tests') and self.__no_tests == True:
            report = None
        else:
            report, _ = self.classification_report_to_dataframe(y_test, y_pred)
        
        return report

    def plot_loss(self):
        """Plots Training and validation accuracy and loss 
        Saves the plot to mlflow
        """
        
        plt.rcParams['figure.dpi'] = 160  # figure size vergroten
        accuracy = self._history.history["accuracy"]
        val_accuracy = self._history.history["val_accuracy"]
        loss = self._history.history["loss"]
        val_loss = self._history.history["val_loss"]
        epochs = range(1, len(accuracy) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(epochs, accuracy, "bo", label="Training accuracy")
        ax1.plot(epochs, val_accuracy, "b", label="Validation accuracy")
        ax1.set_title("Accuracy")
        ax1.legend();

        ax2.plot(epochs, loss, "bo", label="Training loss")
        ax2.plot(epochs, val_loss, "b", label="Validation loss")
        ax2.set_title("Loss")
        ax2.legend();
        
        fig.savefig(f"{self.OUTPUT_FOLDER}/plot_loss.png")
        mlflow.log_figure(fig, f"{self.OUTPUT_FOLDER}/plot_loss.png")
        
    def save_to_mlflow(self):
        """Logs encoder and model shape to mlflow
        """
        self.__model_summary_to_MLFlow()
        try:
            save_encoder(self._encoder, self.ENCODER_PICKLE)
            mlflow.log_artifact(self.ENCODER_PICKLE, artifact_path=self.BACKUP_FOLDER)
        except:
            pass
        try:
            plot_model_filename = f"{self.OUTPUT_FOLDER}/model.png"
            plot_model(self.model, to_file=plot_model_filename, show_shapes = True, show_layer_names = True, show_layer_activations = True)
            mlflow.log_artifact(plot_model_filename, artifact_path=self.OUTPUT_FOLDER)
        except:
            print("error when getting model plot")
    
    def __model_summary_to_MLFlow(self):
        """Logs metrics about the model to mlflow
        """
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        summ_string = "\n".join(stringlist)
        print(summ_string) # entire summary in a variable

        table = stringlist[1:-4][1::2] # take every other element and remove appendix

        new_table = []
        for entry in table:
            entry = re.split(r'\s{2,}', entry)[:-1] # remove whitespace
            new_table.append(entry)

        df = pd.DataFrame(new_table[1:], columns=new_table[0])
        
        mlflow.set_tag("model_name", self._model_name)
        mlflow.set_tag("model_version", self._model_version)
        mlflow.set_tag("number_of_hidden_layers", df.shape[0])
        
        number_of_current_layer = 0
        columns = df.columns
        for r in range(df.shape[0]):
            f = df.iloc[r,0]
            output_shape = df.iloc[r,1]
            params = df.iloc[r,2]
            print(f, output_shape, params)
            
            log_param(f"hidden_layer_{number_of_current_layer:03}_model",f)
            log_param(f"hidden_layer_{number_of_current_layer:03}_shape",output_shape)
            log_param(f"hidden_layer_{number_of_current_layer:03}_numer_of_params",params) 
            number_of_current_layer+=1

    def __call__(self):
        """Call of the class that runs mlflow
        """
        self._mlflow_run()

    def train(self):
        """Trains the model, logs history and creates the classification_report 
        """
        if self.uses_datasets:
            history=self.model.fit(
                self.train_dataset, 
                batch_size=self.batch_size, epochs=self.epochs, 
                validation_data=(self.val_dataset), 
                callbacks=self.callback
                )
        else:
            history=self.model.fit(
                self.x_train, self.y_train, 
                batch_size=self.batch_size, epochs=self.epochs, 
                validation_data=(self.x_val, self.y_val), 
                callbacks=self.callback
                )
        
        self.log_history(history)
        if self.uses_datasets:
            self.classification_report()
        else:
            report = self.classification_report(self.x_test, self.y_test)
        
    def model_summary(self):
        """Prints the model summary
        """
        self.model.summary()

    def set_encoder(self, encoder):
        """Saves encoder
        Needs to be used in the child class

        Args:
            encoder (_type_): Encoder for the labels
        """
        self._encoder = encoder
    
    def _mlflow_run(self):
        """Runs mlflow 
        Sequence of used methods
        1. load_features()
        2. _set_train_options()
        3. _build_model()
        4. train()
        5. mlflow_log()
        """
        self._mlflow_setup()
        
        with mlflow.start_run(run_name=f'{self._model_name}_{self._model_version}') as run:
            print("MLflow:")
            print("  run_id:",run.info.run_id)
            print("  tracking-uri:",run.info.run_id)
            print("  experiment_id:",run.info.experiment_id)
            mlflow.set_tag("version.mlflow", mlflow.__version__)
            mlflow.set_tag("version.keras", keras.__version__)
            mlflow.set_tag("version.tensorflow", tf.__version__)
            
            self.load_features()
            self._set_train_options()
            self.model = self._build_model(self.input_shape, self.labels)
            self.train()
            self.mlflow_log()

    
    def mlflow_log(self):
        """This function is used to log artifacts, tags or metrics to mlflow after the training process
        """
        
        pass
        
    @abstractmethod
    def _build_model(self)-> keras.Model:
        """This function makes, compiles and returns the keras model

        Returns:
            keras.Model: Model to be trained
        """
        pass
    
    @abstractmethod
    def load_features(self):
        """Loads data/features for the training process
        Needs to set the following object variables
        - x_train
        - x_val
        - x_test
        - y_train
        - y_val
        - y_test
        Or (set self.uses_datasets = True)
        - train_dataset
        - val_dataset
        - test_dataset
        """
        pass
    
    @abstractmethod
    def _set_train_options(self):
        """sets the training options
        Needs to set following object variables
        - batch_size
        - epochs
        - callback
        - input_shape
        - labels
        - callback
        """
        pass
    
    @staticmethod
    def classification_report_to_dataframe(y_test, y_pred):
        """Parses the classification_report 

        Args:
            y_test (numpy array): Test labels
            y_pred (numpy array): Predicted labels

        Returns:
            dataframe: Dataframe with the classification report
            float: Accuracy
        """
        report  = classification_report(y_test, y_pred)
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            row = {}
            if line =='':
                pass
            elif 'accuracy' in line:
                row_data = re.split(' +', line)
                accuracy = row_data[2]
                break
            else:
                row_data = re.split(' +', line)
                row['class'] = row_data[1]
                row['precision'] = float(row_data[2])
                row['recall'] = float(row_data[3])
                row['f1_score'] = float(row_data[4])
                row['support'] = float(row_data[5])
                report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        return dataframe, accuracy


def save_encoder(encoder: OneHotEncoder, filename:str):
    """Saves OneHotEncoder to file

    Args:
        encoder (OneHotEncoder): OneHotEncoder to save
        filename (str): Filename
    """
    with open(filename, 'wb') as file:
        pickle.dump(encoder, file)

def load_encoder(filename:str) -> OneHotEncoder:
    """_summary_

    Args:
        filename (str):Filename

    Returns:
        OneHotEncoder: Onehotencoder loaded from file
    """
    pickle_off = open (filename, "rb")
    return pickle.load(pickle_off)
