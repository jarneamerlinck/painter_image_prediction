import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
import random
import shutil

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Global vars
DATA_PATH_FILE = "data/preprocessed/data_path.csv"
DATA_PATH_PICKLE = "data/preprocessed/data_path.pkl"
RAW_FOLDER = "data/raw"
PREPROCESSING_FOLDER = "data/preprocessed"
VALID_IMAGES = [".jpg",".png",".jpeg"]

def get_raw_file_name_for_painter(painter:str) ->list: 
    to_return = []
    path = f"{RAW_FOLDER}/{painter}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"folder {path} does not exist")
    
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in VALID_IMAGES:
            continue
        to_return.append(f)
    
    return to_return

def get_data_set_for_painter(painter:str, number_of_images:int)-> tuple:
    # get images
    total_list = []
    train_list = []
    val_list = []
    test_list = []

    [total_list.append(f"{file_name}") for file_name in get_raw_file_name_for_painter(painter)]
    
    if number_of_images > len(total_list):
        #raise Exception(f"To little images of {painter} to short")
        number_of_images = len(total_list)
        print(f"less images of painter {painter}")
    # shuffle list
    random.shuffle(total_list)
    
    smaller_list = total_list[:number_of_images]
    # devide training test
    n = number_of_images/6
    number_of_train, number_of_val, number_of_test= int(4*n), int(1*n), int(1*n)

    train_list = smaller_list[:number_of_train]
    val_list = smaller_list[number_of_train:number_of_train+number_of_val]
    test_list = smaller_list[-number_of_test:]
    
    return train_list, val_list, test_list

def check_dir(dir:str):
    if not os.path.isdir(dir):
        os.makedirs(dir) 
 
def make_data_sets(painters:list, number_of_images:int=600, shape:tuple=(180, 180)):
    # setup
    shutil.rmtree(f"{PREPROCESSING_FOLDER}/train/", ignore_errors = True)
    shutil.rmtree(f"{PREPROCESSING_FOLDER}/val/", ignore_errors = True)
    shutil.rmtree(f"{PREPROCESSING_FOLDER}/test/", ignore_errors = True)
    
    check_dir(f"{PREPROCESSING_FOLDER}")
    train_list = []
    val_list = []
    test_list = []
    
    check_dir(f"{PREPROCESSING_FOLDER}/train/")
    check_dir(f"{PREPROCESSING_FOLDER}/val/")
    check_dir(f"{PREPROCESSING_FOLDER}/test/")
    
    # get images   
    for painter in painters:
        train_list, val_list, test_list = get_data_set_for_painter(painter, number_of_images)
        
        check_dir(f"{PREPROCESSING_FOLDER}/train/{painter}")
        check_dir(f"{PREPROCESSING_FOLDER}/val/{painter}")
        check_dir(f"{PREPROCESSING_FOLDER}/test/{painter}")
        
        for file_name in train_list:
            image_to_lower_res(file_name, painter, new_dir="train", shape=shape)
        for file_name in val_list:
            image_to_lower_res(file_name, painter, new_dir="val", shape=shape)
        for file_name in test_list:
            image_to_lower_res(file_name, painter, new_dir="test", shape=shape)

def image_to_lower_res(file_name:str, painter:str, new_dir:str, shape:tuple=(180, 180)):
   
    orig_path = f"{RAW_FOLDER}/{painter}/"
    new_path = f"{PREPROCESSING_FOLDER}/{new_dir}/{painter}/"
    img = Image.open(os.path.join(orig_path, file_name))
    
    resizedImage = img.resize((shape), Image.ANTIALIAS)
    if resizedImage.mode != 'RGB':
        resizedImage = resizedImage.convert('RGB')
    
    file_name, _ = os.path.splitext(file_name)
    resizedImage.save(os.path.join(new_path, f"{file_name}.png"))

def image_prepairing_website(file_name:str, shape:tuple=(180, 180)):
    orig_path = f"static/uploads/"
    new_path = f"static/preprocessed/"
    img = Image.open(os.path.join(orig_path, file_name))

    resizedImage = img.resize((shape), Image.ANTIALIAS)
    if resizedImage.mode != 'RGB':
        resizedImage = resizedImage.convert('RGB')

    file_name, _ = os.path.splitext(file_name)
    resizedImage.save(os.path.join(new_path, f"{file_name}.png"))

