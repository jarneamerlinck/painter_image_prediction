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
ORIG_DATA_PATH = "data/raw"
NEW_DATA_PATH = "data/preprocessed"
VALID_IMAGES = [".jpg",".png",".jpeg"]

def get_raw_file_name_for_painter(painter:str) ->list: 
    to_return = []
    path = f"{ORIG_DATA_PATH}/{painter}"
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
    train_list = []
    val_list = []
    test_list = []
    
    temp = []
    
    # [temp.append(f"{ORIG_DATA_PATH}/{painter}/{file_name}") for file_name in get_raw_file_name_for_painter(painter)]
    [temp.append(f"{file_name}") for file_name in get_raw_file_name_for_painter(painter)]
    
    if number_of_images > len(temp):
        raise Exception(f"To little images of {painter} to short")
    
    # shuffle list
    random.shuffle(temp)
    
    smaller_list = temp[:number_of_images]
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
 
def make_data_sets(painters:list, number_of_images:int=600):
    
    shutil.rmtree(NEW_DATA_PATH)
    check_dir(f"{NEW_DATA_PATH}")
    train_list = []
    val_list = []
    test_list = []
    
    check_dir(f"{NEW_DATA_PATH}/train/")
    check_dir(f"{NEW_DATA_PATH}/val/")
    check_dir(f"{NEW_DATA_PATH}/test/")
    
    # get images   
    for painter in painters:
        train_list, val_list, test_list = get_data_set_for_painter(painter, number_of_images)
        
        check_dir(f"{NEW_DATA_PATH}/train/{painter}")
        check_dir(f"{NEW_DATA_PATH}/val/{painter}")
        check_dir(f"{NEW_DATA_PATH}/test/{painter}")
        
        for file_name in train_list:
            image_to_lower_res(file_name, painter, new_dir="train")
        for file_name in val_list:
            image_to_lower_res(file_name, painter, new_dir="val")
        for file_name in test_list:
            image_to_lower_res(file_name, painter, new_dir="test")
            
            
        
   

def images_to_lower_res(painter:str, shape:tuple=(180, 180)) -> bool:
    file_names = get_raw_file_name_for_painter(painter)
    orig_path = f"{ORIG_DATA_PATH}/{painter}/"
    
    [image_to_lower_res(file_name, painter, shape) for file_name in file_names]
    return True
    

def image_to_lower_res(file_name:str, painter:str, new_dir:str, shape:tuple=(180, 180)):
    orig_path = f"{ORIG_DATA_PATH}/{painter}/"
    new_path = f"{NEW_DATA_PATH}/{new_dir}/{painter}/"
    img = Image.open(os.path.join(orig_path, file_name))
    cover = img.resize((180, 180))
    cover.save(os.path.join(new_path, file_name))    
    
# def image_to_lower_res(file_name:str, painter:str, shape:tuple=(180, 180)):
#     orig_path = f"{ORIG_DATA_PATH}/{painter}/"
#     new_path = f"{NEW_DATA_PATH}/{painter}/"
#     img = Image.open(os.path.join(orig_path, file_name))
#     cover = img.resize((180, 180))
#     cover.save(os.path.join(new_path, file_name))

def get_features(path):
    pass