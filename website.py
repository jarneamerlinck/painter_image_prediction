import os
from pickle import NONE
from flask import Flask, flash, request, redirect, url_for, render_template
from flask_ngrok2 import run_with_ngrok

from werkzeug.utils import secure_filename
from helpers.preprocessing_helpers import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Activation
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import mlflow
from dotenv import load_dotenv

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app=Flask(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
MODEL_LINK = "s3://mlflow/6/2a5ec012885541e8b017573e965aff99/artifacts/model"
MODEL=None
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 2048 * 2048

img_width, img_height = 180, 180
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                f"{PREPROCESSING_FOLDER}/test", image_size=(img_width, img_height), batch_size=64, label_mode='categorical'
                )
    
CLASS_NAMES = test_dataset.class_names

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def load_model():
    global MODEL
    if MODEL== None:
        MODEL = mlflow.tensorflow.load_model(MODEL_LINK)
    return MODEL



def predict(filename):
    

    model = load_model()
    filename = f'static/preprocessed/{filename}'
    filename, _ = os.path.splitext(filename)
    filename = filename +".png" 

    img = tf.keras.utils.load_img(
    filename, target_size = (img_width, img_height)
    )   
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    

    return "This image most likely belongs to {} with a {:.2f} percent confidence.".format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score))
    	
@app.route('/')
def upload_form():

	return render_template('upload.html', painters=CLASS_NAMES)

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename))

        image_prepairing_website(filename)
        result  = predict(filename)
        flash(result)
        return render_template('upload.html', filename=filename, painters=CLASS_NAMES)
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    load_dotenv()
    auth_token = os.getenv('NGROK_ACCESS_TOKEN')
    run_with_ngrok(app=app, auth_token=auth_token)
    app.run()