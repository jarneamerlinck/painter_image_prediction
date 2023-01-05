import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
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



ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app=Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 2048 * 2048


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def load_model():
    input_shape = (180, 180, 3)
    output_shape = 3

    data_augmentation = keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2),
                ])
            
    conv_base = keras.applications.vgg19.VGG19(
                weights="imagenet",
                include_top=False
                )

    conv_base.trainable = True
    for layer in conv_base.layers[:-2]:
        layer.trainable = False


    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = data_augmentation(x) 

    x = keras.applications.vgg19.preprocess_input(x)
    x = conv_base(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(output_shape, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
    model.load_weights("painter_baseline.keras")
    return model

def predict(filename):
    img_width, img_height = 180, 180

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
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                f"{PREPROCESSING_FOLDER}/test", image_size=(img_width, img_height), batch_size=64, label_mode='categorical'
                )
    
    class_names = test_dataset.class_names

    return "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    
    
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

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
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(port=8080)