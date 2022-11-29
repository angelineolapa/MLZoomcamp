import tflite_runtime.interpreter as tflite
import numpy as np
import os
from PIL import Image
from io import BytesIO
from urllib import request

#url = "https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg"
MODEL_NAME = os.getenv("MODEL_NAME", "dino-vs-dragon-v2.tflite")

interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

#Functions to download, resize and rescale images 

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def resize_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(img):
    x = np.array(img, dtype="float32")
    X = np.array([x])
    X = np.divide(X, 255)
    return X

def predict(url):
    img = download_image(url)
    img = resize_image(img, (150,150))
    X = preprocess_image(img)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
  
    return float(prediction[0,0])

def lambda_handler(event, context):
    url = event["url"]
    pred = predict(url)
    result = pred
    return result