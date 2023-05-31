from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
from io import BytesIO
#from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI()

model = tf.keras.models.load_model('ml_model/model.h5')

def preprocess_image(image):
    #image = tf.image.decode_image(open(image, "rb").read())
    image = tf.image.resize(image, (331, 331))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = image[tf.newaxis, ...]

    return image

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Read and preprocess the image
    img_bytes = await image.read()  # Read the image file as bytes
    img = Image.open(BytesIO(img_bytes))  # Create a PIL image from the bytes
    img = np.array(img)  # Convert the PIL image to a NumPy array
    image = preprocess_image(img)
    
    # Make predictions
    prediction = np.argmax(model.predict(image))

    return JSONResponse({"prediction": str(prediction)})

