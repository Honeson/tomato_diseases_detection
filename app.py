from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
from PIL import Image
from io import BytesIO
#from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the tomato leaves viruses prediction!"}


model = tf.keras.models.load_model('ml_model/model.h5')

def preprocess_image(image):
    #image = tf.image.decode_image(open(image, "rb").read())
    image = tf.image.resize(image, (331, 331))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = image[tf.newaxis, ...]

    return image

@app.get("/")
async def predict(image: UploadFile = File(...)):
    # Read and preprocess the image
    img_bytes = await image.read()  # Read the image file as bytes
    img = Image.open(BytesIO(img_bytes))  # Create a PIL image from the bytes
    img = np.array(img)  # Convert the PIL image to a NumPy array
    image = preprocess_image(img)


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

