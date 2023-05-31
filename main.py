from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from io import BytesIO
#from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to the tomato leaves viruses prediction!"}


model = tf.keras.models.load_model('ml_model/model.h5')

def preprocess_image(image):
    #image = tf.image.decode_image(open(image, "rb").read())
    image = tf.image.resize(image, (331, 331))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = image[tf.newaxis, ...]

    return image

def image_label_mapper(prediction):
    virus_names = ['Tomato Aspermy Virus', 'Tomato Bushy Stunt Virus', 'Tomato Mosaic Virus', 'Tomato Ring Spot Virus', 'Tomato Yellow Leaf Virus', 'Z Healthy Tomato']
    virus_names= sorted(virus_names)
    return virus_names[prediction]

@app.get("/")
async def predict(image: UploadFile = File(...)):
    # Read and preprocess the image
    img_bytes = await image.read()  # Read the image file as bytes
    img = Image.open(BytesIO(img_bytes))  # Create a PIL image from the bytes
    img = np.array(img)  # Convert the PIL image to a NumPy array
    image = preprocess_image(img)



@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    try:
        if not image:
            raise HTTPException(status_code=400, detail="No image provided.")
        # Read and preprocess the image
        img_bytes = await image.read()  # Read the image file as bytes
        img = Image.open(BytesIO(img_bytes))  # Create a PIL image from the bytes
        img = np.array(img)  # Convert the PIL image to a NumPy array
        image = preprocess_image(img)
        
        # Make predictions
        prediction = np.argmax(model.predict(image))
        prediction =image_label_mapper(prediction)

        return JSONResponse({"prediction": str(prediction)})

    except tf.errors.InvalidArgumentError as e:
        # Handle image processing or preprocessing errors
        raise HTTPException(status_code=400, detail="Error processing image: " + str(e))
    
    except tf.errors.NotFoundError as e:
        # Handle model loading or prediction errors
        raise HTTPException(status_code=500, detail="Server error, try again later...: " + str(e))

    except UnidentifiedImageError as e:
    # Handle invalid file format errors
        raise HTTPException(status_code=400, detail="Invalid file format, only image allowed!: " + str(e))
    
    except Exception as e:
        # Handle other unexpected errors
        raise HTTPException(status_code=500, detail="Unexpected error: " + str(e))

