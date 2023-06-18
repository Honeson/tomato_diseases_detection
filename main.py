from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from io import BytesIO
#from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI()


@app.get("/", summary="Homepage", description="This endpoint is just a homepage ")
async def read_root():
    return {"message": "Welcome to the tomato leaves viruses prediction!"}


model = tf.keras.models.load_model('ml_model/saved_model.h5')
IMAGE_SIZE = (224,224)
class_names = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# def preprocess_image(image):
#     #image = tf.image.decode_image(open(image, "rb").read())
#     image = tf.image.resize(image, (331, 331))
#     image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#     image = image[tf.newaxis, ...]

#     return image

# def image_label_mapper(prediction):
#     virus_names = ['Tomato Aspermy Virus', 'Tomato Bushy Stunt Virus', 'Tomato Mosaic Virus', 'Tomato Ring Spot Virus', 'Tomato Yellow Leaf Virus', 'Z Healthy Tomato']
#     virus_names= sorted(virus_names)
#     return virus_names[prediction]

# def predict_single_image(image, class_names, model):
#     # Create an ImageDataGenerator for a single image
#     data_generator = ImageDataGenerator(rescale=1./255)
#     image_generator = data_generator.flow(np.expand_dims(image, axis=0), batch_size=1)

#     predicted = model.predict(image_generator)
#     predicted_label = class_names[np.argmax(predicted)]

#     return predicted_label


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.post("/predict", summary="Prediction Endpoint", description="Endpoint to predict the virus present in a tomato leaf image. This is the endpoint you need")
async def predict(image: UploadFile = File(...)):
    """
    Make predictions on the provided tomato leaf image.

    - **image**: Only image file will be accepted by this endpoint(e.g., JPEG, PNG,...).
    """

    try:
        if not image:
            raise HTTPException(status_code=400, detail="No image provided.")
        # Read and preprocess the image
        img_bytes = await image.read()  # Read the image file as bytes
        img = Image.open(BytesIO(img_bytes))  # Create a PIL image from the bytes
        #img = np.array(img)  # Convert the PIL image to a NumPy array
        img = img.resize(IMAGE_SIZE)
        data_generator = ImageDataGenerator(rescale=1./255)
        image_generator = data_generator.flow(np.expand_dims(img, axis=0), batch_size=1)
        
        # Make predictions
        predicted = model.predict(image_generator)
        predicted_label = class_names[np.argmax(predicted)]

        #prediction = np.argmax(model.predict(image))
        #prediction =image_label_mapper(prediction)

        return JSONResponse({"prediction": str(predicted_label)})

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

