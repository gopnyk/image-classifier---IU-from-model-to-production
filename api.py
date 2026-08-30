from fastapi import FastAPI, UploadFile, File
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = FastAPI()

UPLOAD_DIR = "to_be_processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the model
model = tf.keras.models.load_model("models/image_classifier.keras")

# All categories within the model
class_names = ['beauty_products', 'electronics', 'fashion', 'fitness_equipments', 'furniture',
               'home_appliances', 'kitchenware', 'musical_instruments', 'study_things', 'toys']


#Image preprocessing - resizng the image
def prediction(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((64, 64))

    # convert the logit into interpretable probability
    predictions = tf.nn.softmax(model.predict(img)[0]).numpy()

    #return top 3 probabilities
    top_3 = np.argsort(predictions)[::-1][:3]

    return [
        {"category": class_names[i], "probability": round(float(predictions[i]), 4)}
        for i in top_3]




#Allows the user to upload multiple pictures via API
#stores the pictures into the "to_be_processed" folder

@app.post("/upload/")
async def upload(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    results = prediction(file_path)

    return {
        "uploaded_file": file.filename,
        "predictions": results
    }