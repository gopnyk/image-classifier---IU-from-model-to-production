import os
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from datetime import date

UPLOAD_DIR = "to_be_processed"
OUTPUT_FILE = f"outputs/predictions_{date.today()}.json"

#create an outputs folder
os.makedirs("outputs", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Load the model
model = tf.keras.models.load_model("models/image_classifier.keras")

# All categories within the model
class_names = ['beauty_products', 'electronics', 'fashion', 'fitness_equipments', 'furniture',
               'home_appliances', 'kitchenware', 'musical_instruments', 'study_things', 'toys']


#Image preprocessing
def preprocess(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((64, 64))
    return np.array(img)

#double preprocessing ruined the results - presentation


#~~~~~~~~~~~~~~~~~~Loading the images into the model and saving the output~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_batch():

    #fetching the images
    files = os.listdir(UPLOAD_DIR)

    images = []
    filenames = []

    for file in files:
        path = os.path.join(UPLOAD_DIR, file)
        images.append(preprocess(path))
        filenames.append(file)

    if len(images) == 0:
        print("No images have been uploaded")
        return

    #assigning the predicted class to the image

    predictions = model.predict(np.array(images))

    results = []

    for i, pred in enumerate(predictions):
        results.append({"file": filenames[i],"predicted_class": class_names[np.argmax(pred)]})

    #save as json file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # the processed images are moved to a separate folder~~~~~~~~~~~~~~~~~~~~~~~~

    done_dir = "processed_images"
    os.makedirs(done_dir, exist_ok=True)
    for file in filenames:
        os.rename(
            os.path.join(UPLOAD_DIR, file),
            os.path.join(done_dir, file))

if __name__ == "__main__":
    run_batch()



