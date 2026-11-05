Image classifier project:
A tenserflow/keras based algorithm that can classify images up to ten distinct categories.
You can use the trained model via a FastAPI endpoint or run it on a whole folder of images with the batch script.
The process is automated via github actions.


Description of individual files:
* train.py.ipynb: model training
* api.py: FastAPI app
* batch.py: batch predictions executable
* models/: saved model (tenserflow/keras)
* outputs/: batch results (json)
* processed_images/: images already classified
* .giithub/worklflows/scheduler.yml : Automated batch processing scheduler (github actions)


Usage:

pip install -r requirements.txt

The trained model is already included in models/, so no training is needed.

Automated (GitHub Actions): add images to to_be_processed/ and enable GitHub Actions. The nightly workflow runs batch.py Monday to Friday at 3 AM , then commits the results to outputs/ and moves the images to processed_images/.

Manual batch run: python batch.py

API: uvicorn api:app --reload, then POST an image to /upload/
