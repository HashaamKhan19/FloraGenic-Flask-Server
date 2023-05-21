from roboflow import Roboflow
rf = Roboflow(api_key="XHKcblVVWk0GVBNDx2bg")
project = rf.workspace().project("plant-disease-detection-z9xot")
model = project.version(2).model

# infer on a local image

def plant_disease(image_path):
    return model.predict(image_path, confidence=50, overlap=30).json()

