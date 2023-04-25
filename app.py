from keras.models import load_model
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np


app = Flask(__name__)
model = load_model('./RESNET50_PLANT_DISEASE.h5')


class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


# Load the model


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/identify-plant-disease', methods=['POST'])
def identify_plant_disease():

    # Get the image from the POST request
    image_file = request.files['image']
    image_path = './images/' + image_file.filename
    image_file.save(image_path)

    # Load the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    # Make prediction using model loaded from disk as per the data. 
    classes = model.predict(x)
    # Take the first value of prediction
    probability = np.max(classes[0])
    print("Probability: ", probability)
    index = np.argmax(classes[0])
    print(class_names[index])
    return jsonify({'disease': class_names[index], 'probability': str(probability)})


if __name__ == '__main__':
    app.run(port=3000, debug=True)
