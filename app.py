from keras.models import load_model
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import controllers.plant_identification as plant_identification
import controllers.plant_disease as plant_disease


app = Flask(__name__)
model = load_model('./RESNET_PLANT_IDENTIFICATION_CLASSES_140.h5')

@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/identify', methods=['POST'])
def identify_plant_disease():

    # Get the image from the POST request
    image_file = request.files['image']
    image_path = './images/' + image_file.filename
    image_file.save(image_path)

    plant_species_pred = plant_identification.plant_species(image_path, model)
    plant_disease_pred = plant_disease.plant_disease(image_path)

    response =  jsonify({'plant_species': plant_species_pred, 'plant_disease': plant_disease_pred})
    response.status_code = 200
    return response
    # if(len(plant_disease_pred["predictions"]) > 0):
    #     return plant_disease_pred
    # elif(plant_species_pred):
    #     return plant_species_pred
    # else:
    #     return jsonify({'error': 'Unable to identify plant'})


if __name__ == '__main__':
    # debug run
    app.run(debug=True)
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=8080)
