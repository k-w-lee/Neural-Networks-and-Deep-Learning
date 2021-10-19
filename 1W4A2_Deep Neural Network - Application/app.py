from flask import Flask, render_template, request, url_for
import pickle
from dnn_app_utils_v3 import *
import numpy as np
from PIL import Image
import pandas as pd

filename = 'parameters.sav'
filename2 = 'titanic_model_voting.sav'
filename3 = 'parkinsons_model.sav'
loaded_parameters = pickle.load(open(filename, 'rb'))
titanic_model = pickle.load(open(filename2, 'rb'))
parkinsons_model = pickle.load(open(filename3, 'rb'))
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

def predict(X, parameters):
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    return p

app = Flask(__name__, template_folder='template')

@app.route('/') 
def index():
    return render_template('index.html')



@app.route("/prediction", methods = ['POST'])
def cat_prediction():

    img = request.files['img']
    img.save("static/img.jpg")
    num_px = 64
    # my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
    fname = "static/" + "img.jpg"
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(image, loaded_parameters)
    result = classes[int(np.squeeze(my_predicted_image)),].decode("utf-8")
    return render_template("cat_prediction.html", data= result)

def survival(x):
    if x == 0.0:
        temp = "Not Survive"
    else:
        temp = "Survived"
    return temp
@app.route("/titanic_prediction", methods = ['POST'])
def titanic_prediction():
    data1 = request.form['Ticket Class']
    data2 = request.form['Gender']
    data3 = request.form['Age']
    data4 = request.form['sibsp']
    data5 = request.form['parch']
    data6 = request.form['Passenger fare']
    data7 = request.form['Port of Embarkation']
    observation = np.array([[data1, data2, data3, data4, data5, data6, data7]])
    titanic_predict = titanic_model.predict(observation)
    titanic_result = survival(titanic_predict[0])
    return render_template("titanic_pred.html", data= titanic_result)

@app.route("/parkinson_prediction", methods = ['POST'])
def parkinson_prediction():
    park1 = request.form['park_age']
    park2 = request.form['park_sex']
    park3 = request.form['jitter']
    park4 = request.form['shimmer']
    park5 = request.form['NHR']
    park6 = request.form['HNR']
    park7 = request.form['RPDE']
    park8 = request.form['DFA']
    park9 = request.form['PPE']
    park_observation = np.array([[park1, park2, park3, park4, park5, park6, park7, park8, park9]])
    parkinson_predict = parkinsons_model.predict(park_observation)
    park_result = parkinson_predict[0]
    return render_template("park_pred.html", data= park_result)
if __name__ == "__main__":
    app.run(debug=True)