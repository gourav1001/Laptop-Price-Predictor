from flask import Flask, render_template, request
import pickle
import numpy as np

# creating flask instance
app = Flask(__name__)

# importing the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    # home page containing laptop configurations input form
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    # prediction page displaying predicted price result along with inputted configurations

    # extracting the form inputs
    features = [x for x in request.form.values()]
    company = features[0]
    laptop_type = features[1]
    ram = int(features[2])
    weight = float(features[3])
    screen_size = float(features[4])
    resolution = features[5]
    if (features[6] == 'Yes'):
        ips = 1
    else:
        ips = 0
    if (features[7] == 'Yes'):
        touchscreen = 1
    else:
        touchscreen = 0
    cpu = features[8]
    hdd = int(features[9])
    ssd = int(features[10])
    gpu = features[11]
    os = features[12]
    # extracting the width and height in pixels
    width = int(resolution.split('x')[0])
    height = int(resolution.split('x')[1])
    # calculating ppi
    ppi = ((width ** 2) + (height ** 2)) ** 0.5 / screen_size
    # creating a numpy array to store the laptop configurations
    laptop_config = np.array([company, laptop_type, ram, weight, ips, touchscreen, cpu, hdd, ssd, gpu, os, ppi])
    # reshaping the 1-d array to 1 * 12d array for prediction of price
    laptop_config = laptop_config.reshape(1, 12)
    # predicting the log price and converting the log value back to the price by applying exponential function and then rounding off to 2 places of decimal point 
    pred_price = round(np.exp(model.predict(laptop_config)[0]), 2)
    return render_template("predict.html", results = [features[0], features[1], features[2], features[3], features[4], features[5], features[6], features[7], features[8], features[9], features[10], features[11], features[12], pred_price])

if __name__ == "__main__":
    app.run(debug = True)