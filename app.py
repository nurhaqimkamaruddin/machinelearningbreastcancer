import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os


from flask import Flask
from flask import render_template
from flaskwebgui import FlaskUI

app = Flask(__name__)
model = pickle.load(open('model3.pkl', 'rb'))


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int (float(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 5)
    return render_template('index.html', prediction_text='The cancer prognosis prediction is {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True, host= 5000)

    app.run()





from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model323.pkl','rb'))



@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int (float(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text='Patient Has {} Cancer'.format(prediction))



if __name__ == '__main__':
    app.run(debug=True)




