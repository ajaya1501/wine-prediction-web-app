import pickle
import numpy as np
from flask import Flask
from flask import render_template, url_for, request, jsonify


app = Flask(__name__)

scalar = pickle.load(open("wine_quality_scaler.pkl", "rb"))
model_load = pickle.load(open("wine_quality_SVC.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_api/', methods=['POST','GET'])
def predict_api():
    data = [float(x) for x in request.form.values()]
    # convert the data into encoded numerics of defined range
    input_val = model_load.transform(np.array(data).reshape(1, -1))
    # using input value predict the output
    output_val = model_load.predict(input_val)[0]
    print(output_val)
    return render_template('index.html', text="Quality of redwine is {}".format(output_val))


if __name__ == '__main__':
    app.run(debug=True)




