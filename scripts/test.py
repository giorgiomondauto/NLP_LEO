
from flask import Flask, request, redirect, url_for, flash, jsonify
from features_calculation import doTheCalculation
import json, pickle
import pandas as pd
import numpy as np
app = Flask(__name__)

@app.route('/api/makecalc/', methods=['POST'])

def makecalc():
    jsonfile = request.get_json()
    data = pd.read_json(json.dumps(jsonfile),orient='index',convert_dates=['dteday'])
    print(data)

	res = dict()

	ypred = model.predict(doTheCalculation(data))

	for i in range(len(ypred)):

    	    res[i] = ypred[i]

    

	return jsonify(res) 

if __name__ == '__main__':

	modelfile = 'modelfile.pickle'    

	model = pickle.load(open(modelfile, 'rb'))

	print("loaded OK")

	app.run(debug=True)