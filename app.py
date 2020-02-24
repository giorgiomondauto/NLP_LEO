import time
import pandas as pd
import numpy as np
import pickle as p
import sys
import connexion
import json
from preprocessing import load_create_data,remove_noisy, remove_comuni,uncommon_words
from preprocessing import split_data, bag_of_words
from training import model_training
from prediction import prediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask import Flask, request, redirect, url_for, flash, jsonify
from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask, request, redirect, url_for, flash, jsonify
# from features_calculation import doTheCalculation
import json, pickle
import pandas as pd
import numpy as np


def main():
    start_time = time.time()
    data, geo_data, italian_vocab, _ = load_create_data('train.csv', 'listacomuni.csv', '660000_parole_italiane.txt')
    data = remove_noisy(data)
    data = remove_comuni(data, 'Job_Description', geo_data)
    data = uncommon_words(data, 'Job_Description', italian_vocab)
    Role_dictionary = pd.Series(data['Role'].unique()).to_dict()
    Role_dictionary = dict([(value, key) for key, value in Role_dictionary.items()])
    role_encoded = []
    for i in data.Role:
        role_encoded.append(Role_dictionary.get(i, None))
    data.Role = role_encoded
    data.Job_Description = data.Job_Description.apply(lambda x: ' '.join(x))
    X_train, X_test, y_train, y_test = split_data(data, text_column = 'Job_Description', target_column = 'Role')
    training_data, testing_data = bag_of_words(X_train, X_test)
    print('Time to preprocess the data is:', round((time.time() - start_time),3), 'seconds')
    model = model_training(training_data, y_train)
    predictions, predictions_keys, actual_predictions = prediction(model, testing_data, Role_dictionary, y_test)
    print( 100 * '%')
    print('Prediction for rows {} are \n {}'.format(X_test.index.tolist(),predictions_keys))
    print( 10 * '%')
    print('actual labels are: \n {}'.format(actual_predictions))
    print( 10 * '%')
    print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
    p.dump(model, open('models/naive_bayes.pickle', 'wb'))
    

    return training_data, testing_data, y_train, y_test

# if __name__ == '__main__':
#     main()
APP = connexion.App(__name__)

@APP.app.route('/prediction', methods=['GET'])
def claim():
    '''
    Returns predicted procedure codes
    '''
    if request.method == 'GET':
        if request.args.get("index") != None:
            index = request.get_json(force=True)
            index = pd.DataFrame({'Job_Description':index})
            _, geo_data, italian_vocab, _ = load_create_data('train.csv', 'listacomuni.csv', '660000_parole_italiane.txt')
            data = remove_noisy(index)
            data = remove_comuni(data, 'Job_Description', geo_data)
            data = uncommon_words(data, 'Job_Description', italian_vocab)
            # # Role_dictionary = pd.Series(data['Role'].unique()).to_dict()
            # Role_dictionary = dict([(value, key) for key, value in Role_dictionary.items()])
            # role_encoded = []
            # for i in data.Role:
                # role_encoded.append(Role_dictionary.get(i, None))
            # data.Role = role_encoded
            data.Job_Description = data.Job_Description.apply(lambda x: ' '.join(x))
            count_vector = CountVectorizer()
            testing_data = count_vector.transform(data.Job_Description)
            print(index)
        else:
            print('Error')
        # index = int(index)
        modelfile = 'models/final_prediction.pickle'
        model = p.load(open(modelfile, 'rb'))
        with open('Role_dictionary.pickle', 'rb') as handle:
             Role_dictionary = p.load(handle)

        predictions, predictions_keys, _ = prediction(model, testing_data, Role_dictionary)
        return jsonify(results=predictions)
        #return {"status": 200, "predictions": predictions, "predictions_keys" : predictions_keys}


app = Flask(__name__)

@app.route('/api/prediction/', methods=['POST'])


def makecalc():
    index =  request.get_json()
    print(index)
    # data = pd.read_json(json.dumps(index))
    index = pd.DataFrame({'Job_Description':str(index)})
    _, geo_data, italian_vocab, _ = load_create_data('train.csv', 'listacomuni.csv', '660000_parole_italiane.txt')
    data = remove_noisy(index)
    data = remove_comuni(data, 'Job_Description', geo_data)
    data = uncommon_words(data, 'Job_Description', italian_vocab)
    # data.Job_Description = data.Job_Description.apply(lambda x: ' '.join(x))
    count_vector = CountVectorizer()
    testing_data = count_vector.transform(data.Job_Description)
    print(testing_data)
    modelfile = 'models/final_prediction.pickle'
    model = p.load(open(modelfile, 'rb'))

    with open('Role_dictionary.pickle', 'rb') as handle:
                Role_dictionary = p.load(handle)

    predictions, predictions_keys, _ = prediction(model, testing_data, Role_dictionary)


    return jsonify(predictions) 

if __name__ == '__main__':
    modelfile = 'models/final_prediction.pickle'
    model = p.load(open(modelfile, 'rb'))
	app.run(debug=True, host='0.0.0.0')
