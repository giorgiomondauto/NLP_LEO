# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
from nltk import word_tokenize
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib
import re
import nltk
import time
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def get_key(value, dictionary):
    for k,v in dictionary.items():
        if value == v:
            return k

# Your API definition
app = Flask(__name__)

def stemming(dataset):
                ''' stemming a text '''
                stemmer = nltk.stem.snowball.ItalianStemmer(ignore_stopwords=False)
                dataset['Job_Description'] = dataset['Job_Description'].apply(lambda x: [stemmer.stem(i) \
                                                                        for i in x])
                return dataset

def uncommon_words(data,column, italian_vocab, english_vocab = None):
                data[column] = data[column].apply(lambda x: [i for i in x if i in italian_vocab])
                return data
def remove_comuni(data, column, geo_data):
            ''' function to remove geographic noisy'''
            comune = [i.lower() for i in geo_data.Comune]
            geo_data.Provincia = geo_data.Provincia.apply(lambda x: str(x))
            provincia = [i.lower() for i in geo_data.Provincia]
            regione = [i.lower() for i in geo_data.Regione]
            
            data[column] = data[column].apply(lambda x: [i for i in x if i not in comune])
            data[column] = data[column].apply(lambda x: [i for i in x if i not in provincia])
            data[column] = data[column].apply(lambda x: [i for i in x if i not in regione])
    
            return data

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            start_time = time.time()
            json_ = request.json
            dati_aprile = pd.DataFrame(json_)
            dati_aprile.Job_Description = dati_aprile.Job_Description.apply(lambda x: re.sub('[^a-zA-Z]',' ',x))
            dati_aprile.Job_Description = dati_aprile['Job_Description'].apply(lambda x: word_tokenize(x.lower()))
            dati_aprile.Job_Description = dati_aprile.Job_Description.apply(lambda x: [i for i in x if len(i) >3])
            Fonti = ['randstad', 'monster', 'infojob', 'technical', 'kelly', 'services', 'italia', 'lavoropi',\
             'quanta','vimercate','temporary','openjobmetis','agenzia']
            dati_aprile.Job_Description = dati_aprile.Job_Description.apply(lambda x: [i for i in x if i not in Fonti])
            list_stopwords = list(set(stopwords.words('italian')))
            dati_aprile.Job_Description = dati_aprile.Job_Description.apply(lambda x: [i for i in x if i not in list_stopwords])
            geo_data = pd.read_fwf('data/listacomuni.txt')
            geo_data.to_csv('data/listacomuni.csv')
            geo_data = pd.read_csv('data/listacomuni.csv', sep=';')

            dati_aprile = remove_comuni(dati_aprile, 'Job_Description', geo_data)
            italian_vocab = pd.read_fwf('data/660000_parole_italiane.txt', names = ['word'])
            italian_vocab = set(italian_vocab.word.tolist())
            english_vocab = sorted(set(w.lower() for w in nltk.corpus.words.words())) # english vocabulary

            dati_aprile = uncommon_words(dati_aprile, 'Job_Description', italian_vocab, english_vocab)
            

            dati_aprile = stemming(dati_aprile)
            dati_aprile.Job_Description = dati_aprile.Job_Description.apply(lambda x: ' '.join(x))

            count_vector = pickle.load(open("count_vector.pkl", "rb" ) )#CountVectorizer()
            testing_data = count_vector.transform(dati_aprile['Job_Description'])

            Role_dictionary = pickle.load(open("Role_dictionary.pkl", "rb" ) )
            naive_bayes = joblib.load("model.sav")
            predictions = naive_bayes.predict(testing_data)
            predictions_prob = naive_bayes.predict_proba(testing_data)
            probability_vectors = []
            predictions_keys = []
            for i in predictions:
                predictions_keys.append(get_key(i,Role_dictionary))
                probability_vectors = [i.round(3) for i in predictions_prob[0]]
                #probability_vectors = [i for i in probability_vectors]
                # print(probability_vectors)
            #labels = ["Commesso","Statistico","Cuoco", "Cameriere", "Tecnico",'Elettromeccanico']
            labels = ['Elettromeccanico','Cameriere','Commesso','Tecnico','Statistico','Cuoco']
            dict_vect = {}
            for lab,prob in zip(labels,probability_vectors):
                dict_vect[lab]=prob
            finish_time = time.time()

            return jsonify({
            'Hai anche questi requisiti ?': 'attestato HACCP - Diploma nel settore alberghiero',\
            'Sei disponibile':'lavorare su turni, nel fine settimana.',\
            'Secondo il nostro sistema, potresti anche candidarti come': 'Cameriere',
            'Ti stai candidando come (system prediction)': ''.join(predictions_keys).upper(),\
            'It took: ':str(round(finish_time - start_time,4)) + ' seconds',\
            'labels: ':dict_vect})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load("models/naive_bayes.pickle") # Load "model.pkl"

    app.run(port=port, debug=True)