import pandas as pd
import matplotlib
import re
import nltk
import time
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def load_create_data(file_name, file_geo_data_name, italian_vocab_name):
    ''' to load and create data '''
    data = pd.read_csv(file_name)
    geo_data = pd.read_csv(file_geo_data_name, sep=';')
    italian_vocab = pd.read_fwf(italian_vocab_name, names = ['word'])
    italian_vocab = set(italian_vocab.word.tolist())
    english_vocab = sorted(set(w.lower() for w in nltk.corpus.words.words()))
    return data, geo_data, italian_vocab, english_vocab


def remove_noisy(data):
    ''' to remove noisy from test '''
    data.Job_Description = data.Job_Description.apply(lambda x: re.sub(r'[^\w\s]','',x))
    data.Role = data.Role.apply(lambda x: x.lower())
    data.Job_Description = data['Job_Description'].apply(lambda x: word_tokenize(x.lower())) # lowercase each word

    data.Fonte = data.Fonte.apply(lambda x: word_tokenize(x))
    source = set([i.lower() for x in data.Fonte for i in x])
    data = data.drop('Fonte', axis = 1)

    list_stopwords = list(set(stopwords.words('italian')))
    data.Job_Description = data.Job_Description.apply(lambda x: [i for i in x if i not in list_stopwords and i !='.'])

    words = []
    for i in data.Job_Description:
        words.append(i)
    word_list = set([i for x in words for i in x])

    data.Job_Description = data.Job_Description.apply(lambda x: [i for i in x if len(i) >3])
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


def uncommon_words(data,column, italian_vocab, english_vocab = None):
    ''' remove uncommon and misspelled words '''
    data[column] = data[column].apply(lambda x: [i for i in x if i in italian_vocab])
    return data


#### Data Split ####
def split_data(data, text_column, target_column):
    ''' function to split data for training and evaluation '''
    X_train, X_test, y_train, y_test = train_test_split(data[text_column], 
                                                    data[target_column])
    print('Number of rows in the total set: {}'.format(data.shape[0]))
    print('Number of rows in the training set: {}'.format(X_train.shape[0]))
    print('Number of rows in the test set: {}'.format(X_test.shape[0]))
    return  X_train, X_test, y_train, y_test

#### Bag of Words ####
def bag_of_words(X_train, X_test):
    ''' function to create bag of words '''
    count_vector = CountVectorizer()
    # Fit the training data and then return the matrix
    training_data = count_vector.fit_transform(X_train)

    # Transform testing data and return the matrix. 
    # Note we are not fitting the testing data into the CountVectorizer()
    testing_data = count_vector.transform(X_test)

    print('training data',training_data.shape)
    print('testing data',testing_data.shape)
    return training_data, testing_data
