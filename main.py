import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib
import re
import nltk
import time
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('words')
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer



start_time = time.time()
professioni = pd.read_csv('data/professioni.csv')
dati_aprile = pd.read_csv('Job_vacancies_aprile_completo.csv', skiprows=1, usecols = [1,2,3],\
                          names = ['Target','Sub_Role','Job_Description'])
print('Dataset shape {}'.format(dati_aprile.shape))

print('I am starting the Preprocessing step')
professioni.Subgroup = professioni.Subgroup.apply(lambda x: x.lower().split('\n'))
professioni.subgroup1 = professioni.subgroup1.apply(lambda x: x.lower().split('\n'))
professioni.subgroup2 = professioni.subgroup2.apply(lambda x: x.lower().split('\n'))
professioni.Subgroup = professioni.Subgroup + professioni.subgroup1 + professioni.subgroup2
professioni.Subgroup = professioni.Subgroup.apply(lambda x: list(set(x)))

professioni_dictionary = pd.Series(professioni.Subgroup.values,index=professioni.Group).to_dict()
# to remove empty space from the subgroup list of values
for i in range(0,professioni.Subgroup.shape[0]):
    while("" in professioni.Subgroup.iloc[i]) : 
        professioni.Subgroup.iloc[i].remove("") 

subgroup_dict = {}
for group, subgroups in professioni_dictionary.items():
    for subgroup in subgroups:
        subgroup_dict[subgroup] = group

# remove punctuation
dati_aprile.Job_Description = dati_aprile.Job_Description.apply(lambda x: re.sub('[^a-zA-Z]',' ',x))
dati_aprile.Target = dati_aprile.Target.apply(lambda x: x.lower())
dati_aprile.Job_Description = dati_aprile['Job_Description'].apply(lambda x: word_tokenize(x.lower())) 

# remove words with lenght < 3 (e.g. numbers and not meaningful words)
dati_aprile.Job_Description = dati_aprile.Job_Description.apply(lambda x: [i for i in x if len(i) >3])

# remove in 'job_description' words from 'fonte'
Fonti = ['randstad', 'monster', 'infojob', 'technical', 'kelly', 'services', 'italia', 'lavoropi',\
             'quanta','vimercate','temporary','openjobmetis','agenzia']
dati_aprile.Job_Description = dati_aprile.Job_Description.apply(lambda x: [i for i in x if i not in Fonti])

# Stopwords are words generally not relevant to a text; thereby we get rid of them
list_stopwords = list(set(stopwords.words('italian')))
dati_aprile.Job_Description = dati_aprile.Job_Description.apply(lambda x: [i for i in x if i not in list_stopwords] )

# load a text file containing a list of 'Comuni, Provincie, Regioni' italiane
geo_data = pd.read_fwf('data/listacomuni.txt')
geo_data.to_csv('data/listacomuni.csv')
geo_data = pd.read_csv('data/listacomuni.csv', sep=';')

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

dati_aprile = remove_comuni(dati_aprile, 'Job_Description', geo_data)


# Get rid of unusual_words : misspelled words - uncommon words

italian_vocab = pd.read_fwf('data/660000_parole_italiane.txt', names = ['word'])
italian_vocab = set(italian_vocab.word.tolist())
english_vocab = sorted(set(w.lower() for w in nltk.corpus.words.words())) # english vocabulary

def uncommon_words(data,column, italian_vocab, english_vocab = None):
    data[column] = data[column].apply(lambda x: [i for i in x if i in italian_vocab])
    return data

dati_aprile = uncommon_words(dati_aprile, 'Job_Description', italian_vocab, english_vocab)

#Create Role dictionary and encode the column Role
Role_dictionary = pd.Series(dati_aprile['Target'].unique()).to_dict()
Role_dictionary = dict([(value, key) for key, value in Role_dictionary.items()])

role_encoded = []
for i in dati_aprile.Target:
    role_encoded.append(Role_dictionary.get(i, None))

dati_aprile['Multi_Class'] = role_encoded

dati_aprile.Job_Description = dati_aprile.Job_Description.apply(lambda x: ' '.join(x))

print(20*'#')
print('I am splitting the data')
# split of dataset
X_train, X_test, y_train, y_test = train_test_split(dati_aprile['Job_Description'], 
                                                    dati_aprile['Multi_Class'],
                                                   test_size=0.20, random_state=42)
                                                   

print(20*'#')
print('I am creating the Bag of Words')
# Bag of Words processing to our dataset
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

print(20*'#')
print('training data',training_data.shape)
print('testing data',testing_data.shape)

# Define the Model
print(20*'#')
print('I am initializing the Bayes Model')
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

def get_key(value, dictionary):
    for k,v in dictionary.items():
        if value == v:
            return k

predictions = naive_bayes.predict(testing_data)
predictions_keys = []
for i in predictions:
    predictions_keys.append(get_key(i,Role_dictionary))

actual_predictions = []
for i in y_test.tolist():
    actual_predictions.append(get_key(i,Role_dictionary))

# check the accuracy of our model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(20*'#')
print('Test Set looks something like: \n', X_test.iloc[:5])
print(20*'*')
print('Predictions on those examples: \n ',actual_predictions[:5])
print(20*'*')
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print(20*'#')
print('Time to run the script: {}'.format(round(time.time() - start_time,4)))