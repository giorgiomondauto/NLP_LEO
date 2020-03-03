from sklearn.naive_bayes import MultinomialNB

def model_training(training_data, y_train):
    ''' train the model '''
    naive_bayes = MultinomialNB()
    naive_bayes.fit(training_data, y_train)
    return naive_bayes

