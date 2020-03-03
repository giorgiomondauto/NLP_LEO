from utils import get_key

def prediction(model, testing_data, Role_dictionary, y_test = None):
    ''' model prediction '''
    predictions = model.predict(testing_data)
    predictions_keys = []
    for i in predictions:
        predictions_keys.append(get_key(i,Role_dictionary))
    actual_predictions = []
    if y_test != None:
        for i in y_test.tolist():
            actual_predictions.append(get_key(i,Role_dictionary))
    return predictions, predictions_keys, actual_predictions