import numpy as np
import pickle
from sklearn import preprocessing

#load and standardize data
cond = open("condition", "r")
X = np.loadtxt(cond, usecols=range(6))
X_std = preprocessing.scale(X)

#load trained model
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

#test random sample
Y_predictions = pickle_model.predict(X_std)

#output result
np.savetxt('output_predictions', Y_predictions) 
