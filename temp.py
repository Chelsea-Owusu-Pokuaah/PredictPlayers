import numpy as np
import pickle as pk
import sklearn
with open('best_model.pkl', 'rb') as file:
    loaded_model = pk.load(file)
