#Script to load the model

import pickle

with open('model1.bin', 'rb') as model_file:  
    model = pickle.load(model_file)
model_file.close()

with open('dv.bin', 'rb') as dic_file:  
    dict_vectorizer = pickle.load(dic_file)
dic_file.close()

client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

#Turn the client into a feature matrix
X = dict_vectorizer.transform([client])

#Score client
probability = model.predict_proba(X)[0,1]

#Print results
print(f"The probability that client will get a credit card if {probability}")