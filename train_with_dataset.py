# now loading the data we trained in the classifer
import pickle 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

data_dict = pickle.load(open('/Users/crinitinikos/Desktop/summer coding porjects/sign language detector/data/data.pickle', 'rb')) #read bytes

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
#currenly all the data is as a list
# thats why they need to be converted into numpy arrays


#split data into training and test sets

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
''' ^ train test split, splitting the two arrays into seperate sets, and then doing the saem for the labels
 test size is like 20%, 80%, its the amount of the data 
shuffle the data (when training the classfier this is import for the bias
strategfy is giving the same size for the labels in orginal array and new sets
'''

#----------
''' TRAINING THE CLASSIFER '''
# model is the random forest
model = RandomForestClassifier() #--> fast training

# training and fitting it 
model.fit(x_train, y_train)

y_predict = model.predict(x_test) # made the predictions 
#----------

score = accuracy_score(y_predict, y_test)

print('{}% this percentiage is of the classfifers that were correct'.format(score * 100))

# save the model, use this to test the performace of the model
f = open('/Users/crinitinikos/Desktop/summer coding porjects/sign language detector/data/model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

#print(data_dict.keys())
#print(data_dict)
# call the pickle file (that has the dataset)