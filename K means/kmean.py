import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)
print(train.head(3))
print(test.head(3))

train = train.drop(['Name','Ticket', 'Cabin','Embarked','Sex'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked','Sex'], axis=1)

X = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])

kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
kmeans.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("Accuracy :" + str(correct/len(X)))
