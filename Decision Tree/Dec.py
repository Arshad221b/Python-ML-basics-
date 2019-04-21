import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split



data = pd.read_csv("diabetes.csv")

print(data.head(3))

X = data.drop(['Outcome'],axis = 1)
Y = data['Outcome'].values

X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 0)

dtree = tree.DecisionTreeClassifier(criterion = 'entropy' , max_depth = 4, random_state  = 0 )
dtree.fit(X_train,Y_train)

Y_pred = dtree.predict(X_test)

misclassified = (Y_test != Y_pred).sum()
accuracy = metrics.accuracy_score(Y_test,Y_pred)
print("Accuracy :" + str(accuracy*100) )

