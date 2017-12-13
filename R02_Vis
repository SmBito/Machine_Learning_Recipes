##1.Import Dataset:

from sklearn.datasets import load_iris
iris= load_iris() #Load and return the iris dataset (classification).

#print(iris.feature_names)
#print(iris.target_names)
#print(iris.data[0])
#print(iris.target[0])


##Printing the entire data set
# for i in range(len(iris.target)):
#    print("Example %d: label %s, featutrs %s" % (i, iris.target[i],iris.data[i]))


##2.Train a Classifier
##first split up the data (to get testing data)
import numpy as np
test_idx=[0,50,100]
    
##Training Data
train_target=np.delete(iris.target, test_idx)
train_data=np.delete(iris.data, test_idx, axis=0)

##Testing Data
test_target=iris.target[test_idx]
test_data=iris.data[test_idx]

##3. Predict label for new flower
from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)


print(test_target)
print(clf.predict(test_data))


##4. Viz Code
from subprocess import call
from sklearn.datasets import load_iris
from sklearn import tree
iris = (load_iris())
clf = (tree.DecisionTreeClassifier())
clf = (clf.fit(iris.data, iris.target))

tree.export_graphviz(clf,
out_file='irisTree.dot',  feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)
call(["dot", "-Tpdf", "irisTree.dot", "-o", "irisTree.pdf"])
