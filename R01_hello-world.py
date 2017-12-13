from sklearn import tree

#Training Data
#smooth=1 / bumpy=0
features = [[140,1],[130,1],[150,0],[170,0]]
# appple=0 / orange=1
labels = [0,0,1,1]

#Training Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

#Make Predictions
print(clf.predict([[162, 0]]))
print(clf.predict([[102, 1]]))
