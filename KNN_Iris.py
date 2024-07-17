import numpy as np  
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split   
from sklearn import metrics 
# Define column names 
names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Class'] 
# Read dataset to pandas dataframe, skipping the header row 
dataset = pd.read_csv("C:\\Users\\prasa\\Downloads\\Iris.csv", names=names, header=0) 
# Extract features (X) and target variable (y) 
X = dataset.drop(columns=['Class'])   
y = dataset['Class'] 
# Split data into training and testing sets 
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.10)  
# Initialize and train the KNN classifier 
classifier = KNeighborsClassifier(n_neighbors=5).fit(Xtrain, ytrain)  
# Make predictions 
ypred = classifier.predict(Xtest) 
# Print original labels, predicted labels, and whether the prediction was correct or wrong 
print("\n-------------------------------------------------------------------------") 
print ('%-25s %-25s %-25s' % ('Original Label', 'Predicted Label', 'Correct/Wrong')) 
print ("-------------------------------------------------------------------------") 
for label, pred_label in zip(ytest, ypred): 
    correct = 'Correct' if label == pred_label else 'Wrong' 
    print ('%-25s %-25s %-25s' % (label, pred_label, correct)) 
print ("-------------------------------------------------------------------------") 
# Print confusion matrix and classification report 
print("\nConfusion Matrix:\n", metrics.confusion_matrix(ytest, ypred))   
print ("-------------------------------------------------------------------------") 
print("\nClassification Report:\n", metrics.classification_report(ytest, ypred))  
print ("-------------------------------------------------------------------------") 
# Print accuracy of the classifier 
print('Accuracy of the classifier: %.2f' % metrics.accuracy_score(ytest, ypred)) 
print ("-------------------------------------------------------------------------") 
