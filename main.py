import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import feature
import sys
import os
import pandas as pd
from skimage.transform import resize
from skimage.io import imread
from skimage.feature import hog
from LocalBinaryPattern import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import argparse
import cv2
import os



def compute_LBP(path_train, path_valid, path_test):
    numberOfPoints = 15
    radius = 3
    # initialize the local binary patterns descriptor along with
    # the data and label lists
    desc = LocalBinaryPatterns(numberOfPoints, radius)
    train_data_arr = []
    train_labels_arr = []
    test_data_arr = []
    test_labels_arr = []
    predictions_arr = []


    #train image
    #Initialise labels
    Categories = ['male', 'female']
    for i in Categories:
        print(f'loading... category : {i}')
        path = path_train + "/" + i
        #path = os.path.join(path_train, i)
        print(path)
        for imagePath in paths.list_images(path + "/"):
	        # load the image, convert it to grayscale, and describe it
            image = cv2.imread(imagePath)
            img_resized = resize(image, (1500, 900, 3))
            img_float32 = np.float32(img_resized)
            gray = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            train_data_arr.append(hist)
            train_labels_arr.append(Categories.index(i))
            # train a Linear SVM on the data

    train_data = np.array(train_data_arr)
    train_label = np.array(train_labels_arr)
    df = pd.DataFrame(train_data)  # dataframe
    df['Target'] = train_label
    x_train = df.iloc[:, :-1]
    y_train = df.iloc[:, -1]  # output data
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['poly', 'linear', 'rbf', 'sigmoid']}
    svc = svm.SVC()
    model = GridSearchCV(svc, param_grid)
    model.fit(x_train, y_train)
    print(model.best_params_)
    print('The Model is trained well with the given images')


    # loop over the valid images
    print("Compute validation")
    for i in Categories:
        print(f'loading... category : {i}')
        path = path_test + "/" + i
        for imagePath in paths.list_images(path + "/"):
            image = cv2.imread(imagePath)
            img_resized = resize(image, (1500, 900, 3))
            img_float32 = np.float32(img_resized)
            gray = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            test_data_arr.append(hist)
            test_labels_arr.append(Categories.index(i))

    test_data = np.array(test_data_arr)
    test_label = np.array(test_labels_arr)
    df = pd.DataFrame(test_data)  # dataframe
    df['Target'] = test_label
    x_test = df.iloc[:, :-1]  # input data
    y_test = df.iloc[:, -1]  # output data
    predictions = model.predict(x_test)
    print(f"Accuracy : {accuracy_score(predictions, y_test) * 100}%")
    matrix = confusion_matrix(y_test, predictions)
    print('Confusion matrix : \n', matrix)
    print('Classification report : \n', classification_report(y_test,predictions))

    f = open("result.txt", "w+")
    f.write(f"number of points :{numberOfPoints}\n")
    f.write(f"radius :{radius}\n")
    f.write(f"Accuracy : {accuracy_score(predictions, y_test) * 100}%\n")
    f.write(f"Confusion matrix : {matrix}\n")

    f.close()



def main():

    path_train = sys.argv[1]
    path_valid = sys.argv[2]
    path_test = sys.argv[3]
    result = compute_LBP(path_train, path_valid, path_test)






if __name__ == '__main__':
    main()


