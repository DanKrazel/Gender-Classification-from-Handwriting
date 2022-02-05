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






def convertToGray(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImage

def hog_features(img):
    data_gray = img
    ppc = 16
    hog_images = []
    hog_features = []
    for image in data_gray:
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4), block_norm='L2')
        hog_images.append(hog_image)
        hog_features.append(fd)

    hog_features = np.array(hog_features)
    return hog_features

def compute_LBP(path_train, path_valid, path_test):
    numberOfPoints = 15
    radius = 3
    # construct the argument parse and parse the arguments
    """ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=True,
                    help=path_train)
    ap.add_argument("-e", "--testing", required=True,
                    help=path_test)
    args = vars(ap.parse_args())
    print(args)"""
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
            #data_frame = hog_features(gray)
            #hist = data_frame
            hist = desc.describe(gray)
            #print("lbp: ",hist)
            #print("hog: ",data_frame)
            # width, height = img_float32.size
            # print(width, height)
            # extract the label from the image path, then update the
            # label and data lists
            #train_labels.append(imagePath.split(os.path.sep)[-2])
            train_data_arr.append(hist)
            train_labels_arr.append(Categories.index(i))
            # train a Linear SVM on the data

    train_data = np.array(train_data_arr)
    train_label = np.array(train_labels_arr)
    df = pd.DataFrame(train_data)  # dataframe
    #print("df :")
    df['Target'] = train_label
    #print(df)
    #print(df.head())
    print("x_train :")
    x_train = df.iloc[:, :-1]
    print(x_train)
    y_train = df.iloc[:, -1]  # output data
    print("y_train :")
    print(y_train)
    #x_train = df.iloc[:, :-1]  # input data
    #y_train = df.iloc[:, -1]  # output data

    #print(train_data)
    #print(train_label)
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['poly', 'linear', 'rbf', 'sigmoid']}
    svc = svm.SVC()
    model = GridSearchCV(svc, param_grid)
    #print(model.best_params_)
    model.fit(x_train, train_label)
    print(model.best_params_)
    print('The Model is trained well with the given images')
    #y_pred = model.predict(x_test)
    #print("The predicted Data is :")
    #print(y_pred)
    #print("The actual data is:")
    #print(np.array(y_test))

    # loop over the valid images
    print("Compute validation")
    for i in Categories:
        print(f'loading... category : {i}')
        path = path_valid + "/" + i
        for imagePath in paths.list_images(path + "/"):
            image = cv2.imread(imagePath)
            img_resized = resize(image, (1500, 900, 3))
            img_float32 = np.float32(img_resized)
            gray = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            test_data_arr.append(hist)
            test_labels_arr.append(Categories.index(i))
            #prediction = model.predict(hist.reshape(1, -1))
            #predictions_arr.append(prediction[0])

    test_data = np.array(test_data_arr)
    test_label = np.array(test_labels_arr)
    df = pd.DataFrame(test_data)  # dataframe
    #print("df :")
    df['Target'] = test_label
    # print(df)
    #print(df.head())
    #print("x_train :")
    x_test = df.iloc[:, :-1]  # input data
    y_test = df.iloc[:, -1]  # output data
    print("x_test :")
    print(x_test)
    print("y_test :")
    print(y_test)
    #predictions_arr = model.predict(train_data)
    predictions = model.predict(x_test)
    #print("Accuracy: {}%".format(model.score(test_data, test_labels) * 100))
    #print(test_label)
    #print(predictions)
    #print(f"Accuracy : {accuracy_score(predictions, y_test) * 100}%")
    #print("Accuracy: {}%".format(model.score(predictions, y_test) * 100))
    matrix = confusion_matrix(test_label, predictions)
    print('Confusion matrix : \n', matrix)
    print('Classification report : \n', classification_report(y_test,predictions))

    f = open("result.txt", "w+")
    f.write(f"number of points :{numberOfPoints}\n")
    f.write(f"radius :{radius}\n")
    f.write(f"Accuracy : {accuracy_score(predictions, y_test) * 100}%\n")
   # for line in matrix:
            #np.savetxt(f, line, fmt='%.2f')
    f.write(f"Confusion matrix : {matrix}\n")

    f.close()

    # display the image and the prediction
    #print(classification_report(test_labels, model.predict(test_data)))"""

        #cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)

def compute_test(path_test, Categories):
    # loop over the testing images
    print("Compute test")
    for i in Categories:
        print(f'loading... category : {i}')
        path = path_test + "/" + i
        #path = os.path.join(path_test, i)
        for imagePath in paths.list_images(path + "/"):
            # load the image, convert it to grayscale, describe it,
            # and classify it
            image = cv2.imread(imagePath)
            img_resized = resize(image, (1500, 900, 3))
            img_float32 = np.float32(img_resized)
            gray = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            #test_labels.append(imagePath.split(os.path.sep)[-2])
            test_data_arr.append(hist)
            test_labels_arr.append(i)
            #test_data = np.array(test_data_arr)
            #test_label = np.array(test_labels_arr)
            #df = pd.DataFrame(test_data)  # dataframe
            #df['Target'] = test_label
            #x_test = df.iloc[:, :-1]  # input data
            #y_test = df.iloc[:, -1]  # output data
            prediction = model.predict(hist.reshape(1, -1))
            predictions_arr.append(prediction[0])
            # display the image and the prediction
            #print(prediction[0])
            #cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        #1.0, (0, 0, 255), 3)
            #cv2.imshow("Image", image)
            #cv2.waitKey(0)

def main():
    #input = cv2.imread(sys.argv[1])
    path_train = "images/train"
    path_test = "images/test"
    path_valid = "images/valid"
    datadir = 'images/train'
    result = compute_LBP(path_train, path_valid, path_test)
    #print(result)
    #cv2.imwrite(output, result)





if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
