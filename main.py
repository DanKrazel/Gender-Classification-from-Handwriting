import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import feature
import sys
import os
import pandas as pd
from skimage.transform import resize
from skimage.io import imread
from LocalBinaryPattern import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import argparse
import cv2
import os






def convertToGray(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImage

def compute_LBP(path_train, path_test):
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
    desc = LocalBinaryPatterns(24, 8)
    train_data_arr = []
    train_labels_arr = []
    test_data_arr = []
    test_labels_arr = []


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
    df['Target'] = train_label
    x_train = df.iloc[:, :-1]  # input data
    y_train = df.iloc[:, -1]  # output data

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['poly','linear']}
    svc = svm.SVC(probability=True)
    model = GridSearchCV(svc, param_grid)
    model.fit(x_train, y_train)

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
            test_labels_arr.append(Categories.index(i))

    test_data = np.array(test_data_arr)
    test_label = np.array(test_labels_arr)
    df = pd.DataFrame(test_data)  # dataframe
    df['Target'] = test_label
    x_test = df.iloc[:, :-1]  # input data
    y_test = df.iloc[:, -1]  # output data
    prediction = model.predict(x_test)
    #print("Accuracy: {}%".format(model.score(test_data, test_labels) * 100))
    print(f"The model is {accuracy_score(prediction, y_test) * 100}% accurate")
    matrix = confusion_matrix(y_test, prediction)
    print('Confusion matrix : \n', matrix)
    # display the image and the prediction
    #print(classification_report(test_labels, model.predict(test_data)))

        #cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)


def compute_SVM(img, datadir):
    Categories = ['male', 'female']
    flat_data_arr = []  # input array
    target_arr = []  # output array
    # path which contains all the categories of images
    for i in Categories:
        print(f'loading... category : {i}')
        path = os.path.join(datadir, i)
        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (1500, 900, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
        print(f'loaded category:{i} successfully')
    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)  # dataframe
    df['Target'] = target
    x = df.iloc[:, :-1]  # input data
    y = df.iloc[:, -1]  # output data

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
    svc = svm.SVC(probability=True)
    model = GridSearchCV(svc, param_grid)




def main():
    #input = cv2.imread(sys.argv[1])
    path_train = "images/train"
    path_test = "images/test"
    datadir = 'images/train'
    result = compute_LBP(path_train, path_test)
    #print(result)
    #cv2.imwrite(output, result)





if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
