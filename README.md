Problem Statement:
Gender Classification from Handwriting

Background:
We are using machine learning to classify different gender from Handwriting.

Dataset:
We were provided with the dataset HHD_gender. The images in the dataset is writing in hebrew.

Technologies:
The project uses Python >= 3.6

Other technologies used:
OpenCV, NumPy, Panda, SVM, LBP

Mathematical Aspects
1. Local Binary Patterns (LBP)
LBP is an effective texture pattern descriptor introduced by Ojala et al. to describe the local texture patterns of an image. 
It is widely used in the applications based on image processing. 

2. Support Vector Machine (SVM)
The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.
It is using the model train-valid-test.

3. GridSearchCV
It runs through all the different parameters that is fed into the parameter grid and produces the best combination of parameters, 
based on a scoring metric of your choice (accuracy, f1, etc).



Setup
- pip install -r requirements.txt
- pip install opencv
- pip install numpy
- pip install sys
- pip install skimage
- pip install sklearn

Results:
Here are the results on the for the algorithm for the dataset HHD_gender:
number of points :15
radius :3
Accuracy : 68.57142857142857%
Confusion matrix : [[27  8]
                    [14 21]]




