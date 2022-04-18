import sklearn
from glob import glob
import numpy as np
import joblib
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer

# Load training data for fitting the PCA object
X_train = []
y_train = []
folders = glob("Digit_training/*")
for folder in folders:
    label = float(folder.split("\\")[-1])
    images = glob( folder + "/*" )
    for image in images:
        # Open image and calculate fourier transform
        im = Image.open(image)
        arr = np.array(im)
        ft = np.fft.fft2(arr)
        ft = np.real(ft)
        ft = ft.flatten()
        X_train.append(ft)
        y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = sklearn.preprocessing.minmax_scale(X_train, feature_range=(0, 1))

# Load testing data
X_test = []
y_test = []
folders = glob("Digit_testing/*")
for folder in folders:
    label = float(folder.split("\\")[-1])
    images = glob( folder + "/*" )
    for image in images:
        im = Image.open(image)
        arr = np.array(im)
        ft = np.fft.fft2(arr)
        ft = np.real(ft)
        ft = ft.flatten()
        X_test.append(ft)
        y_test.append(label)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = sklearn.preprocessing.minmax_scale(X_test, feature_range=(0, 1))

# - - - - - - Test SVM poly kernal with 784 feature vectors.

# Predict with the regular svm classifier
svm = joblib.load('lib/svm.joblib')

start = timer()
y_pred = svm.predict(X_test)
end = timer()

# Stats for SVM model
print("784 feature model stats: ")
print( "Accuracy", accuracy_score(y_test, y_pred) * 100, "%" )
print("Time to predict", len(X_test), "numbers:", (end-start), "seconds.")
print()
# - - - - - - Test SVM poly kernel with pca applied to the data, 196 feature vectors.

# Fit the pca object with the training data
pca = PCA(n_components=196)
pca.fit(X_train)

# Then use it to transform the testing data
X_test = pca.transform(X_test)

# Predict
svm = joblib.load('lib/svmpca.joblib')

start = timer()
y_pred = svm.predict(X_test)
end = timer()

# Stats for PCA SVM model
print("196 feature model stats: ")
print("Accuracy", accuracy_score(y_test, y_pred) * 100, "%")
print("Time to predict", len(X_test), "numbers:", (end-start), "seconds.")