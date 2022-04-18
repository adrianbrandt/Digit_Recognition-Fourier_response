import sklearn
from sklearn.svm import SVC
import numpy as np
import joblib
from glob import glob
from PIL import Image
from sklearn.decomposition import PCA
from timeit import default_timer as timer

# Load training data and
X_train = []
y_train = []
folders = glob("Digit_training/*")
for folder in folders:
    label = float(folder.split("\\")[-1])
    images = glob( folder + "/*" )
    for image in images:
        im = Image.open(image)
        arr = np.array(im)
        ft = np.fft.fft2(arr) # Extract fourier transform
        ft = np.real(ft) # Get the real part
        ft = ft.flatten() # Flatten the matrix
        X_train.append(ft)
        y_train.append(label)

# Convert data to numpy array and scale between 0 - 1
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = sklearn.preprocessing.minmax_scale(X_train, feature_range=(0, 1))

# Apply pca by doing dimentionality reduction to all features into the first 196 features.
pca = PCA(n_components=196)
X_train = pca.fit_transform(X_train)
clf = SVC( kernel="poly", degree=1, C=1 )

# Train the SVM
start = timer()
clf.fit( X_train, y_train )
end = timer()
print("Training took: ", end - start, "seconds.")
# Save model
joblib.dump(clf, 'lib/svmpca.joblib')