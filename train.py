import cv2 as cv
import numpy as np
import os


def load_data():
    images = []
    labels = []
    files = os.listdir("D:/python/cv_demo/ocr_demo/digits")
    count = len(files)
    sample_data = np.zeros((count, 28*48), dtype=np.float32)
    index = 0
    for file_name in files:
        file_path = os.path.join("D:/python/cv_demo/ocr_demo/digits", file_name)
        if os.path.isfile(file_path) is True:
            images.append(file_path)
            labels.append(file_name[:1])
            img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (28, 48))
            row = np.reshape(img, (-1, 28*48))
            sample_data[index] = row
            index += 1
    return sample_data, np.asarray(labels, np.int32)


# load data stage
train_data, train_labels = load_data()

# train stage
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
svm.save("svm_data.yml")

svm = cv.ml.SVM_load("svm_data.yml")
result = svm.predict(train_data)[1]
print(result)


