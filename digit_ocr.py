import cv2 as cv
import numpy as np


class DigitNumberRecognizer:

    def __init__(self):
        print("create object...")
        self.svm = cv.ml.SVM_load("svm_data.yml")

    def predict(self, data_set):
        result = self.svm.predict(data_set)[1]
        text = ""
        for i in range(len(result)):
            text += str(np.int32(result[i][0]))
        print(text)
        return text