import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def get_data_set(image):
    print("generate dataset...")
    contours, hireachy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # get all digits
    rois = []
    for c in range(len(contours)):
        box = cv.boundingRect(contours[c])
        if box[3] < 10:
            continue
        rois.append(box)

    # sort(box)
    num = len(rois)
    for i in range(num):
        for j in range(i+1, num, 1):
            x1, y1, w1, h1 = rois[i]
            x2, y2, w2, h2 = rois[j]
            if x2 < x1:
                temp = rois[j]
                rois[j] = rois[i]
                rois[i] = temp

    bgr = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    index = 0
    digit_data = np.zeros((num, 28*48), dtype=np.float32)
    for x, y, w, h in rois:
        cv.rectangle(bgr, (x, y), (x+w, y+h), (0, 0, 255), 2, 8)
        cv.putText(bgr, str(index), (x, y+10), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1)
        digit = image[y:y+h,x:x+w]
        img = cv.resize(digit, (28, 48))
        row = np.reshape(img, (-1, 28 * 48))
        digit_data[index] = row
        index += 1
    cv.imshow("split digits", bgr)
    return digit_data, rois


def split_lines(image):
    print("start to analysis text layout...");
    # Y-Projection
    h, w = image.shape
    hist = np.zeros(h, dtype=np.int32)
    for i in range(h):
        for c in range(w):
            pv = image[i, c]
            if pv == 255:
                hist[i] += 1

    # x = np.arange(h)
    # plt.bar(x, height=hist)
    # plt.show()

    # find lines
    hist[np.where(hist>5)] = 255
    hist[np.where(hist<=5)] = 0

    text_lines = []
    found = False
    count = 0
    start = -1
    for i in range(h):
        if hist[i] > 0 and found is False:
            found = True
            start = i
            count += 1
        if hist[i] > 0 and found is True:
            count += 1
        if hist[i] == 0 and found is True:
            found = False
            text_lines.append(image[start-2:start+count+2, 0:w])
            start = -1
            count = 0

    if found is True:
        text_lines.append(image[start - 2:start + count + 2, 0:w])
    print(len(text_lines))
    return text_lines