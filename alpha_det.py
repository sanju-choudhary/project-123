import cv2
import numpy as np
import pandas as pd
#from sklearn.datasets import fetch_openml
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

# Reading the data
X = np.load("image.npz")["arr_0"]
Y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(Y).value_counts())
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
n_classes = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 8, test_size = 0.25)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

# Fitting the training data into model 
# It will take a minute or two
clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Starting the camera
cap = cv2.VideoCapture(0)
i = 0 
while (i == 0):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Drawing a box in the center of the video
        height, width = gray.shape
        upper_left = (int(width/2-56), int(height/2-56))
        bottom_right = (int(width/2 + 56), int(height/2 + 56))

        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        # To only consider the are inside the box for detecting the digits
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        # converting cv2 image to PIL formate
        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert("L")
        image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)

        # Inverting the Image
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
        test_predicted = clf.predict(test_sample)
        print("Predicted class is : ", test_predicted)

        # Displaying the result in frame
        cv2.imshow("Frame", gray)
        if(cv2.waitKey(1) & 0xFF == ord("q")):
            break
    except Exception as e:
        pass 

cap.release()
cv2.destroyAllWindows()