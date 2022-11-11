import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
import time
import csv

#Importing model and getting all the images from data folder
new_model = load_model('mymodel.h5')
files = os.listdir(str(os.getcwd()) + "/data")
header = ['filename', 'sort_result']

#Variable for storing the current number of processed images
cnt = 1

#Variable for measuring the time that program takes
start = time.time()

#Opening csv file
with open('output.csv', 'w') as file:

    writer = csv.writer(file)

    writer.writerow(header)

    for element in files:

        try:
            #Getting the current file
            filename = str(os.getcwd()) + "/data/" + element

            img = cv2.imread(filename)

            #Resizing it
            resize = tf.image.resize(img, (256,256))

            #Getting the prediction
            pred = new_model.predict(np.expand_dims(resize/255, 0))
            if pred[0][0] > 0.5:
                writer.writerow([element, 'p'])
            else:
                writer.writerow([element, 'n'])
            print(cnt)
            cnt += 1
        except Exception as e:
            print(e)
            print('Occured with file' + element)

end = time.time()

print('Estimated execution time: ', end - start, 'seconds')
