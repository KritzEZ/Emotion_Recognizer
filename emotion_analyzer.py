import os
import cv2
import numpy as np
from pathlib import Path
from keras.models import model_from_json
from keras.preprocessing import image

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

#Loading json file with model strusture
f = Path("model_structure.json")
model_structure = f.read_text()

#Creating Keras model object
#model = model_from_json(open("model_structure.json", "r").read())
model = model_from_json(model_structure)

#Loading model's trained weights from emotion_test.py
model.load_weights("model_weights.h5")

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)

while True:

    retval, picture = vid.read() #Reading image --> retval holds boolean for img read and picture holds image
    if not retval:
        continue
    grayscaled = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY) #Grayscaling the image
    face_detect = face_haar_cascade.detectMultiScale(grayscaled, 1.32, 5) #Finding the face in the picture

    for (x_val, y_val, width, height) in face_detect:
        #cv2.rectangle(picture,(x_val, y_val),(x_val+width, y_val+height),(255,0,0),thickness=5) #Creates rectangle around the face
        roi_gray=grayscaled[y_val:y_val+width,x_val:x_val+height] #Crops and resizes the image to 48X48 ppixels becuase test images with 48X48 pixels
        roi_gray=cv2.resize(roi_gray,(48,48)) #Resizes the image to 48X48 ppixels becuase test images with 48X48 pixels
        img_to_pixels = image.img_to_array(roi_gray) #Puts image pizel values to array
        img_to_pixels = np.expand_dims(img_to_pixels, axis = 0) #Adding 4th dimension (Keras takes a list of images, not only one image)
        img_to_pixels /= 255 #Normalizing the pixles

        emotion_predict = model.predict(img_to_pixels) #Sending image to the model to predict

        max_index = np.argmax(emotion_predict[0]) #Getting the highest possibile emotion
        predicted_emotion = emotions[max_index] #Finding the emotion in emotion array

        cv2.putText(picture, predicted_emotion, (int(x_val), int(y_val)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) #Puts the text of emotion on top of rectangle

    resized_img = cv2.resize(picture, (1000, 700)) #Window size of camera
    cv2.imshow('Emotion Recognizer',resized_img)

    if cv2.waitKey(10) == ord('q'):#Press 'q' to exit window
        break

vid.release()
cv2.destroyAllWindows    

