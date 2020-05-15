# Emotion_Recognizer

This repo comtains a program that tells predicts the user's mood/emotion based on the facial expression using the webcam.
<br>
It is trained with the data set provided by: 
<br>https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
<br>
The Harrcascade File can be downloaded from the floowing link:
<br>https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

In order to run the program, first run the emotion_test.py file then run the emotion_analyzer.py file.

```
python3 emotion_test.py 
python3 emotion_analyzer.py
```

With the training data, the neural network was able to predict with a 90% accuracy. With a bit more reasearch on methods to rescale data, and other techniques, the numbers can be imporved.
