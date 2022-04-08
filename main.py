from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

faceClassifier = cv2.CascadeClassifier(r'C:\Users\Dell\Desktop\Natalija\proektna\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\Dell\Desktop\Natalija\proektna\cvmodel.h5')

emotionLabels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

vid = cv2.VideoCapture(0)


while True:
    _, frame = vid.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(gray)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roiGray = gray[y:y+h, x:x+w]
        roiGray = cv2.resize(roiGray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roiGray]) != 0:
            roi = roiGray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotionLabels[prediction.argmax()]
            labelPosition = (x, y-10)
            cv2.putText(frame, label, labelPosition, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(frame, 'No faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
vid.release()
cv2.destroyAllWindows()
