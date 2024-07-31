import cv2
import tensorflow as tf
from tensorflow import keras

#Initialize the video capture
cap = cv2.VideoCapture(0)

# Assuming you have a face detection model loaded as 'face'
face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Placeholder functions for predicting gender, emotion, and age
def predict_gender(img):
    model = keras.models.load_model('gender_detection.keras')

    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = img.reshape(1, 48, 48, 3)
    class_names = ['female', 'male']

    return class_names[tf.argmax(model(img), axis=-1).numpy()[0]]
def predict_emotion(img):

    model = keras.models.load_model('my_model2.keras')

    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = img.reshape(1, 48, 48, 3)
    class_names = ['angry', 'happy', 'sad']

    return class_names[tf.argmax(model(img), axis=-1).numpy()[0]]

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = face.detectMultiScale(gr, 1.3, 5)

        for (x, y, w, h) in f:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Extract the face region
            face_img = frame[y:y+h, x:x+w]

            # Predict gender, emotion, and age
            gender = predict_gender(face_img)
            emotion = predict_emotion(face_img)

            # Create the text to display
            text = f"{gender}, {emotion}"

            # Put the text below the rectangle
            cv2.putText(frame, text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
