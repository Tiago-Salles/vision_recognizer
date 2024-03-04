import cv2
import numpy as np
import keras
from keras.applications.mobilenet import preprocess_input, decode_predictions

model = keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    preprocessed_frame = preprocess_image(frame)
    
    predictions = model.predict(preprocessed_frame)
    label = decode_predictions(predictions)[0][0][1]

    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Image Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()