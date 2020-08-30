from fer import FER
from cv2 import cv2
import os

<<<<<<< HEAD
def run_FER(path):
    img = cv2.imread(path)
    image_name = os.path.basename(path)
    detector = FER()
    emotion_label = detector.detect_emotions(img)
    
    return {image_name : emotion_label}
=======
class FER_Wrapper():
    def __init__(self):
        self.detector = FER()
        
    def run_FER(self, path):
        detector = self.detector
        img = cv2.imread(path)
        image_name = os.path.basename(path)
        emotion_label = detector.detect_emotions(img)
        if len(emotion_label) == 0:
            emotion_label = 0

        return {image_name : emotion_label}
>>>>>>> 6b6dbf0dff5de2b633cb72bd8a758f89c0a30751
