from fer import FER
from cv2 import cv2
import os

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