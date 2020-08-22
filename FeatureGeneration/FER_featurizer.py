from fer import FER
from cv2 import cv2
import os

def run_FER(path):
    img = cv2.imread(path)
    image_name = os.path.basename(path)
    detector = FER()
    emotion_label = detector.detect_emotions(img)
    
    return {image_name : emotion_label}