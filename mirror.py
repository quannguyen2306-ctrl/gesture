import cv2
import numpy as np

def mirror_frame(frame): 
    (h, w, _) = frame.shape
    left_half = frame[:, :h // 2]
    right_half = frame[:, w // 2:]
    mirrored_frame = cv2.hconcat([left_half, cv2.flip(left_half, 1)])
    return mirrored_frame