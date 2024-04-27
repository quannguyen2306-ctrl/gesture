import cv2
import numpy as np



def explode_frame(frame): 
    (h, w, _) = frame.shape
    center_x = w / 2
    center_y = h / 2
    scale_x = 1
    scale_y = 1
    radius = 400
    amount = -0.4
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    delta_x = scale_x * (x_coords - center_x)
    delta_y = scale_y * (y_coords - center_y)
    distance = delta_x ** 2 + delta_y ** 2
    mask = distance < (radius ** 2)
    factor = np.where(mask, np.power(np.sin(np.pi * np.sqrt(distance) / radius / 2), -amount), 1.0)
    flex_x = factor * delta_x / scale_x + center_x
    flex_y = factor * delta_y / scale_y + center_y
    flex_x = flex_x.astype(np.float32)
    flex_y = flex_y.astype(np.float32)
    dst = cv2.remap(frame, flex_x, flex_y, cv2.INTER_LINEAR)
    
    return dst

