import cv2
import mediapipe as mp

def apply_face_filter(frame):
    apple_img = cv2.imread('butterflies.png', cv2.IMREAD_UNCHANGED)
    apple_img = cv2.resize(apple_img, (150, 75))
    roi_width = 150
    roi_height = 75

    def overlay_image(background, overlay, x, y):
        h, w, _ = overlay.shape
        if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
            print("Overlay exceeds frame boundaries.")
            return background

        roi = background[y:y+roi_height, x:x+roi_width]
        overlay_mask = overlay[:,:,3] / 255.0
        background_mask = 1.0 - overlay_mask
        for c in range(0, 3):
            roi[:, :, c] = (overlay_mask * overlay[:, :, c] +
                            background_mask * roi[:, :, c])

        return background

    mp_drawing = mp.solutions.drawing_utils
    mp_face_detection = mp.solutions.face_detection

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                       int(bboxC.width * iw), int(bboxC.height * ih)
                apple_x = bbox[0] + bbox[2] // 2 - apple_img.shape[1] // 2
                apple_y = bbox[1] - apple_img.shape[0]
                frame = overlay_image(frame, apple_img, apple_x, apple_y)

    return frame