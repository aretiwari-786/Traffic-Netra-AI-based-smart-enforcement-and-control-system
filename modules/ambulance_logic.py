# modules/ambulance_logic.py
import math
import cv2
import cvzone

def check_for_ambulance(img, model_ambulance):
    """
    Checks the frame for an ambulance using a dedicated model.
    Returns: Boolean (True if found, False otherwise). Draws directly on image.
    """
    ambulance_detected = False
    
    results_amb = model_ambulance(img, stream=True, verbose=False)
    for r in results_amb:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > 0.5:
                ambulance_detected = True
                
                # Draw directly on the original image
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cvzone.putTextRect(img, "AMBULANCE DETECTED", (max(0, x1), max(35, y1 - 20)), 
                                   colorR=(0, 0, 255), scale=1.5, thickness=2)
                
    return ambulance_detected