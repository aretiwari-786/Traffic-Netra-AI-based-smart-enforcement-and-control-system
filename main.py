import cv2
import cvzone
from ultralytics import YOLO

print("Starting Traffic-Netra: Static Image Analysis...")

# ==============================================================================
# -------------------- ALGORITHM 1: DYNAMIC SIGNAL TIMING --------------------
# ==============================================================================
def calculate_green_time(vehicle_count):
    min_time = 10
    max_time = 60
    time_per_vehicle = 2
    
    calculated_time = min_time + (vehicle_count * time_per_vehicle)
    
    if calculated_time < min_time: 
        return min_time
    elif calculated_time > max_time: 
        return max_time
    else: 
        return calculated_time

def get_signal_color(green_time):
    if green_time >= 50: return (0, 200, 0)     # Green (High Density)
    elif green_time > 20: return (0, 165, 255)  # Orange (Medium Density)
    else: return (0, 0, 255)                    # Red (Low Density)

# ==============================================================================
# ----------------------- ALGORITHM 2: TRIPLING DETECTOR -----------------------
# ==============================================================================
def detect_tripling(img, x1, y1, x2, y2, model_general, classNames):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    y1_crop = max(0, int(y1 - 20)) 
    bike_crop = img[y1_crop:y2, x1:x2]
    
    if bike_crop.size == 0:
        return {"violation": False, "text": "", "color": (255, 0, 0)}

    bike_results = model_general(bike_crop, verbose=False)
    
    rider_count = 0
    for br in bike_results:
        for b_box in br.boxes:
            b_cls = int(b_box.cls[0])
            if classNames[b_cls] == "person":
                rider_count += 1
                
    if rider_count > 2:
        return {"violation": True, "text": f"TRIPLING: {rider_count} Riders!", "color": (0, 0, 255)}
    else:
        return {"violation": False, "text": "", "color": (255, 0, 0)}

# ==============================================================================
# ------------------------------ INITIALIZATION --------------------------------
# ==============================================================================
# Ensure your yolov8n.pt is inside the 'models' folder!
model_general = YOLO("models/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
target_vehicles = ["car", "bus","bicycle","truck", "motorbike"]

# ==============================================================================
# 1. READ THE IMAGE (UPDATED FOR YOUR FOLDER STRUCTURE)
# ==============================================================================
# Yahan aap 'low_traffic.jfif', 'medium_traffic.jfif', ya 'high_traffic.jfif' likh sakte hain
image_name = "traffic_bike.jpeg" 

# Automatically routes to your videos-images folder
image_path = f"videos-images/{image_name}" 
img = cv2.imread(image_path)

if img is None:
    print(f"❌ Error: Could not find image at '{image_path}'. Please check the name!")
    exit()

# ==============================================================================
# 2. RUN AI ON THE SINGLE IMAGE
# ==============================================================================
vehicle_count = 0
results_gen = model_general(img, verbose=False)

for r in results_gen:
    for box in r.boxes:
        cls = int(box.cls[0])
        currentClass = classNames[cls]

        if currentClass in target_vehicles:
            vehicle_count += 1
            x1, y1, x2, y2 = box.xyxy[0]
            
            box_color = (255, 0, 0) # Default Blue
            
            # Check for Tripling
            if currentClass == "motorbike":
                tripling_data = detect_tripling(img, x1, y1, x2, y2, model_general, classNames)
                
                if tripling_data["violation"]:
                    box_color = tripling_data["color"]
                    cvzone.putTextRect(img, tripling_data["text"], (max(0, int(x1)), max(35, int(y1) - 20)), 
                                       colorR=box_color, scale=1.2, thickness=2)

            # Draw standard boxes
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            if currentClass != "motorbike" or not tripling_data["violation"]:
                 cvzone.putTextRect(img, f'{currentClass}', (max(0, int(x1)), max(35, int(y1))), 
                                   scale=1, thickness=1, offset=3)

# ==============================================================================
# 3. RENDER THE SMART DASHBOARD
# ==============================================================================
req_time = calculate_green_time(vehicle_count)
sig_color = get_signal_color(req_time)

cv2.rectangle(img, (10, 10), (550, 110), (0, 0, 0), cv2.FILLED)
cvzone.putTextRect(img, f'VEHICLES IN LANE: {vehicle_count}', 
                   (20, 40), colorR=(0, 0, 0), colorT=(255, 255, 255), scale=1.5, thickness=2, offset=0)
cvzone.putTextRect(img, f'SIGNAL TIME: {req_time} Sec', 
                   (20, 90), colorR=sig_color, colorT=(255, 255, 255), scale=1.5, thickness=2, offset=0)

# ==============================================================================
# 4. SAVE AND DISPLAY THE RESULT
# ==============================================================================
output_name = f"processed_{image_name}.jpg"
cv2.imwrite(output_name, img)
print(f"✅ Success! Analyzed {vehicle_count} vehicles.")
print(f"📸 Saved the final result as '{output_name}' in your main folder.")

# Show the image on screen
cv2.imshow(f"Traffic-Netra: {image_name}", img)

# Wait infinitely until the user presses ANY key, then close
cv2.waitKey(0) 
cv2.destroyAllWindows()