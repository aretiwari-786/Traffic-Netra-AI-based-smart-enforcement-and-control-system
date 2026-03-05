# modules/tripling_logic.py

def detect_tripling(img, x1, y1, x2, y2, model_general, classNames):
    """
    Checks for tripling on a motorcycle by cropping and re-running the model.
    Returns: A dictionary with warning text and color.
    """
    warning_text = ""
    color = (255, 0, 0) # Default Blue

    # 1. Crop the bike area (add some top margin to catch heads)
    y1_crop = max(0, y1 - 20)
    bike_crop = img[y1_crop:y2, x1:x2]
    
    # 2. Check if crop is valid
    if bike_crop.size != 0:
        bike_results = model_general(bike_crop, stream=True, verbose=False)
        rider_count = 0
        
        # 3. Count persons
        for br in bike_results:
            for b_box in br.boxes:
                b_cls = int(b_box.cls[0])
                if b_cls < len(classNames) and classNames[b_cls] == "person":
                    rider_count += 1
        
        # 4. Check rule
        if rider_count > 2:
            color = (0, 0, 255) # Red for Violation
            warning_text = f"TRIPLING: {rider_count} Riders!"
            
    return {
        "text": warning_text,
        "color": color
    }