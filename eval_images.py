import cv2
from ultralytics import YOLO
import os

def evaluate_images():
    # Load the best model
    model_path = r'runs\detect\runs\emergency_vehicle_yolov8s\weights\best.pt'
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    image_files = ['cam1.png', 'cam2.png', 'cam3.png', 'cam4.png']
    
    # Priority classes (Ambulance=0, Fire-engine=3)
    PRIORITY_CLASSES = [0, 3] 
    
    for img_file in image_files:
        if not os.path.exists(img_file):
            print(f"Image {img_file} not found!")
            continue
            
        print(f"\n--- Processing {img_file} ---")
        # conf=0.05  -> very sensitive; catches distant/partially hidden vehicles
        # imgsz=1280 -> high-res inference so small far-away vehicles have enough pixels
        # iou=0.40   -> lower NMS threshold so close/overlapping vehicles are NOT merged into one box
        results = model(img_file, conf=0.05, iou=0.40, imgsz=1280)[0]
        
        priority_detected = False
        emergency_types = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in PRIORITY_CLASSES:
                priority_detected = True
                emergency_types.append(results.names[cls_id])
                
        if priority_detected:
            types_str = ", ".join(set(emergency_types))
            print(f"[ALERT] EMERGENCY VEHICLE DETECTED ({types_str}) - GRANTING PRIORITY!")
        else:
            print("[INFO] Normal traffic flow.")
            
        # Save the annotated image
        annotated_frame = results.plot()
        output_name = f"evaluated_{img_file}"
        cv2.imwrite(output_name, annotated_frame)
        print(f"Saved inference result to {output_name}")

if __name__ == "__main__":
    evaluate_images()
