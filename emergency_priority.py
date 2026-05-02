import cv2
from ultralytics import YOLO

# Dictionary for colors (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 0, 255),    # Ambulance - Red
    3: (0, 0, 255),    # Fire-engine - Red
}
DEFAULT_COLOR = (255, 0, 0) # Other vehicles - Blue

# Priority classes (Ambulance, Fire-engine) based on the data.yaml
PRIORITY_CLASSES = [0, 3] 

def detect_and_prioritize(source_path, model_path='runs/emergency_vehicle_detection/weights/best.pt'):
    # Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    # Check if the source is an image or video/webcam
    import mimetypes
    mime_type, _ = mimetypes.guess_type(source_path)
    
    if mime_type and mime_type.startswith('image'):
        process_image(model, source_path)
    else:
        # Fallback to video processing if it's not an image or is a camera index
        process_video(model, source_path)

def process_image(model, image_path):
    # Run inference on the image (lowered threshold to catch distant vehicles)
    results = model(image_path, conf=0.15)[0]
    
    # Check prioritization
    priority_detected = False
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in PRIORITY_CLASSES:
            priority_detected = True
            break
            
    if priority_detected:
        print("\n[ALERT] EMERGENCY VEHICLE DETECTED - GRANTING PRIORITY!\n")
    else:
        print("\n[INFO] Normal traffic flow.\n")
        
    # Display the image with bounding boxes
    annotated_frame = results.plot()
    cv2.imshow("Traffic Management System - Inference", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(model, source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error opening source: {source}")
        return

    print("Press 'q' to quit the video.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference on the frame
        results = model(frame)[0]
        
        priority_detected = False
        
        # Check for emergency vehicles
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in PRIORITY_CLASSES:
                priority_detected = True
                break
                
        # If an emergency vehicle is detected, we could hypothetically
        # trigger a signal change here.
        if priority_detected:
            # Draw an alert on the screen
            cv2.putText(frame, "EMERGENCY VEHICLE DETECTED - PRIORITY ACTIVATED", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Draw the resulting boxes
        annotated_frame = results.plot()
            
        cv2.imshow("Traffic Management System", annotated_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Emergency Prioritization")
    parser.add_argument('--source', type=str, required=True, help="Path to test image or video (or 0 for webcam)")
    parser.add_argument('--model', type=str, default='runs/emergency_vehicle_detection/weights/best.pt', help="Path to best model weights")
    args = parser.parse_args()
    
    detect_and_prioritize(args.source, args.model)
