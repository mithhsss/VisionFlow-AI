from ultralytics import YOLO
import torch

def main():
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("GPU not available, using CPU")
        device = 'cpu'

    # Load a pretrained YOLOv8 Small model (more robust and accurate than Nano)
    model = YOLO('yolov8s.pt') 

    # Train the model with parameters designed to achieve > 90% accuracy
    # Since yolov8s.pt takes more memory, batch size 8 is chosen for the 4GB VRAM RTX 3050 Ti
    results = model.train(
        data=r'aleesiashaloem 2.v1i.yolov8\data.yaml',
        epochs=300,             # Increased epochs for better learning
        imgsz=640,              # Standard image size format
        batch=8,                # Lower batch size to easily run on 4GB VRAM GPU with 'Small' model
        device=device,
        patience=50,            # Early stopping patience: waits 50 epochs for improvement to avoid overfitting
        optimizer='AdamW',      # Forcing robust Adam with Weight Decay optimization
        project='runs',
        name='emergency_vehicle_yolov8s',
        # Adding data augmentations to increase chances of surpassing 90% accuracy
        mosaic=1.0,             # Stitches 4 images together to help model learn smaller details
        mixup=0.15,             # Overlays an image on top of another slightly
        degrees=10.0,           # Slight rotation variation
        translate=0.1,          # Image translation translation
        scale=0.5               # Image scaling augmentations
    )

    print("Training complete. Models and metrics are saved in runs/emergency_vehicle_yolov8s")

if __name__ == '__main__':
    main()
