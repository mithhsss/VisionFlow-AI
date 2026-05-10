# Inference Flow & Data Analysis Report - VisionFlow AI

This report documents the end-to-end data flow from image input to detection output using the custom-trained **YOLOv8s** model.

## 1. Inference Test Case
- **Test Image:** `101Ambulance.jpg`
- **Model Used:** `runs/detect/runs/emergency_vehicle_yolov8s/weights/best.pt`
- **Inference Speed:** ~211.7ms (Total flow: Preprocess -> Inference -> Postprocess)

## 2. Data Flow Visualization

### Step 1: Input (Tensor Transformation)
The raw image pixels are ingested and normalized into a **(1, 3, 640, 640)** tensor.

### Step 2: Detection (YOLO Output)
The model identifies spatial bounding boxes and class probabilities. In this test, **2 objects** were detected.

| Detection # | Class | Confidence | Position (Center X, Y) | Size (WxH) | **Proximity Area** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Ambulance | 86% | (328, 394) | 620 x 399 | **247,512 px²** |
| **2** | Ambulance | 52% | (338, 318) | 569 x 558 | **317,616 px²** |

## 3. Scheduling Logic Integration

### Proximity Tie-Breaking
The "Proximity Area" is calculated as $Width \times Height$ of the bounding box. In this scenario:
- **Ambulance 2 (317k Area)** is mathematically identified as being **closer** to the camera/signal than Ambulance 1 (247k Area).
- **Decision:** If these vehicles were in conflicting lanes, the lane with the larger area (Ambulance 2) would be granted the green light first.

### Confidence Filtering
- **Thresholding:** The system uses a confidence threshold (e.g., >0.5) to filter out false positives. 
- **Action:** Both detections in this test pass the threshold and would trigger the scheduling priority.

---
*Report generated on 2026-05-03*
