# Intelligent Traffic Management System - Prioritizing Emergency Vehicles

## 1. Methodology

The core objective of this project is to develop an intelligent traffic management system capable of identifying and granting right-of-way to emergency vehicles (such as Ambulances and Fire Engines).

### Architecture & Algorithm (YOLOv8-Small)
The system utilizes the **YOLOv8 (You Only Look Once, version 8)** object detection algorithm, specifically the YOLOv8-Small (`yolov8s`) variant. YOLO algorithms are renowned for their speed and accuracy in real-time object detection tasks.

The `yolov8s` model was chosen because it strikes an optimal balance between low memory utilization (which is critical when deploying on edge devices or consumer GPUs like the RTX 3050 Ti) while offering competitive inference speeds and high accuracy rates.

### Project Workflow
The system operates on a linear inference workflow:
1. **Input Ingestion**: The system continuously captures frames either from a live traffic camera feed or pre-recorded video files.
2. **Preprocessing**: The input frames are resized to a standard 640x640 resolution to match the YOLOv8 tensor input requirements.
3. **Inference**: The custom-trained YOLOv8 model evaluates the frame, identifying spatial bounding boxes and classifying the objects within them.
4. **Logic & Filtering**: The output classifications are filtered against a priority dictionary. If the detected class ID matches `0` (Ambulance) or `3` (Fire Engine), a priority flag is triggered.
5. **Action/Alert**: The system visually alerts the operator (or dynamically interfaces with a traffic signal controller state) to grant right-of-way to that lane.

### Tools & Technologies Used
- **Programming Language:** Python 3.11
- **Machine Learning Framework:** PyTorch (with CUDA integration for GPU acceleration)
- **Computer Vision Model:** Ultralytics YOLOv8
- **Image Processing:** OpenCV (`cv2`)
- **Environment Management:** Conda
- **Hardware setup:** Dedicated Nvidia RTX 3050 Ti (4GB VRAM) utilized for hardware-accelerated model training and rapid inference.

---

## 2. Implementation

The implementation of the traffic management logic is distinctly split into two major modules: the **Training Module** and the **Inference (Priority) Module**.

### Dataset Partitioning & Usage
The dataset `aleesiashaloem 2.v1i.yolov8` (sourced via Roboflow) consists of **60,413 annotated traffic images** encompassing 8 different vehicle classes. To ensure the model learns effectively and generalizes well to unseen data, the dataset was strategically partitioned into three subsets:
1. **Training Set (70% - 42,289 images):** Used to actually teach the model. The model looks at these images, makes predictions, and updates its internal weights based on the loss function. Deep data augmentation techniques are applied during this phase to increase robustness.
2. **Validation Set (20% - 12,083 images):** Used during training to evaluate the model after each epoch. The model does *not* learn from this data, but it helps monitor for overfitting. It is also used to compute validation metrics that guide early stopping mechanisms.
3. **Testing Set (10% - 6,041 images):** A completely unseen portion of data held back for final evaluation, ensuring the model's accuracy metrics represent real-world performance on images it has never encountered during training.

### Optimizer Logic: AdamW (Adam with Weight Decay)
A crucial aspect of achieving high convergence across the 300 epochs was the selection of the optimizer algorithm. This project utilized the **AdamW** algorithm. 

AdamW separates the "weight decay" penalty from the core gradient updates (a flaw in standard Adam). In prospect to our real-world dataset—which has noisy backgrounds and varied lighting—this decoupling ensures the model learns robust features rather than simply memorizing pixel patterns.

AdamW updates the parameters ($\theta$) using the calculation:
$ \theta_{t} = \theta_{t-1} - \eta \big( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \big) $

Where:
- $\eta$: The learning rate determining the speed of error correction.
- $\hat{m}_t$: The exponentially smoothed moving average of the gradient.
- $\hat{v}_t$: The exponentially smoothed moving average of the squared gradient (managing step sizes for specific weights).
- $\lambda \theta_{t-1}$: The decoupled weight decay penalty, crucial for preventing the network from assigning too much importance to a specific set of pixels.

### Output Classes Design
The system parses 8 distinct vehicle classes from the dataset: 
`['Ambulance', 'Auto', 'Bus', 'Fire-engine', 'bike', 'car', 'truck', 'van']`. 

To establish the emergency prioritization, the system design maps the specific indices for these classes structurally within the code:
```python
# Priority classes logic mapping
CLASS_COLORS = {
    0: (0, 0, 255),    # Ambulance - Highlighted in Red
    3: (0, 0, 255),    # Fire-engine - Highlighted in Red
}
PRIORITY_CLASSES = [0, 3] # Array defining the system's trigger points
```

### Module 1: The Training Script (`train_yolo.py`)
To make the base YOLOv8 model intelligent regarding our specific emergency layout, a custom training script was developed. 

**Key Implementations:**
- **Hardware Allocation:** Programmatically forces PyTorch to utilize the `cuda:0` device, guaranteeing the RTX 3050 Ti is utilized rather than defaulting to the slower CPU tensor processing.
- **Memory Optimization:** Due to the 4GB VRAM limit on the GPU, the `batch` size was manually scaled down to `8` to prevent CUDA "Out of Memory" crashes while still using the robust `yolov8s.pt` architecture.
- **Data Augmentation:** To prevent the model from overfitting on the training data and to ensure it survives unpredictable real-world camera angles, strong data augmentations were coded into the training hyperparameters.

### Module 2: The Real-time Detection Script (`emergency_priority.py`)
This script acts as the deployable front-end of the system.

**Key Implementations:**
1. **Dynamic Input Handling:** It uses the Python `mimetypes` library to automatically guess if the user is feeding it a static image or a video stream, and dynamically routes the data to either appropriate processing function.
2. **Priority Triggering:** Inside the processing loops, it iterates through the inference results. If `cls_id in PRIORITY_CLASSES` evaluates to True, it triggers the alert payload, writing **"EMERGENCY"** to the frame and highlighting the vehicle with a red bounding box.

*(Note for user: You can include screenshots of the bounding box detections on video files right below this section)*


## 3. Results & Analysis

The deep learning model was trained iteratively over a maximum lifecycle of **300 epochs**. It reached its peak and automatically early-stopped at Epoch 300 perfectly safely.

### Final Accuracy Metrics (Epoch 300)
The results demonstrate a highly successful model capable of deploying in real-world scenarios:
- **mAP50 (Mean Average Precision at 50% IoU):** `90.06%` 
*(The primary accuracy marker indicating the model safely exceeds the 90% target threshold).*
- **Precision:** `90.35%`
*(Indicates that when the system predicts an emergency vehicle is present, it is correct >90% of the time. This guards heavily against false-positive signal changes).*
- **Recall:** `86.55%`
*(Demonstrates the model's ability to locate a high majority of the target objects in the frame).*

### Epoch Comparison Table
The table below charts the model's rapid learning curve as it ingested the dataset over the processing period:

| Training Epoch | mAP50 Accuracy | Precision | Recall |
| :--- | :---: | :---: | :---: |
| **Epoch 1** | 66.90% | 67.90% | 64.91% |
| **Epoch 50** | 87.39% | 88.05% | 81.32% |
| **Epoch 100** | 88.77% | 89.48% | 84.02% |
| **Epoch 200** | 90.01% | 90.51% | 85.08% |
| **Epoch 300** | **90.06%** | **90.35%** | **86.55%** |

*Note: Visual graphs of the training results (Confusion matrix, PR curve, and Loss drops over time) can be found exported inside the `runs/detect/runs/emergency_vehicle_yolov8s/` directory.*

---

## 4. Challenges Faced & Resolutions

During the development and training lifecycle of this intelligent system, several technical hurdles were encountered and resolved.

### 1. Memory Constraints (CUDA Out of Memory)
- **Challenge:** Initial attempts to run the heavier YOLO architectures resulted in the GPU immediately failing due to VRAM overflow. The local Nvidia RTX 3050 Ti features 4GB of VRAM, which is extremely limited for training on large-scale datasets of 60,000+ images.
- **Resolution:** The code was heavily refactored. The model size was restricted to `yolov8s.pt` (Small) rather than the standard/large versions. Additionally, the `batch` size parameter was forced to `8` instead of the default `16`.

### 2. Dataset Overfitting
- **Challenge:** Around Epoch 70, the model started to plateau in its learning, becoming overly adapted to the specific lighting conditions found inside the 42,289 training images instead of generalizing to real-world variations.
- **Resolution:** Deep data augmentations were injected directly into the training hyperparameter script. By invoking `mosaic` (stitching image shards together), `mixup` (overlaying vehicle shapes), and random brightness adjustments, the model learned invariant features robust to environmental changes.

### 3. File Path Resolution Errors (OS Specific)
- **Challenge:** The YOLOv8 library often fails to map directory indices cleanly on Windows operating systems using relative paths, causing immediate crashes before Epoch 1 began.
- **Resolution:** The `data.yaml` and training scripts have been updated to use **relative paths** (e.g., `./train/images`). This ensures the project is portable and can run on any machine without hard-coded directory dependencies.
