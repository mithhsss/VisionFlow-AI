import cv2
import time
from ultralytics import YOLO
from scheduling_logic import TrafficController, Detection, VehicleType, TrafficState

# Mapping YOLO class IDs to our VehicleType hierarchy
# Assuming YOLOv8 standard COCO classes if not specific
VEHICLE_MAP = {
    0: VehicleType.AMBULANCE,    # Our custom class 0
    3: VehicleType.FIRE_ENGINE,  # Our custom class 3
    # COCO defaults (standard yolov8n.pt) if used:
    # 2: car, 5: bus, 7: truck
}

class VisionFlowSystem:
    def __init__(self, model_path='runs/emergency_vehicle_detection/weights/best.pt'):
        print(f"Initializing VisionFlow AI with model: {model_path}")
        self.model = YOLO(model_path)
        self.controller = TrafficController()
        # Velocity filter history for tracking (Ghost EV prevention)
        self.ev_history = {1: [], 2: [], 3: [], 4: []}
        
    def process_lane(self, frame, lane_id):
        """Process a single frame for a specific lane and update controller."""
        results = self.model(frame, conf=0.25, verbose=False)[0]
        
        density = 0
        evs = []
        current_y_centers = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            # Normalized Y coordinate of bbox center
            y_center = float((box.xyxyn[0][1] + box.xyxyn[0][3]) / 2)
            
            if cls_id in VEHICLE_MAP:
                # --- VELOCITY FILTER ---
                current_y_centers.append(y_center)
                is_parked = False
                
                # Check history for this lane to find a matching track
                matched_hist = None
                for hist in self.ev_history[lane_id]:
                    if abs(hist[-1] - y_center) < 0.1: # Matches an existing track
                        matched_hist = hist
                        break
                
                if matched_hist is not None:
                    matched_hist.append(y_center)
                    if len(matched_hist) > 10:
                        matched_hist.pop(0)
                        # If max variance over last 10 frames is tiny, it's stationary/parked
                        if max(matched_hist) - min(matched_hist) < 0.02:
                            is_parked = True
                else:
                    self.ev_history[lane_id].append([y_center])
                    
                if not is_parked:
                    evs.append(Detection(
                        cls_id=cls_id,
                        v_type=VEHICLE_MAP[cls_id],
                        bbox_y=y_center,
                        lane_id=lane_id
                    ))
            else:
                # Count other vehicles for density
                density += 1
                
        # Cleanup old tracks (if a track wasn't updated this frame, remove it)
        new_history = []
        for hist in self.ev_history[lane_id]:
            if hist[-1] in current_y_centers:
                new_history.append(hist)
        self.ev_history[lane_id] = new_history
                
        self.controller.update_data(lane_id, density, evs)
        return results.plot()

    def run_simulation(self, sources: dict):
        """
        Run simulation with multiple sources. 
        sources: {lane_id: source_path}
        """
        caps = {l_id: cv2.VideoCapture(src) for l_id, src in sources.items()}
        
        print("\n[SYSTEM] VisionFlow AI Traffic Controller Started")
        print("Press 'q' to exit simulation.\n")
        
        try:
            while True:
                frames = {}
                for l_id, cap in caps.items():
                    ret, frame = cap.read()
                    if not ret:
                        # Restart video if ended
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    frames[l_id] = frame

                # 1. Update Controller with all lane data
                annotated_frames = {}
                for l_id, frame in frames.items():
                    annotated_frames[l_id] = self.process_lane(frame, l_id)

                # 2. Step the controller
                self.controller.step()
                status = self.controller.get_status()

                # 3. Visualization
                # Create a 2x2 grid for 4 lanes
                h, w, _ = annotated_frames[1].shape
                top_row = cv2.hconcat([annotated_frames[1], annotated_frames[2]])
                bottom_row = cv2.hconcat([annotated_frames[3], annotated_frames[4]])
                grid = cv2.vconcat([top_row, bottom_row])

                # Overlay Status
                overlay = grid.copy()
                cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, grid, 0.4, 0, grid)
                
                color = (0, 255, 0) if status['state'] == 'NORMAL' else (0, 0, 255)
                cv2.putText(grid, f"STATE: {status['state']}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(grid, f"PHASE: {status['phase']}", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(grid, f"BURST: {status['burst_duration']}", (20, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(grid, f"ELAPSED: {status['elapsed']}", (20, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("VisionFlow AI - Global Traffic Dashboard", cv2.resize(grid, (1280, 720)))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            for cap in caps.values():
                cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VisionFlow AI Traffic Simulation")
    parser.add_argument('--source', type=str, required=True, help="Path to test video/image (will be duplicated for 4 lanes)")
    parser.add_argument('--model', type=str, default='runs/emergency_vehicle_detection/weights/best.pt', help="Path to best model weights")
    args = parser.parse_args()
    
    system = VisionFlowSystem(args.model)
    
    # Simulating 4 lanes using the same source for demonstration
    sources = {1: args.source, 2: args.source, 3: args.source, 4: args.source}
    system.run_simulation(sources)
