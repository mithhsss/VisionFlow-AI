import time
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

class TrafficState(Enum):
    NORMAL = "NORMAL"
    PRE_CLEAR = "PRE_CLEAR"
    EV_ACTIVE = "EV_ACTIVE"
    MICRO_BURST = "MICRO_BURST"
    COOLDOWN = "COOLDOWN"

class VehicleType(Enum):
    AMBULANCE = 1
    FIRE_ENGINE = 2
    POLICE = 3
    OTHER = 4

@dataclass
class Detection:
    cls_id: int
    v_type: VehicleType
    bbox_y: float  # BBox center Y (0 to 1 normalized)
    lane_id: int

class TrafficController:
    def __init__(self):
        # Constants
        self.BASE_TIME = 5
        self.SECONDS_PER_VEHICLE = 1.5
        self.MAX_BURST = 20
        self.TRANSITION_YELLOW = 3
        self.TRANSITION_RED = 2
        
        # State
        self.state = TrafficState.NORMAL
        self.current_phase = 0  # 0: Phase A (L1, L3), 1: Phase B (L2, L4)
        self.phase_start_time = time.time()
        self.current_burst_duration = self.BASE_TIME
        
        # Lane Data
        self.lane_data = {
            1: {"density": 0, "evs": [], "wait_time": 0},
            2: {"density": 0, "evs": [], "wait_time": 0},
            3: {"density": 0, "evs": [], "wait_time": 0},
            4: {"density": 0, "evs": [], "wait_time": 0},
        }
        
        self.last_update_time = time.time()

    def update_data(self, lane_id: int, density: int, evs: List[Detection]):
        """Update sensor data for a specific lane."""
        self.lane_data[lane_id]["density"] = density
        self.lane_data[lane_id]["evs"] = evs
        # Update wait time logic would go here in a real loop

    def get_priority_score(self, lane_id: int) -> float:
        """Calculate DWP score for a lane."""
        data = self.lane_data[lane_id]
        # P = (W_density * N) + (W_ev * E) + (W_wait * T)
        # Simplified for logic demo
        ev_score = sum([1000 / ev.v_type.value for ev in data["evs"]])
        return (data["density"] * 2) + ev_score + (data["wait_time"] * 0.5)

    def determine_state(self):
        """State Machine Transition Logic."""
        all_evs = []
        for l_id, data in self.lane_data.items():
            all_evs.extend(data["evs"])
        
        active_ev_count = len([ev for ev in all_evs if ev.bbox_y > 0.7]) # Close to line
        far_ev_count = len([ev for ev in all_evs if 0.3 < ev.bbox_y <= 0.7]) # Pre-clearance zone
        
        # Check for conflicting EVs (Phase A vs Phase B)
        phase_a_evs = len(self.lane_data[1]["evs"]) + len(self.lane_data[3]["evs"])
        phase_b_evs = len(self.lane_data[2]["evs"]) + len(self.lane_data[4]["evs"])
        
        if phase_a_evs > 0 and phase_b_evs > 0:
            self.state = TrafficState.MICRO_BURST
        elif active_ev_count > 0:
            self.state = TrafficState.EV_ACTIVE
        elif far_ev_count > 0:
            self.state = TrafficState.PRE_CLEAR
        elif self.state == TrafficState.COOLDOWN and (time.time() - self.phase_start_time > 5):
            self.state = TrafficState.NORMAL
        elif self.state not in [TrafficState.EV_ACTIVE, TrafficState.MICRO_BURST, TrafficState.PRE_CLEAR]:
            self.state = TrafficState.NORMAL

    def resolve_conflicts(self) -> int:
        """Strict Hierarchy + Tiebreaker Logic."""
        best_ev = None
        for l_id, data in self.lane_data.items():
            for ev in data["evs"]:
                if best_ev is None:
                    best_ev = ev
                    continue
                
                # 1. Vehicle Type Hierarchy (Lower value is higher priority)
                if ev.v_type.value < best_ev.v_type.value:
                    best_ev = ev
                elif ev.v_type.value == best_ev.v_type.value:
                    # 2. Distance Tiebreaker (Lower BBox Y = Further away = Arrived first)
                    if ev.bbox_y < best_ev.bbox_y:
                        best_ev = ev
        
        return best_ev.lane_id if best_ev else None

    def calculate_burst_time(self, lane_id: int) -> float:
        """Dynamic Burst Scaling."""
        n_l = self.lane_data[lane_id]["density"]
        duration = self.BASE_TIME + (n_l * self.SECONDS_PER_VEHICLE)
        return min(self.MAX_BURST, duration)

    def step(self):
        """Main control loop step."""
        self.determine_state()
        now = time.time()
        elapsed = now - self.phase_start_time
        
        if self.state == TrafficState.MICRO_BURST:
            # Handle conflict using hierarchy
            target_lane = self.resolve_conflicts()
            target_phase = 0 if target_lane in [1, 3] else 1
            
            if target_phase != self.current_phase:
                self.switch_phase(target_phase)
            
            self.current_burst_duration = self.calculate_burst_time(target_lane)
            
        elif self.state == TrafficState.EV_ACTIVE:
            # Grant green to the EV phase
            target_lane = self.resolve_conflicts()
            target_phase = 0 if target_lane in [1, 3] else 1
            if target_phase != self.current_phase:
                self.switch_phase(target_phase)
            self.current_burst_duration = 999 # Keep green
            
        elif self.state == TrafficState.NORMAL:
            # Standard DWP switching
            if elapsed > self.current_burst_duration:
                # Calculate scores for phases
                score_a = self.get_priority_score(1) + self.get_priority_score(3)
                score_b = self.get_priority_score(2) + self.get_priority_score(4)
                
                next_phase = 0 if score_a > score_b else 1
                if next_phase != self.current_phase:
                    self.switch_phase(next_phase)
                
                # Update burst time for the new phase
                lead_lane = 1 if next_phase == 0 else 2
                self.current_burst_duration = self.calculate_burst_time(lead_lane)

    def switch_phase(self, new_phase: int):
        print(f"[TRANSITION] Switching Phase {self.current_phase} -> {new_phase}")
        print(f"[SAFETY] Yellow: {self.TRANSITION_YELLOW}s | All-Red: {self.TRANSITION_RED}s")
        time.sleep(0.1) # Mock delay
        self.current_phase = new_phase
        self.phase_start_time = time.time()

    def get_status(self):
        return {
            "state": self.state.value,
            "phase": "Phase A (N-S)" if self.current_phase == 0 else "Phase B (E-W)",
            "burst_duration": f"{self.current_burst_duration:.1f}s",
            "elapsed": f"{time.time() - self.phase_start_time:.1f}s"
        }

if __name__ == "__main__":
    # Quick Test
    controller = TrafficController()
    
    # Simulate conflicting EVs
    print("--- Test 1: Conflicting EVs (Ambulance L2 vs Fire Engine L1) ---")
    controller.update_data(1, density=5, evs=[Detection(3, VehicleType.FIRE_ENGINE, 0.5, 1)])
    controller.update_data(2, density=10, evs=[Detection(0, VehicleType.AMBULANCE, 0.4, 2)])
    
    controller.step()
    print(f"Controller Status: {controller.get_status()}")
    
    target = controller.resolve_conflicts()
    print(f"Hierarchy Winner Lane: {target} (Should be 2 - Ambulance)")
