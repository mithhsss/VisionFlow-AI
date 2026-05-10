# Traffic Scheduling Logic - VisionFlow AI

This plan outlines the design for a **Dynamic Weighted Pressure (DWP)** scheduling algorithm for a 4-signal intersection. The goal is to maximize throughput for normal traffic while providing instantaneous, safe preemption for emergency vehicles.

## User Review Required

> [!IMPORTANT]
> **Priority Hierarchy:** I have proposed a hierarchy (Ambulance > Fire Engine). Please confirm if this matches your local traffic regulations.
>
> **Sensor Reliance:** The plan assumes high confidence from the YOLO model. We need to decide if the system should revert to a fixed-time "safe mode" if the camera feed fails.

## Edge Cases & Resolutions

| Edge Case                               | Impact                                   | Proposed Resolution                                                                                          |
| :-------------------------------------- | :--------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| **Simultaneous Conflicting EV Arrival** | Two EVs (North/East) cross paths. | **Strict Hierarchy:** Use Ambulance > Fire > Police. If same type, lower BBox Y (further away) wins tiebreaker. |
| **Parallel EV Arrival**                 | EVs on North/South (non-conflicting).    | **Phase Compatibility:** Grant **Parallel Green** to both lanes simultaneously. Zero delay for either.       |
| **"The Plug" (Congestion Ahead)**       | EV is stuck behind 20 cars at red.       | **Pre-Clearance Flushing:** Trigger green 5-10s _before_ the EV reaches the line to clear the queue ahead.   |
| **"Ghost" Emergency Vehicle**           | Parked EV near intersection.             | **Velocity Filter:** Only trigger preemption if the bbox center moves >X pixels over 3 frames.               |
| **Transition Safety**                   | Potential collision during instant flip. | **Mandatory Intervals:** Never skip the 3s Yellow / 2s All-Red clearance, even for emergency preemption.     |
| **Camera Failure/Occlusion**            | Blind spots in the system.               | **Safe Mode:** Revert to standard fixed-time round-robin if confidence <0.4 for >5 seconds.                  |

## Proposed Scheduling Algorithm: Dynamic Weighted Pressure with Phase Compatibility (DWP-PC)

The algorithm calculates a **"Priority Score" ($P_L$)** for each lane ($L \in \{1, 2, 3, 4\}$):

$$P_L = (W_{density} \times N_L) + (W_{emergency} \times E_L) + (W_{wait} \times T_L)$$

### 1. Formalized State Machine
To manage transitions, the system operates in 5 explicit states:
- **NORMAL**: Standard density-based DWP logic.
- **PRE_CLEAR**: EV detected at distance (50-100m); flushes current queue.
- **EV_ACTIVE**: EV is in the intersection zone; permanent green until cleared.
- **MICRO_BURST**: Multiple conflicting EVs; cycles green in dynamic bursts.
- **COOLDOWN**: Post-preemption cleanup to restore normal flow.

**Interruption Rules:** `EV_ACTIVE` overrides all other states. `MICRO_BURST` is triggered when multiple `EV_ACTIVE` signals conflict.

### 2. Strict EV Hierarchy & Tiebreaker
If multiple EVs arrive, priority is determined by:
1. **Vehicle Type:** `Ambulance` > `Fire Engine` > `Police` > `Other`.
2. **Distance Tiebreaker:** If types are identical, the vehicle with the lower BBox Y center (further from the stop line at time of detection) is prioritized as it has been in the "system" longer.

### 3. Dynamic Burst Scaling
Instead of fixed timers, burst duration is calculated based on queue length ($N_L$):
$$T_{burst} = \min(20, 5 + (N_L \times 1.5))$$
This ensures large queues are fully cleared before switching phases during a `MICRO_BURST` or `PRE_CLEAR` event.

### 4. Normal Mode (Weighted Round-Robin)
- If no EVs are present, the system defaults to density-based switching using the $P_L$ formula.

## Proposed Changes

### [Traffic Logic Component]

#### [NEW] [scheduling_logic.py](file:///c:/Users/Mithul/Desktop/VisionFlow%20AI/scheduling_logic.py)

This file will contain the core `TrafficController` class.

- `calculate_density()`: Integrates with YOLO output to count $N_L$ and $E_L$.
- `get_next_lane()`: Implements the $P_L$ formula to decide the next green.
- `handle_preemption()`: Manages the state machine for emergency overrides.

#### [MODIFY] [emergency_priority.py](file:///c:/Users/Mithul/Desktop/VisionFlow%20AI/emergency_priority.py)

- Refactor to support 4 simultaneous feeds (or a split-screen 4-way feed).
- Pass detection results to `scheduling_logic.py` instead of just printing alerts.

## Verification Plan

### Automated Tests

- **Simulated Traffic Scenarios:** Create a test script that feeds pre-recorded videos of 4 different lanes to the controller and logs the switching times.
- **Preemption Latency Test:** Measure the time from "Ambulance Detected" to "Signal State = Green" (Target: <500ms).

### Manual Verification

- **Stress Test:** Run the system with a "Conflict" video (two ambulances arriving) and verify the Priority Matrix works as expected.
- **Starvation Test:** Block one lane with high virtual density and ensure other lanes still eventually get a turn.
