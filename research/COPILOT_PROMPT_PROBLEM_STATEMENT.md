# Copilot Prompt: Generate Project Problem Statement

Use the prompt below with GitHub Copilot Chat to generate a high-quality problem statement for this project.

## Context Snapshot (from this repository)

- Project goal: build a Python ROS autonomy backend that controls the Udacity Unity 3D highway simulator car safely for at least one full lap and stops correctly at red traffic lights.
- Main runtime orchestrator: `ros/launch/styx.launch`.
- Simulator bridge: `ros/src/styx/server.py` and `ros/src/styx/bridge.py`.
- Core ROS Python autonomy nodes:
	- Perception: `ros/src/tl_detector/tl_detector.py`, `ros/src/tl_detector/light_classification/tl_classifier.py`
	- Planning: `ros/src/waypoint_updater/waypoint_updater.py`
	- Control: `ros/src/twist_controller/dbw_node.py`, `ros/src/twist_controller/twist_controller.py`, `ros/src/twist_controller/yaw_controller.py`, `ros/src/twist_controller/pid.py`
- Deep learning model usage: frozen TensorFlow SSD MobileNet graph loaded by `tl_classifier.py`.
- Known operating constraints from implementation/readme:
	- Camera latency in simulator is a practical issue.
	- Traffic-light image processing is throttled to ~250 ms in `tl_detector.py`.
	- Traffic-light classification is used when within waypoint threshold (150 waypoints).
	- Waypoint updater publishes reduced lookahead set (25 waypoints) at ~20 Hz.
	- DBW control loop runs at ~50 Hz.
	- Red light is the primary stop condition; non-red generally allows proceed.

## Copy/Paste Prompt for GitHub Copilot

```text
You are a senior robotics autonomy systems engineer and technical writer.

Write a professional Problem Statement for my project: a Python ROS deep-learning-driven self-driving car backend integrated with the Udacity Unity 3D highway simulator.

Project intent:
- The autonomy backend commands and controls the main simulated car.
- The car must safely maneuver for at least 1 full lap on the virtual highway.
- The car must detect and respond to red traffic lights by stopping appropriately at stop lines.

Technical context to anchor the problem statement:
- ROS stack uses perception, planning, and control nodes in Python.
- Perception: traffic-light detection/classification from `/image_color` using a TensorFlow SSD MobileNet frozen graph.
- Planning: waypoint management and velocity shaping from `/base_waypoints` + `/traffic_waypoint` to `/final_waypoints`.
- Control: drive-by-wire throttle/brake/steering command generation from `/twist_cmd` and `/current_velocity`.
- Simulator-ROS integration uses a bridge/server layer exchanging telemetry and control with Unity simulator over socket interface.

Important constraints and realities to include:
- Real-time behavior under simulator camera/processing latency.
- Need for robust red-light stopping while minimizing false stops and unstable behavior.
- Requirement for stable closed-loop operation across perception -> planning -> control.
- Emphasis on backend autonomy software reliability, not UI.

Output requirements:
1. Write in concise, formal engineering style.
2. Include these sections with headings:
	 - Background and System Context
	 - Core Problem
	 - Why the Problem is Non-Trivial
	 - Scope and Boundaries
	 - Operational Constraints and Assumptions
	 - Success Criteria (measurable)
3. In Success Criteria, include quantifiable examples for:
	 - completing >= 1 full lap safely,
	 - stopping at red lights before/at stop line,
	 - maintaining lane-following/waypoint-tracking stability,
	 - avoiding oscillatory throttle/brake/steering behavior.
4. End with a short paragraph titled "Problem Statement Summary" (4-6 sentences).

Keep the response specific to this ROS + Unity simulator backend project and avoid generic autonomous driving boilerplate.
```

