# Problem Statement: Python ROS Deep Learning Autonomy Backend for Udacity Unity Simulator

## Background and System Context
This project implements an end-to-end autonomy backend for the Udacity Self-Driving Car Nanodegree system integration simulator. The backend is built in Python on ROS and is connected to the Udacity Unity 3D highway-driving scene through a simulator bridge that exchanges telemetry and control commands in real time. The autonomy stack is organized into perception, planning, and control subsystems: traffic light perception from camera images, waypoint-based trajectory planning, and drive-by-wire (DBW) actuation of throttle, brake, and steering. The operating objective is to command the primary simulated vehicle to complete at least one full highway lap safely while obeying traffic signal constraints, specifically stopping for red lights.

## Core Problem
The core engineering problem is to design and integrate a reliable closed-loop autonomy pipeline that converts simulator sensor streams into stable, safe, and regulation-compliant vehicle behavior in real time. The system must correctly detect red traffic lights from image data, map those detections to stop-line waypoints, propagate deceleration intent through waypoint velocity updates, and produce physically plausible throttle/brake/steering commands that keep the vehicle on route without unsafe oscillations. This must be achieved under simulator timing variability and communication latency while maintaining sustained performance over a complete lap.

## Why the Problem is Non-Trivial
The problem is non-trivial because correctness depends on coupled behavior across multiple asynchronous ROS nodes and topics rather than a single model or algorithm. Perception errors, delayed image processing, or unstable classification can propagate into late braking decisions. Planning must reconcile base waypoints, current pose, and stop-line indices while preserving smooth speed transitions. Control must transform target motion commands into stable actuation at DBW loop rates, with adequate filtering and PID behavior to avoid throttle-brake chattering and steering jitter. Additionally, the simulator-ROS bridge introduces timing and data-interface constraints that can degrade closed-loop stability if subsystem rates and thresholds are not tuned coherently.

## Scope and Boundaries
In scope:
- Python ROS backend behavior for perception, planning, and control in the Udacity Unity simulator environment.
- Traffic light handling focused on reliable red-light stop decisions from camera-driven deep learning inference.
- Waypoint tracking and velocity profiling required to complete at least one safe lap with signal compliance.
- Closed-loop control quality measured through smoothness and command stability.

Out of scope:
- UI/UX features and frontend visualization beyond simulator-provided displays.
- Full autonomous stack extensions not required by this project, such as obstacle avoidance policy optimization, HD map generation, or production-grade sensor fusion.
- Real-vehicle deployment hardening, certification, and regulatory validation outside the simulator test envelope.

## Operational Constraints and Assumptions
- Runtime environment is ROS-based Python autonomy software connected to the Udacity Unity 3D simulator via bridge/server messaging.
- Traffic-light perception uses a TensorFlow SSD MobileNet frozen graph and classifies camera frames from the simulator image topic.
- System behavior must tolerate simulator camera latency and asynchronous callback execution.
- Traffic-light image processing is intentionally rate-limited (approximately 250 ms) to reduce unstable behavior under latency.
- Red-light decision publication is stabilized with repeated-state confirmation before commanding stop behavior.
- Waypoint planning uses reduced lookahead and periodic publication to balance responsiveness and simulator timing limits.
- DBW command generation operates at high control frequency and assumes stable consumption of target twist and current velocity topics.

## Success Criteria (measurable)
1. Lap completion and safety:
The autonomous vehicle completes at least 1 continuous full highway lap in the simulator without collisions, off-road departures, or manual intervention.

2. Red-light compliance:
For at least 95% of encountered red-light events in a test run, the vehicle comes to a full stop before or at the configured stop line, with no intentional red-light run-through.

3. Waypoint tracking stability:
During nominal cruising segments, lateral path-tracking remains stable with no sustained lane weaving, and waypoint progression remains monotonic with no repeated large-index regressions.

4. Longitudinal smoothness:
During cruise and approach-to-stop phases, commanded acceleration/deceleration remains smooth, with no persistent throttle-brake oscillation pattern lasting longer than 2 seconds.

5. Control command stability:
Steering, throttle, and brake outputs remain bounded and non-erratic, showing no repeated high-frequency command sign-flips under steady-road conditions.

6. End-to-end timing viability:
Perception, planning, and control nodes maintain operational loop behavior consistent with design intent (camera processing throttled near 250 ms, waypoint publishing near 20 Hz, DBW near 50 Hz) throughout the evaluation run.

## Problem Statement Summary
This project addresses the challenge of building a robust Python ROS autonomy backend that can safely control a simulated vehicle in the Udacity Unity highway environment. The system must integrate deep-learning-based traffic light perception with waypoint planning and DBW control so that red-light stops and continuous driving behavior are both reliable. The principal difficulty lies in maintaining closed-loop stability under simulator latency and asynchronous ROS execution while preserving signal compliance and smooth vehicle dynamics. Success requires measurable performance at both mission level (safe lap completion and stop-line obedience) and subsystem level (tracking quality, actuation smoothness, and loop-rate viability). The resulting problem is a systems-integration challenge where perception, planning, control, and simulator communication must function as one coherent autonomy backend.
