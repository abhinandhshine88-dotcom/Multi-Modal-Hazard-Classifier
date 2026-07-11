

# Multi-Modal Edge Hazard Detection & Flame Classification System

An intelligent, low-latency edge-computing node designed to detect, track, and classify complex environmental fire hazards by fusing real-time computer vision stream telemetry with an electrochemical electronic nose (E-Nose) sensor array.

## System Architecture Overview

Unlike traditional threshold-based safety systems that suffer from high false-alarm rates, this framework implements **Multi-Modal Sensor Fusion** to cross-validate environmental threats:

* **Vision Module:** Leverages a custom-trained **YOLOv8** object detection model to isolate localized hazard boundaries (fire/smoke zones) while concurrently tracking human presence within the area.
* **Chemical Ingestion Module (E-Nose):** Interacts asynchronously with an **Arduino Uno** gateway reading live data vectors from an **MQ-135** (Air Quality/Toxic gases) and **MQ-2** (Combustible smoke/gas) sensor array.
* **Deep Learning Classifier:** Instead of using hardcoded numeric cutoffs, the raw chemical vectors are fed directly into a feed-forward **Multi-Layer Perceptron (MLP)** neural network implemented using optimized matrix mathematics (`NumPy`) right at the edge layer.

## Classification Matrix

By evaluating shifting gas ratios in real time, the network calculates live forward-pass probabilities to classify distinct real-world threat signatures:

1. **Electrical / Plastic Fire:** Spikes in highly toxic combustion byproducts (MQ-135 dominance).
2. **Combustible Gas / Chemical Leak:** Spikes in unburnt alkanes/fuel vapors (MQ-2 dominance).
3. **Biomass Fire (Wood/Smoke):** Deep simultaneous spikes across both sensors due to heavy carbon dioxide and thick particulate accumulation.

## Tech Stack & Components

* **Core Language:** Python 3.x
* **AI/Frameworks:** Ultralytics YOLOv8, NumPy (MLP Inference Implementation)
* **Libraries:** OpenCV (UI Dashboard HUD & Video Processing), PySerial (Hardware Gateway Link)
* **Hardware Node:** Arduino Uno Microcontroller, MQ-135 Sensor, MQ-2 Sensor, Active Piezo Buzzer, RGB Camera.

## Simulation & Testing

The system features an automated fallback architecture. When physical hardware is disconnected, it securely enters **Simulation Mode**, allowing developers to simulate active hazards via keyboard hotkeys (`1` for Electrical, `2` for Gas, `3` for Biomass, `r` to Reset) to evaluate the deep learning model’s live responsiveness.
