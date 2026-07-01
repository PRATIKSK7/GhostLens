# 👻 GhostLens - Advanced AI Vision System
**Author**: [Pratik Kanoj (PRATIKSK7)](https://github.com/PRATIKSK7)

GhostLens is a state-of-the-art, real-time AI vision and augmented reality (AR) pipeline built in Python. Powered by Streamlit, MediaPipe, and OpenCV, it transforms your standard webcam into a high-performance gesture recognition and reality-bending tool with blazing-fast latencies (<35ms).

---

## 👔 Interview Privacy & Professional Focus

GhostLens was purpose-built to help you ace your interviews without distractions:
- **Zero Background Clutter:** The *Background Blur* and *Invisibility* modes completely eliminate messy rooms or unexpected background events.
- **Maintain Professionalism:** *Primary Face Lock* and *Blur Others* ensure that the camera's AI prioritizes you over anyone else accidentally walking into the frame.
- **Confident Presentations:** Look clean, focused, and professional during any video call, guaranteeing that interviewers see only what matters: you.

---

## ✨ Key Features

- **Background Segmentation:** Instantly swap between Normal Mode, Background Blur, and Full Invisibility.
- **Gesture Control:** Control the UI intuitively using your hands! 
  - ✋ **Open Palm:** Trigger background blur/invisibility.
  - 👍 **Thumbs Up:** Take a screenshot instantly.
  - 🤏 **Pinch:** Adjust invisibility level or trigger effects.
  - ☝️ **Pointing:** Start drawing in mid-air!
- **Privacy Controls:** Features like **Primary Face Lock** and **Blur Others** ensure only you remain in clear focus, securing your environment.
- **Live AI Telemetry:** Monitor pipeline performance in real-time, including FPS, Gesture Latency, Face Tracking status, and current modes.

---

## 📸 See it in Action

Check out GhostLens manipulating reality in real-time:

### Background Blur & Gesture Control
![Background Blur](screenshots/ghostlens_20260701_171217.jpg)
*Open Palm gesture detected and triggering background blur.*

### Gesture-Triggered Screenshots (Thumbs Up)
![Thumbs Up Detection](screenshots/ghostlens_20260701_171226.jpg)
*A simple "Thumbs Up" instantly snaps a screenshot!*

### Real-Time Air Drawing
![Air Drawing](screenshots/ghostlens_20260701_171229.jpg)
*Pointing your index finger lets you draw directly on the live feed.*

### Full Invisibility Cloak
![Invisibility Mode](screenshots/ghostlens_20260701_171232.jpg)
*Full background segmentation removing the subject from the frame.*

---

## 🏗️ System Architecture

GhostLens uses a highly optimized concurrent architecture to ensure high framerates while running multiple deep learning models simultaneously.

```mermaid
graph TD
    A[Webcam Input] -->|cv2.VideoCapture| B(Vision Pipeline)
    B --> C{MediaPipe ML Engine}
    
    C -->|Hand Tracking| D[Gesture Engine]
    C -->|Selfie Segmentation| E[Background Segmenter]
    C -->|Face Mesh| F[Face Tracker]
    
    D --> G[AR Overlays & Air Drawing]
    E --> H[Privacy Controls & Invisibility]
    F --> I[Primary Face Lock]
    
    G --> J(Streamlit UI / Dashboard)
    H --> J
    I --> J
    
    style J fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A working webcam

### Installation
1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/PRATIKSK7/GhostLens.git
   cd GhostLens
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit Application:
   ```bash
   streamlit run app.py
   ```

---

## 🎮 Controls & Configuration

GhostLens can be controlled fully via gestures or through the intuitive Streamlit sidebar!
- Toggle **Demo Mode** for a clean, distraction-free presentation UI.
- Adjust **Blur Strength** dynamically to suit your environment.
- Control **Drawing Lifespan** for how long your mid-air art persists (e.g., 5 Seconds, Infinite).
- Enable **Face Follow Mode** to keep the camera tightly focused on you.

---

### Built with ❤️ using Python, OpenCV, MediaPipe, and Streamlit.
