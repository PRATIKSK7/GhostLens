import streamlit as st
import cv2
import numpy as np
import threading
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx

try:
    from main import PhantomFrame
except ImportError:
    st.error("Failed to import PhantomFrame from main.py. Make sure you are in the correct directory.")
    st.stop()

st.set_page_config(
    page_title="PhantomFrame", 
    page_icon="👻",
    layout="wide"
)

st.title("👻 PhantomFrame - AI Vision Control Panel")

# 1. Initialization
if 'pf' not in st.session_state:
    st.session_state.pf = PhantomFrame(init_camera=False)
    st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    st.session_state.cap.set(cv2.CAP_PROP_FPS, 30)
    st.session_state.bg_captured = False

pf = st.session_state.pf
cap = st.session_state.cap

# 2. Background Capture Phase
if not st.session_state.bg_captured:
    st.warning("Please step out of frame. We need a clean background!")
    if st.button("Start Background Capture"):
        with st.spinner("Capturing 30 frames (Wait 1 second)..."):
            bg = pf.bg_capture.capture(cap, num_frames=30)
            st.session_state.bg_captured = True
            st.rerun()
    st.stop()

# 3. Sidebar Controls
with st.sidebar:
    st.header("🎮 Mode Controls")
    
    invisibility = st.selectbox(
        "Invisibility Mode",
        ["None", "Partial (50%)", "Full Invisibility"]
    )
    
    focus_mode = st.toggle("Focus Box Mode", value=pf.focus_mode)
    face_anchor = st.toggle("Keep Face Visible", value=pf.face_anchor_on)
    depth_mode = st.toggle("MiDaS AI Masking", value=pf.depth_mode)
    recording = st.toggle("Record Output", value=pf.recorder.is_recording)
    
    # Apply UI State to PhantomFrame
    if invisibility == "None":
        pf.mode = "NORMAL"
    elif invisibility == "Partial (50%)":
        pf.mode = "PARTIAL"
    elif invisibility == "Full Invisibility":
        pf.mode = "INVISIBLE"

    pf.focus_mode = focus_mode
    pf.face_anchor_on = face_anchor
    
    if depth_mode and not pf.depth_mode:
        pf._toggle_depth_mode()
    elif not depth_mode and pf.depth_mode:
        pf._toggle_depth_mode()
        
    if recording and not pf.recorder.is_recording:
        pf.recorder.toggle(1280, 720)
    elif not recording and pf.recorder.is_recording:
        pf.recorder.toggle(1280, 720)

    st.divider()
    st.header("📊 Stats")
    fps_display = st.empty()
    mode_display = st.empty()

# 4. Main UI Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Live Feed")
    frame_placeholder = st.empty()

with col2:
    st.subheader("Background")
    bg_placeholder = st.empty()
    if pf.bg_capture.bg_frame is not None:
        bg_rgb = cv2.cvtColor(pf.bg_capture.bg_frame, cv2.COLOR_BGR2RGB)
        bg_placeholder.image(bg_rgb, use_container_width=True)
    
    st.markdown("---")
    if st.button("🛑 Stop Application"):
        st.session_state.run_thread = False
        if 'thread' in st.session_state:
            st.session_state.thread.join()
            del st.session_state.thread
        pf.cleanup()
        st.success("Application stopped safely. You can close this tab.")
        st.stop()

# 5. Background Thread for Camera
def run_camera_thread():
    while st.session_state.get('run_thread', True):
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue
            
        frame = cv2.flip(frame, 1)
        
        # Process frame
        result = pf.process_frame(frame)
        
        # Convert BGR to RGB for Streamlit
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # Update UI Elements safely
        try:
            frame_placeholder.image(result_rgb, channels="RGB", use_container_width=True)
            fps_display.metric("FPS", f"{pf.hud.get_fps():.1f}")
            mode_display.metric("Active Mode", pf.mode)
        except Exception:
            # Catch exceptions if context is lost
            pass
            
        time.sleep(0.01) # Small sleep to prevent maxing out CPU

# Start thread if not already running
if 'thread' not in st.session_state:
    st.session_state.run_thread = True
    thread = threading.Thread(target=run_camera_thread)
    add_script_run_ctx(thread)
    thread.start()
    st.session_state.thread = thread
