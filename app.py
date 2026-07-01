import streamlit as st # Force reload
import time
import cv2
import os
import datetime
import numpy as np
import psutil
import threading

from vision.camera import CameraController
from vision.gestures import GestureEngine
from vision.segmentation import BackgroundSegmenter
from vision.focus_box import FocusBox
from vision.face_tracker import FaceTracker
from vision.overlays import AROverlay
try:
    import importlib
    import ui.dashboard as dashboard
    importlib.reload(dashboard)
    # Startup validation
    required_functions = [
        "render_demo_section",
        "render_feature_section",
        "render_architecture_section",
        "render_performance_section"
    ]
    for func in required_functions:
        if not hasattr(dashboard, func):
            st.error(f"Cache Error: Module ui.dashboard is missing '{func}'. Please restart Streamlit.")
            st.stop()
except ImportError as e:
    print(f"[ERROR] UI Import failed: {e}")
    class DummyDashboard:
        def __getattr__(self, name):
            if name == "render_ai_info":
                return lambda *args, **kwargs: ""
            return lambda *args, **kwargs: None
    dashboard = DummyDashboard()

def is_valid_point(p):
    return (
        isinstance(p, (tuple, list))
        and len(p) == 2
        and all(isinstance(v, (int, float)) for v in p)
    )

def safe_point(p):
    if not is_valid_point(p):
        return None
    return (int(p[0]), int(p[1]))

def safe_rectangle(frame, pt1, pt2, color, thickness=-1, lineType=cv2.LINE_8):
    try:
        p1 = safe_point(pt1)
        p2 = safe_point(pt2)
        
        if p1 is None or p2 is None:
            return frame

        print("PT1 =", p1)
        print("PT2 =", p2)

        cv2.rectangle(frame, p1, p2, color, thickness, lineType)
    except Exception as e:
        print("Rectangle Error:", e)
    return frame

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="GhostLens v1.0", layout="wide")


# ============================================================
# SESSION STATE — Initialize ALL variables ONCE
# ============================================================
def init_session_state():
    """Initialize every session state variable with safe defaults."""
    defaults = {
        'cam': CameraController(),
        'gestures': GestureEngine(),
        'segmenter': BackgroundSegmenter(),
        'focus': FocusBox(),
        'face': FaceTracker(),
        'overlay': AROverlay(),
        'bg_captured': False,
        'screenshot_cooldown': 0.0,
        'screenshot_count': 0,
        'fps_history': [],
        'conf_history': [],
        'app_state': {
            'running': True,
            'current_mode': 'NORMAL_MODE',
            'pointer_points': [],
            'trail': [],
            'take_screenshot': False,
            # Rectangle assist
            'rect_corner_a': None,
            'rect_corner_b': None,
            'rect_active': False,
            'pinch_was_active': False,
            # Gesture hold
            'last_mode_time': time.time(),
            'mode_hold_timeout': 0.6,
            'onboarding_shown': False,
            'notification': '',
            'notification_time': 0.0,
            # Face Follow Mode
            'face_follow': False,
            'face_follow_box': None,
            'face_follow_lost_time': 0.0,
        },
        'demo_mode': False,
        'blur_level': 3,
        'drawing_lifespan': "5 Seconds",
        'profile': {
            'cam_ms': 0.0,
            'gesture_ms': 0.0,
            'face_ms': 0.0,
            'seg_ms': 0.0,
            'overlay_ms': 0.0,
            'frames': 0,
        },
        'module_status': {
            'camera': '🟢 Active',
            'gesture': '🟢 Active',
            'face': '🟢 Active',
            'effects': '🟢 Active',
        },
        'ai_quality': 'Balanced',
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# Local aliases for performance (avoid dict lookups in hot loop)
cam = st.session_state.cam
gestures = st.session_state.gestures
segmenter = st.session_state.segmenter
focus = st.session_state.focus
face = st.session_state.face
overlay = st.session_state.overlay
state = st.session_state.app_state
prof = st.session_state.profile

# Imports moved to top
# Load all Custom CSS immediately
try:
    dashboard.load_css()
except Exception as e:
    st.error(f"CSS Error: {e}")

# ============================================================
# SECTION 1: HERO BANNER
# ============================================================
try:
    dashboard.render_hero_section()
except Exception as e:
    st.error(f"Hero Section Error: {e}")

# ============================================================
# BACKGROUND CAPTURE / LAUNCH SEQUENCE
# ============================================================
if not st.session_state.bg_captured:
    st.markdown("<div style='text-align: center; color: #8F9BB3; margin-bottom: 20px;'>⚠️ Ensure the background is clear before launching.</div>", unsafe_allow_html=True)
    col_l, col_btn, col_r = st.columns([1, 1, 1])
    with col_btn:
        if st.button("🚀 Launch Demo", use_container_width=True):
            with st.spinner("Initializing AI Vision Engine..."):
                if cam.start():
                    bg = cam.capture_background()
                    if bg is not None:
                        segmenter.bg_frame = bg
                        st.session_state.bg_captured = True
                        st.rerun()
                    else:
                        st.error("❌ Failed to capture background. Check camera connection.")
                else:
                    st.error(f"❌ Camera error: {cam.last_error}")
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    # Render the rest of the landing page even if camera isn't active
    for section_name, render_func in [
        ("Demo Placeholder", dashboard.render_demo_section),
        ("Core Features", dashboard.render_feature_section),
        ("Gesture Guide", dashboard.render_gesture_guide),
        ("Performance Metrics", dashboard.render_performance_section),
        ("Tech Stack", dashboard.render_tech_stack),
        ("Architecture Diagram", dashboard.render_architecture_section),
        ("Showcase", dashboard.render_showcase_section)
    ]:
        try:
            render_func()
        except Exception as e:
            st.error(f"{section_name} Section Error: {e}")
            
    st.stop()

# ============================================================
# SECTION 2: MAIN PRODUCT AREA
# ============================================================

col_main, col_right = st.columns([7, 3])

with col_main:
    frame_placeholder = st.empty()

with col_right:
    # Demo Mode Toggle
    st.markdown("#### 🎬 Presentation")
    demo_toggled = st.toggle("Demo Mode (Clean UI)", value=st.session_state.demo_mode)
    if demo_toggled != st.session_state.demo_mode:
        st.session_state.demo_mode = demo_toggled
        st.rerun()

    st.markdown("#### ⚡ Live AI Status")
    ai_info_placeholder = st.empty()
    
    if not st.session_state.demo_mode:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ⚙️ Privacy Controls")
        
        # Render Face Follow checkbox and update state immediately
        face_follow_toggled = st.toggle("🎯 Face Follow Mode", value=st.session_state.app_state.get('face_follow', False))
        if face_follow_toggled != st.session_state.app_state.get('face_follow', False):
            st.session_state.app_state['face_follow'] = face_follow_toggled
            
        face_lock_toggled = st.toggle("🔒 Primary Face Lock", value=st.session_state.app_state.get('face_lock', True))
        if face_lock_toggled != st.session_state.app_state.get('face_lock', True):
            st.session_state.app_state['face_lock'] = face_lock_toggled
            
        blur_others_toggled = st.toggle("🫥 Blur Others Only", value=st.session_state.app_state.get('blur_others', True))
        if blur_others_toggled != st.session_state.app_state.get('blur_others', True):
            st.session_state.app_state['blur_others'] = blur_others_toggled
    
        st.slider("Blur Strength", min_value=1, max_value=5, value=3, key="blur_level")
        st.selectbox("Drawing Lifespan", ["5 Seconds", "10 Seconds", "Permanent"], key="drawing_lifespan")
        st.selectbox("AI Quality", ["High Performance", "Balanced", "Quality"], index=1, key="ai_quality")
        
        btn_reset = st.button("🔄 Reset Region", use_container_width=True)
        btn_focus = st.button("🎯 Toggle Focus", use_container_width=True)
    else:
        btn_reset = False
        btn_focus = False

# Handle buttons BEFORE the loop
if btn_reset:
    focus.clear()
    state['pointer_points'].clear()
    state['trail'].clear()
    state['rect_corner_a'] = None
    state['rect_corner_b'] = None
    state['rect_active'] = False

if btn_focus:
    focus.active = not focus.active

profile_placeholder = st.empty()

# Render Landing Page below Camera
st.markdown("<br><br>", unsafe_allow_html=True)
for section_name, render_func in [
    ("Core Features", dashboard.render_feature_section),
    ("Gesture Guide", dashboard.render_gesture_guide),
    ("Performance Metrics", dashboard.render_performance_section),
    ("Tech Stack", dashboard.render_tech_stack),
    ("Architecture Diagram", dashboard.render_architecture_section),
    ("Showcase", dashboard.render_showcase_section)
]:
    try:
        render_func()
    except Exception as e:
        st.error(f"{section_name} Section Error: {e}")

def set_notification(state, msg, current_time):
    state['notification'] = msg
    state['notification_time'] = current_time

def save_screenshot_async(frame, fname):
    try:
        cv2.imwrite(fname, frame)
        print(f"[Screenshot] Saved async: {fname}")
    except Exception as e:
        print(f"[Screenshot] Error saving: {e}")

def handle_gesture(gesture, confidence, pointer, state, focus, original_frame, current_time):
    """
    Action Dispatcher & State Machine Engine.
    Processes the raw gesture and executes explicit state transitions.
    """
    print(f"Detected: {gesture} ({confidence:.2f})")
    
    if gesture != "NONE":
        state['onboarding_shown'] = True

    # Reset pinch flag if not pinching
    if gesture != "PINCH":
        state['pinch_was_active'] = False
        
    # 0. Handle CLOSED_FIST (Reset Privacy Box)
    if gesture == "CLOSED_FIST":
        if state['current_mode'] == "PRIVACY_RECT" or len(focus.points) > 0:
            focus.clear()
            state['current_mode'] = "NORMAL_MODE"
            state['rect_state'] = "INACTIVE"
            set_notification(state, "Privacy Window Closed", current_time)
            return

    # 1. Handle THUMBS_UP Screenshot (Immediate Action)
    if gesture == "THUMBS_UP" and confidence > 0.85:
        if (current_time - st.session_state.screenshot_cooldown) > 3.0:
            print("Executing: save_screenshot_async()")
            os.makedirs("screenshots", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"screenshots/ghostlens_{ts}.jpg"
            # Launch background thread to save without freezing UI
            threading.Thread(target=save_screenshot_async, args=(original_frame.copy(), fname), daemon=True).start()
            st.session_state.screenshot_count += 1
            st.session_state.screenshot_cooldown = current_time
            set_notification(state, "Screenshot Saved", current_time)
        return
        
    # 2. Handle OPEN_PALM (Destroy Windows / Reset)
    if gesture == "OPEN_PALM":
        print("Executing: activate_blur() or reset_window()")
        if len(focus.points) > 0 or state.get('rect_state', 'INACTIVE') != "INACTIVE":
            focus.clear()
            state['current_mode'] = "NORMAL_MODE"
            state['rect_state'] = "INACTIVE"
            state['pointer_points'].clear()
            set_notification(state, "Privacy Window Removed", current_time)
        else:
            state['current_mode'] = "BACKGROUND_BLUR"
        return
        
    # 3. Handle PRIVACY_RECT_CREATE State Machine
    if gesture == "PRIVACY_RECT_CREATE":
        print("Executing: activate_rectangle_mode_live()")
        if state['current_mode'] != "PRIVACY_RECT":
            if not is_valid_point(pointer[0]) or not is_valid_point(pointer[1]):
                return
            pt1 = safe_point(pointer[0])
            pt2 = safe_point(pointer[1])
            if pt1 is None or pt2 is None:
                return
                
            state['rect_state'] = "CREATING"
            state['live_rect'] = (pt1, pt2)
            
            if 'rect_create_start' not in state or state.get('rect_create_start') == 0:
                state['rect_create_start'] = current_time
                state['last_rect_pts'] = (pt1, pt2)
                set_notification(state, "Hold fingers stable to lock window", current_time)
            else:
                old_pt1, old_pt2 = state.get('last_rect_pts', (pt1, pt2))
                old_pt1 = safe_point(old_pt1)
                old_pt2 = safe_point(old_pt2)
                
                if old_pt1 and old_pt2:
                    dist1 = ((pt1[0]-old_pt1[0])**2 + (pt1[1]-old_pt1[1])**2)**0.5
                    dist2 = ((pt2[0]-old_pt2[0])**2 + (pt2[1]-old_pt2[1])**2)**0.5
                    
                    # Reset stabilization timer if fingers move too much
                    if dist1 > 25 or dist2 > 25:
                        state['rect_create_start'] = current_time
                        
                state['last_rect_pts'] = (pt1, pt2)
                
                if (current_time - state['rect_create_start']) >= 1.0:
                    state['current_mode'] = "PRIVACY_RECT"
                    state['rect_state'] = "ACTIVE"
                    state['rect_create_start'] = 0
                    
                    # Lock focus box
                    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
                    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
                    focus.clear()
                    focus.add_point(x1, y1)
                    focus.add_point(x2, y1)
                    focus.add_point(x2, y2)
                    focus.add_point(x1, y2)
                    set_notification(state, "Privacy Window Locked", current_time)
        return
    else:
        state['rect_create_start'] = 0
        state['live_rect'] = None
        
    # 4. Handle POINTING Mode
    if gesture == "POINTING":
        print("Executing: activate_draw_mode()")
        # Do not override VICTORY mode if we are building a rectangle
        if state.get('rect_state', 'INACTIVE') in ("SELECTING_A", "SELECTING_B", "ACTIVE"):
            pass
        else:
            state['current_mode'] = "DRAWING"
            state['rect_state'] = "INACTIVE"
        return
        
    # 5. Handle PINCH Action (Contextual depending on state)
    if gesture == "PINCH":
        print("Executing: contextual_pinch_action()")
        if not pointer:
            return
            
        mode = state['current_mode']
        
        if mode == "PRIVACY_RECT":
            # Pinch to interact with privacy rect was removed in favor of purely live-drawing
            pass

        elif mode == "DRAWING":
            if not state.get('pinch_was_active', False):
                state['pinch_was_active'] = True
                pts = state['pointer_points']
                if len(pts) > 0:
                    last = pts[-1][:2]
                    dist = ((pointer[0]-last[0])**2 + (pointer[1]-last[1])**2)**0.5
                    if dist > 15:
                        dx = pointer[0] - last[0]
                        dy = pointer[1] - last[1]
                        if abs(dx) < 25 and abs(dy) > 25:
                            pts.append((last[0], pointer[1], current_time))
                        elif abs(dy) < 25 and abs(dx) > 25:
                            pts.append((pointer[0], last[1], current_time))
                        else:
                            pts.append((pointer[0], pointer[1], current_time))
                        focus.add_point(pts[-1][0], pts[-1][1])
                else:
                    pts.append((pointer[0], pointer[1], current_time))
                    focus.add_point(pointer[0], pointer[1])
        else:
            state['current_mode'] = "FULL_INVISIBILITY"
        return
        
    # 6. Handle NONE (Timeout decay)
    if gesture == "NONE":
        # Only decay NORMAL_MODE if we are NOT in a sticky state
        if state['current_mode'] not in ("PRIVACY_RECT", "DRAWING"):
            if (current_time - state.get('last_mode_time', current_time)) > state['mode_hold_timeout']:
                state['current_mode'] = "NORMAL_MODE"

# ============================================================
# MAIN PROCESSING LOOP
# ============================================================
DASH_INTERVAL = 0.5      # Dashboard update interval (seconds)
PROFILE_INTERVAL = 5.0   # Profiling print interval

last_dash_update = 0.0
last_profile_time = 0.0
loop_counter = 0

cached_gesture = "NONE"
cached_confidence = 0.0
cached_pointer = None
cached_seg_mask = None

if state['running']:
    try:
        while True:
            # ═══════════ TIMING & STATE ═══════════
            t0 = time.time()
            current_time = time.time()
            loop_counter += 1
            
            # ═══════════ CAMERA ═══════════
            try:
                frame = cam.read_frame()
                if frame is None:
                    if not cam.is_running:
                        cam.start()
                    time.sleep(0.01)
                    continue
                st.session_state.module_status['camera'] = '🟢 Active'
            except Exception as e:
                print(f"[ERROR] Camera Crashed: {e}")
                st.session_state.module_status['camera'] = '🔴 Recovering'
                continue
                
            t_cam = time.time()
            original = frame.copy()
            
            q_map = {"High Performance": (640, 360), "Balanced": (960, 540), "Quality": (1280, 720)}
            ai_res = q_map.get(st.session_state.get("ai_quality", "Balanced"), (960, 540))
            
            # ═══════════ GESTURE DETECTION (Fault Tolerant) ═══════════
            if loop_counter % 2 != 0:
                try:
                    frame, gesture, confidence, seg_mask, pointer = gestures.detect(frame, ai_res=ai_res)
                    cached_gesture = gesture
                    cached_confidence = confidence
                    cached_seg_mask = seg_mask
                    cached_pointer = pointer
                    st.session_state.module_status['gesture'] = '🟢 Active'
                except Exception as e:
                    print(f"[ERROR] Gesture Detection Crashed: {e}")
                    gesture, confidence, pointer, seg_mask = "NONE", 0.0, None, None
                    st.session_state.module_status['gesture'] = '🔴 Recovering'
                    set_notification(state, "⚠ Gesture Module Recovering...", current_time)
            else:
                gesture = cached_gesture
                confidence = cached_confidence
                seg_mask = cached_seg_mask
                pointer = cached_pointer
            
            t_gesture = time.time()
            
            # ═══════════ DISPATCH ACTIONS ═══════════
            try:
                handle_gesture(gesture, confidence, pointer, state, focus, original, current_time)
            except Exception as e:
                print(f"[ERROR] Dispatcher Crashed: {e}")
            
            mode = state['current_mode']
            
            # ═══════════ FACE TRACKER (Fault Tolerant) ═══════════
            try:
                face.enabled = True
                face.face_lock_enabled = state.get('face_lock', True)
                face.blur_others_only = state.get('blur_others', True)
                frame, primary_bbox, face_count, face_status, blurred_count = face.apply(original, frame, ai_res=ai_res)
                st.session_state.module_status['face'] = '🟢 Active'
            except Exception as e:
                print(f"[ERROR] Face Tracker Crashed: {e}")
                primary_bbox, face_count, face_status, blurred_count = None, 0, "Error", 0
                st.session_state.module_status['face'] = '🔴 Recovering'
                set_notification(state, "⚠ Face Tracker Recovering...", current_time)
                
            t_face = time.time()
            
            # ═══════════ CURSOR DISPLAY & LASER ═══════════
            if mode in ("DRAWING", "PRIVACY_RECT"):
                if pointer:
                    if isinstance(pointer, (list, tuple)) and len(pointer) == 2 and isinstance(pointer[0], (int, float)) and isinstance(pointer[1], (int, float)):
                        state['trail'].append((int(pointer[0]), int(pointer[1])))
                        
                    if len(state['trail']) > 15:
                        state['trail'].pop(0)
                        
                    valid_trail = []
                    for p in state['trail']:
                        if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                            valid_trail.append([int(p[0]), int(p[1])])
                            
                    if len(valid_trail) > 2:
                        print("POINTS:", valid_trail)
                        pts = np.array(valid_trail, np.int32)
                        cv2.polylines(frame, [pts], False, (0, 0, 255), 2)
                        
                    if isinstance(pointer, (list, tuple)) and len(pointer) == 2 and isinstance(pointer[0], (int, float)):
                        pt_draw = (int(pointer[0]), int(pointer[1]))
                        cv2.circle(frame, pt_draw, 14, (0, 150, 255), 2)
                        cv2.circle(frame, pt_draw, 8, (0, 0, 255), -1)
                        cv2.circle(frame, pt_draw, 4, (255, 255, 255), -1)
            else:
                state['trail'].clear()
            
            # ═══════════ PRIVACY RECTANGLE LIVE PREVIEW ═══════════
            r_state = state.get('rect_state', 'INACTIVE')
            if r_state == "CREATING" and state.get('live_rect'):
                pt1, pt2 = state['live_rect']
                s_pt1 = safe_point(pt1)
                s_pt2 = safe_point(pt2)
                
                if s_pt1 and s_pt2:
                    safe_rectangle(frame, s_pt1, s_pt2, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    # Draw locking progress arc
                    lock_duration = current_time - state.get('rect_create_start', current_time)
                    progress = min(1.0, lock_duration / 1.0)
                    cx = (s_pt1[0] + s_pt2[0]) // 2
                    cy = (s_pt1[1] + s_pt2[1]) // 2
                    radius = 30
                    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, 360 * progress, (0, 255, 0), 4)
                else:
                    state['rect_state'] = 'INACTIVE'
            
            # Privacy HUD
            if mode == "PRIVACY_RECT" and len(focus.points) >= 4:
                pts = focus.points
                x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
                x2, y2 = max(p[0] for p in pts), max(p[1] for p in pts)
                w_rect = x2 - x1
                h_rect = y2 - y1
                area = w_rect * h_rect
                
                hud_x, hud_y = 30, 150
                safe_rectangle(frame, (hud_x-10, hud_y-30), (hud_x+250, hud_y+80), (0, 0, 0), -1)
                cv2.putText(frame, "PRIVACY RECTANGLE ACTIVE", (hud_x, hud_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Width: {w_rect} px", (hud_x, hud_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Height: {h_rect} px", (hud_x, hud_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Area: {area} px^2", (hud_x, hud_y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # ═══════════ DRAWING MODE RENDER ═══════════
            if mode == "DRAWING":
                lifespan_str = st.session_state.get('drawing_lifespan', "5 Seconds")
                if lifespan_str != "Permanent":
                    max_age = 5.0 if lifespan_str == "5 Seconds" else 10.0
                    valid_pts = []
                    valid_focus = []
                    for i, p in enumerate(state['pointer_points']):
                        if (current_time - p[2]) < max_age:
                            valid_pts.append(p)
                            if i < len(focus.points):
                                valid_focus.append(focus.points[i])
                    state['pointer_points'] = valid_pts
                    focus.points = valid_focus
                
                pts = state['pointer_points']
                if len(pts) > 1:
                    # Draw with smooth alpha fading based on point age
                    overlay_draw = frame.copy()
                    for i in range(1, len(pts)):
                        p1 = pts[i-1][:2]
                        p2 = pts[i][:2]
                        pt_time = pts[i][2]
                        age = current_time - pt_time
                        
                        # Calculate alpha: 1.0 (new) to 0.0 (old)
                        if max_age == 10.0:  # If 10 seconds, fade slower
                            alpha = max(0.0, 1.0 - (age / max_age))
                        elif max_age == 5.0:
                            alpha = max(0.0, 1.0 - (age / max_age))
                        else:
                            alpha = 1.0
                            
                        # Fade towards Cyan (0, 255, 255)
                        color = (0, int(255 * alpha), int(255 * alpha))
                        cv2.line(overlay_draw, p1, p2, color, 3, cv2.LINE_AA)
                    
                    # Blend faded lines onto frame
                    cv2.addWeighted(overlay_draw, 1.0, frame, 0.0, 0, frame)
                    
                    if pointer:
                        cv2.line(frame, pts[-1][:2], pointer, (0, 150, 255), 2, cv2.LINE_AA)
            
            # Smart Rectangle Assist
            if gesture == "CLOSED_FIST" and len(state['pointer_points']) > 3:
                pts_xy = [p[:2] for p in state['pointer_points']]
                valid_points = []
                for p in pts_xy:
                    if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                        valid_points.append([int(p[0]), int(p[1])])
                
                if len(valid_points) >= 4:
                    print("POINTS:", valid_points)
                    pts_arr = np.array(valid_points, np.int32)
                    epsilon = 0.02 * cv2.arcLength(pts_arr, True)
                approx = cv2.approxPolyDP(pts_arr, epsilon, True)
                
                if len(approx) >= 4:
                    rect = cv2.minAreaRect(approx)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    area_approx = cv2.contourArea(approx)
                    area_box = cv2.contourArea(box)
                    if area_box > 0 and area_approx / area_box > 0.7:
                        approx = box.reshape(-1, 1, 2)
                
                state['pointer_points'] = [(int(p[0][0]), int(p[0][1]), current_time) for p in approx]
                focus.clear()
                for p in state['pointer_points']:
                    focus.add_point(p[0], p[1])
            
            # ═══════════ APPLY EFFECTS (Fault Tolerant) ═══════════
            try:
                if mode == "NORMAL_MODE":
                    pass
                elif mode == "FULL_INVISIBILITY":
                    frame = segmenter.apply_invisibility(frame, pose_segmentation=seg_mask)
                elif mode == "BACKGROUND_BLUR":
                    level = st.session_state.get('blur_level', 3)
                    frame = segmenter.apply_ultra_blur(frame, pose_segmentation=seg_mask, blur_level=level)
                elif mode == "PRIVACY_RECT":
                    if len(focus.points) > 2:
                        frame = focus.apply(frame, meeting_mode=True)
                elif mode == "DRAWING":
                    if len(focus.points) > 2:
                        frame = focus.apply(frame, segmenter.bg_frame)
                st.session_state.module_status['effects'] = '🟢 Active'
            except Exception as e:
                print(f"[ERROR] Effects Crashed: {e}")
                st.session_state.module_status['effects'] = '🔴 Recovering'
                set_notification(state, "⚠ Effects Module Recovering...", current_time)
                
            # ═══════════ FACE FOLLOW MODE ═══════════
            if state.get('face_follow', False):
                if primary_bbox is not None:
                    x, y, fw, fh = primary_bbox
                    target_w = int(2.5 * fw)
                    target_h = int(3.0 * fh)
                    cx = x + fw // 2
                    cy = y + fh // 2
                    target_x = int(cx - target_w / 2)
                    target_y = int(cy - target_h * 0.4) # Shift downwards to keep head and shoulders visible
                    
                    prev_box = state.get('face_follow_box')
                    alpha = 0.2
                    if prev_box is not None:
                        new_x = int(alpha * target_x + (1 - alpha) * prev_box[0])
                        new_y = int(alpha * target_y + (1 - alpha) * prev_box[1])
                        new_w = int(alpha * target_w + (1 - alpha) * prev_box[2])
                        new_h = int(alpha * target_h + (1 - alpha) * prev_box[3])
                    else:
                        new_x, new_y, new_w, new_h = target_x, target_y, target_w, target_h
                        
                    state['face_follow_box'] = (new_x, new_y, new_w, new_h)
                    state['face_follow_lost_time'] = 0.0
                else:
                    if state.get('face_follow_lost_time', 0.0) == 0.0:
                        state['face_follow_lost_time'] = current_time
                        
                    if (current_time - state['face_follow_lost_time']) > 5.0:
                        state['face_follow_box'] = None

                # Render the mask
                f_box = state.get('face_follow_box')
                if f_box is not None:
                    fx, fy, fw, fh = f_box
                    h, w = frame.shape[:2]
                    
                    mask = np.zeros((h, w), dtype=np.uint8)
                    safe_rectangle(mask, (fx, fy), (fx + fw, fy + fh), 255, -1)
                    mask_blur = cv2.GaussianBlur(mask, (51, 51), 0)
                    alpha_mask = mask_blur.astype(np.float32) / 255.0
                    alpha_3d = np.stack([alpha_mask] * 3, axis=-1)
                    
                    if mode == "NORMAL_MODE":
                        # Generate heavy blur for the OUTSIDE
                        small = cv2.resize(original, (w // 4, h // 4))
                        b_small = cv2.GaussianBlur(small, (21, 21), 0)
                        b_small = cv2.GaussianBlur(b_small, (21, 21), 0)
                        bg_blur = cv2.resize(b_small, (w, h))
                        outside = bg_blur.astype(np.float32) * 0.5
                    else:
                        outside = frame.astype(np.float32)
                        
                    frame = (alpha_3d * original.astype(np.float32) + (1.0 - alpha_3d) * outside).astype(np.uint8)
                    
            # ═══════════ ONBOARDING & NOTIFICATIONS ═══════════
            if not state.get('onboarding_shown', False):
                h, w = frame.shape[:2]
                overlay_img = frame.copy()
                safe_rectangle(overlay_img, (w//2 - 250, h//2 - 60), (w//2 + 250, h//2 + 60), (0, 0, 0), -1)
                cv2.addWeighted(overlay_img, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, "Welcome to GhostLens!", (w//2 - 220, h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(frame, "Perform any gesture to begin.", (w//2 - 190, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            if state.get('notification') and (current_time - state.get('notification_time', 0)) < 3.0:
                # Fade out over the last 1 second
                age = current_time - state['notification_time']
                alpha = 1.0 if age < 2.0 else (3.0 - age)
                
                h, w = frame.shape[:2]
                msg = state['notification']
                # Calculate text width
                (text_width, text_height), baseline = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                x = (w - text_width) // 2
                y = h - 60
                
                overlay_img = frame.copy()
                safe_rectangle(overlay_img, (x - 20, y - text_height - 15), (x + text_width + 20, y + 15), (0, 0, 0), -1)
                cv2.addWeighted(overlay_img, alpha * 0.7, frame, 1.0 - (alpha * 0.7), 0, frame)
                
                # We can't easily alpha-blend cv2.putText without creating a separate channel, so we just draw it if alpha > 0.1
                if alpha > 0.1:
                    cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            t_seg = time.time()
            
            # ═══════════ HUD OVERLAY ═══════════
            try:
                frame, fps = overlay.apply(
                    frame, mode=mode, gesture=gesture,
                    hand_conf=confidence, face_count=face_count,
                    face_follow=state.get('face_follow', False)
                )
            except Exception as e:
                print(f"[ERROR] HUD Overlay Crashed: {e}")
                fps = overlay.get_fps() if hasattr(overlay, 'get_fps') else 0
            t_overlay = time.time()
            
            # ═══════════ PROFILING ═══════════
            prof['cam_ms'] += (t_cam - t0) * 1000
            prof['gesture_ms'] += (t_gesture - t_cam) * 1000
            prof['face_ms'] += (t_face - t_gesture) * 1000
            prof['seg_ms'] += (t_seg - t_face) * 1000
            prof['overlay_ms'] += (t_overlay - t_seg) * 1000
            prof['frames'] += 1
            
            # ═══════════ TELEMETRY ═══════════
            st.session_state.fps_history.append(fps)
            st.session_state.conf_history.append(confidence)
            if len(st.session_state.fps_history) > 50:
                st.session_state.fps_history.pop(0)
            if len(st.session_state.conf_history) > 50:
                st.session_state.conf_history.pop(0)
            
            # ═══════════ DASHBOARD UPDATE (500ms throttle) ═══════════
            if (current_time - last_dash_update) >= DASH_INTERVAL:
                last_dash_update = current_time
                try:
                    ai_info_placeholder.markdown(
                        dashboard.render_ai_info(
                            gesture=gesture, 
                            mode=mode, 
                            faces=face_count, 
                            fps=overlay.get_fps(), 
                            face_status=face_status, 
                            blurred_count=blurred_count
                        ),
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    ai_info_placeholder.error("Telemetry unavailable")
                
                # Detailed Profiling HUD
                n = max(1, prof['frames'])
                ac = prof['cam_ms'] / n
                ag = prof['gesture_ms'] / n
                af = prof['face_ms'] / n
                aseg = prof['seg_ms'] / n
                ao = prof['overlay_ms'] / n
                total = ac + ag + af + aseg + ao
                
                # Live System Telemetry
                cpu = psutil.cpu_percent()
                mem = psutil.Process().memory_info().rss / (1024 * 1024)
                
                ms = st.session_state.module_status
                
                profile_text = (
                    f"**Capture**: {ac:.1f}ms  \n"
                    f"**Gesture**: {ag:.1f}ms  \n"
                    f"**Face Track**: {af:.1f}ms  \n"
                    f"**Segmentation**: {aseg:.1f}ms  \n"
                    f"**Render**: {ao:.1f}ms  \n"
                    f"**Total**: {total:.1f}ms  \n"
                    f"---  \n"
                    f"**CPU**: {cpu:.1f}%  \n"
                    f"**RAM**: {mem:.1f} MB  \n"
                    f"---  \n"
                    f"### ⚙️ Diagnostics Panel  \n"
                    f"**Camera**: {ms['camera']}  \n"
                    f"**Gestures**: {ms['gesture']}  \n"
                    f"**Face Track**: {ms['face']}  \n"
                    f"**Effects**: {ms['effects']}"
                )
                if not st.session_state.get('demo_mode', False):
                    try:
                        profile_placeholder.markdown(f"### ⚡ Performance\n{profile_text}")
                    except Exception:
                        profile_placeholder.error("Telemetry unavailable")
            
            # ═══════════ PROFILER PRINT (5s) ═══════════
            if (current_time - last_profile_time) >= PROFILE_INTERVAL and prof['frames'] > 0:
                last_profile_time = current_time
                # Reset
                for k in prof:
                    if k != 'frames':
                        prof[k] = 0
                prof['frames'] = 0
            
            # ═══════════ RENDER FRAME ═══════════
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # ═══════════ FPS THROTTLE & GC ═══════════
            elapsed = time.time() - t0
            sleep_time = (1.0 / 30.0) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            # Explicit cleanup
            del frame
            del original
            if loop_counter % 150 == 0:
                import gc
                gc.collect()
            
    except BrokenPipeError:
        try:
            cam.stop()
        except Exception:
            pass
    except Exception as e:
        import traceback
        try:
            traceback.print_exc()
        except Exception:
            pass
        try:
            st.error(f"⚠️ Pipeline error: {e}")
        except Exception:
            pass
        try:
            cam.stop()
        except Exception:
            pass
else:
    cam.stop()
    st.info("Application stopped. Refresh to restart.")
