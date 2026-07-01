import streamlit as st # Streamlit Hot Reload Bump
import textwrap

def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
        
        #MainMenu {visibility: hidden !important;}
        header {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 4rem !important;
            padding-left: 5% !important;
            padding-right: 5% !important;
            max-width: 1400px !important;
        }
        
        .stApp {
            background-color: #050816;
            color: #FAFAFA;
            font-family: 'Outfit', sans-serif;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(121, 40, 202, 0.15), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(0, 229, 255, 0.15), transparent 25%);
            background-attachment: fixed;
            overflow-x: hidden;
        }

        h1, h2, h3, h4, h5, p, span { font-family: 'Outfit', sans-serif; }

        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            z-index: -1;
            animation: float 10s infinite ease-in-out alternate;
        }
        .orb-1 { width: 400px; height: 400px; background: rgba(121, 40, 202, 0.2); top: -100px; left: -100px; }
        .orb-2 { width: 500px; height: 500px; background: rgba(0, 229, 255, 0.15); bottom: -200px; right: -100px; animation-delay: -5s; }

        @keyframes float {
            0% { transform: translateY(0) translateX(0); }
            100% { transform: translateY(50px) translateX(30px); }
        }
        
        .section-header { margin: 80px 0 40px; text-align: center; }
        .section-title { font-size: 3rem; font-weight: 800; margin-bottom: 15px; background: linear-gradient(90deg, #FFFFFF, #A0AAB5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .section-subtitle { color:#8F9BB3; font-size: 1.1rem; max-width: 600px; margin: 0 auto; }
        </style>
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
    """, unsafe_allow_html=True)

def render_hero_section():
    html = textwrap.dedent("""
        <div style="text-align: center; padding: 60px 20px; position: relative; z-index: 10;">
            <h1 style="font-size: 5rem; font-weight: 800; line-height: 1.1; margin-bottom: 24px; background: linear-gradient(135deg, #FFFFFF 0%, #A0AAB5 50%, #7928CA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">GhostLens AI</h1>
            <p style="font-size: 1.35rem; color: #A0AAB5; max-width: 700px; margin: 0 auto 40px;">Real-time gesture-controlled privacy protection using advanced computer vision and neural networks.</p>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <div style="background: rgba(255, 255, 255, 0.05); padding: 10px 24px; border-radius: 40px; border: 1px solid rgba(255,255,255,0.1);">🟢 30+ FPS Processing</div>
                <div style="background: rgba(255, 255, 255, 0.05); padding: 10px 24px; border-radius: 40px; border: 1px solid rgba(255,255,255,0.1);">🎯 99% Gesture Accuracy</div>
                <div style="background: rgba(255, 255, 255, 0.05); padding: 10px 24px; border-radius: 40px; border: 1px solid rgba(255,255,255,0.1);">⚡ <35ms Latency</div>
            </div>
        </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

def render_demo_section():
    # Placeholder for where the demo video or live camera usually goes
    pass

def render_ai_info(fps=0, gesture="NONE", mode="NORMAL", faces=0, face_status="Unknown", blurred_count=0, **kwargs):
    html = f"""
    <div style="background: rgba(20, 20, 25, 0.8); border-radius: 16px; padding: 20px; border: 1px solid rgba(0, 229, 255, 0.3);">
        <h4 style="color: #00E5FF; margin-top: 0;">AI Pipeline Telemetry</h4>
        <div style="display: flex; justify-content: space-between; font-family: monospace;">
            <div>FPS: {fps:.0f}</div>
            <div>Mode: {mode}</div>
            <div>Gesture: {gesture}</div>
            <div>Faces: {faces}</div>
            <div>Status: {face_status}</div>
        </div>
    </div>
    """
    html = "\\n".join([line.strip() for line in html.split("\\n")])
    return html

def render_feature_section():
    html = textwrap.dedent("""
        <div class="section-header">
            <h2 class="section-title">Core Features</h2>
            <p class="section-subtitle">Advanced privacy tools mapped to intuitive hand gestures.</p>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px;">
            <div style="background: rgba(255,255,255,0.05); padding: 30px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 2rem; margin-bottom: 10px;">🛡</div>
                <h3>Privacy Shield</h3>
                <p style="color: #A0AAB5;">Blur all secondary faces automatically. Prevent photobombers from appearing in your feed.</p>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 30px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 2rem; margin-bottom: 10px;">🎯</div>
                <h3>Face Lock</h3>
                <p style="color: #A0AAB5;">Automatically identifies and tracks the primary user seamlessly.</p>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 30px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 2rem; margin-bottom: 10px;">👋</div>
                <h3>Gesture Control</h3>
                <p style="color: #A0AAB5;">Control your privacy settings using intuitive hand gestures.</p>
            </div>
        </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

def render_gesture_guide():
    html = textwrap.dedent("""
        <div class="section-header">
            <h2 class="section-title">Gesture Guide</h2>
        </div>
        <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
            <div style="text-align: center; padding: 20px; background: rgba(0,0,0,0.5); border-radius: 12px; width: 150px;">
                <div style="font-size: 2rem;">✋</div>
                <div>Open Palm</div>
            </div>
            <div style="text-align: center; padding: 20px; background: rgba(0,0,0,0.5); border-radius: 12px; width: 150px;">
                <div style="font-size: 2rem;">👍</div>
                <div>Thumbs Up</div>
            </div>
            <div style="text-align: center; padding: 20px; background: rgba(0,0,0,0.5); border-radius: 12px; width: 150px;">
                <div style="font-size: 2rem;">✌️</div>
                <div>Two Fingers</div>
            </div>
        </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

def render_performance_section():
    html = textwrap.dedent("""
        <div class="section-header">
            <h2 class="section-title">Performance Metrics</h2>
        </div>
        <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
            <div style="padding: 30px; background: rgba(0,229,255,0.1); border: 1px solid rgba(0,229,255,0.3); border-radius: 16px; text-align: center; width: 200px;">
                <h2 style="margin:0; color:#00E5FF;">30+</h2>
                <div style="color:#A0AAB5;">Stable FPS</div>
            </div>
            <div style="padding: 30px; background: rgba(121,40,202,0.1); border: 1px solid rgba(121,40,202,0.3); border-radius: 16px; text-align: center; width: 200px;">
                <h2 style="margin:0; color:#7928CA;">99%</h2>
                <div style="color:#A0AAB5;">Accuracy</div>
            </div>
        </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

def render_tech_stack():
    html = textwrap.dedent("""
        <div class="section-header">
            <h2 class="section-title">Technology Stack</h2>
        </div>
        <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
            <span style="padding: 8px 16px; background: rgba(255,255,255,0.1); border-radius: 8px;">OpenCV</span>
            <span style="padding: 8px 16px; background: rgba(255,255,255,0.1); border-radius: 8px;">MediaPipe</span>
            <span style="padding: 8px 16px; background: rgba(255,255,255,0.1); border-radius: 8px;">Streamlit</span>
            <span style="padding: 8px 16px; background: rgba(255,255,255,0.1); border-radius: 8px;">NumPy</span>
        </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

def render_architecture_section():
    html = textwrap.dedent("""
        <div class="section-header">
            <h2 class="section-title">Architecture</h2>
        </div>
        <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 12px; margin: 0 auto; max-width: 600px;">
            Camera Feed ➔ Gesture Detection ➔ Face Lock ➔ Blur Engine ➔ Output
        </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

def render_showcase_section():
    html = textwrap.dedent("""
        <div class="section-header">
            <h2 class="section-title">Showcase</h2>
        </div>
        <div style="text-align: center; color: #A0AAB5;">
            Built for Privacy. Designed for Performance.<br><br>
            <a href="#" style="color: #00E5FF; text-decoration: none;">GitHub</a> | <a href="#" style="color: #00E5FF; text-decoration: none;">LinkedIn</a>
        </div>
    """)
    st.markdown(html, unsafe_allow_html=True)
