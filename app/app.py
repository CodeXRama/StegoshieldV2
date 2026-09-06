
import os
from pathlib import Path
import socket
import sys
import threading
import time

import streamlit as st
import streamlit.components.v1 as components

# Setup system path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Minimalist clean launcher



def is_port_in_use(port: int = 8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def ensure_backend_server():
    if not is_port_in_use(8000):
        try:
            import uvicorn
            t = threading.Thread(
                target=lambda: uvicorn.run(
                    "app.web_server:app",
                    host="127.0.0.1",
                    port=8000,
                    log_level="warning",
                ),
                daemon=True,
            )
            t.start()
            time.sleep(1.5)
        except Exception as e:
            print(f"[WARN] Could not auto-launch background web server: {e}")


st.set_page_config(
    page_title="STEGOSHIELD // Image Forensics Workstation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ensure_backend_server()

st.markdown(
    """
    <style>
        /* Hide Streamlit default branding, header, sidebar, and padding */
        [data-testid="stHeader"] { display: none !important; }
        [data-testid="stSidebar"] { display: none !important; }
        footer { display: none !important; }
        #MainMenu { visibility: hidden !important; }
        .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
            height: 100vh !important;
            overflow: hidden !important;
        }
        .main {
            padding: 0 !important;
            margin: 0 !important;
            background-color: #0d0e10 !important;
        }
        iframe {
            border: none !important;
            width: 100vw !important;
            height: 100vh !important;
            min-height: 100vh !important;
            display: block !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

components.iframe("http://127.0.0.1:8000", height=1080, scrolling=True)
