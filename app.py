import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from fpdf import FPDF
from datetime import datetime
import tempfile
import os

def analyze_with_mediapipe(image_path):
    mp_hands = mp.solutions.hands
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return None
        return results.multi_hand_landmarks[0]

def detect_lines_opencv(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lines_detected = {"life_line": False, "fate_line": False, "heart_line": False}
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 80 and y > 100 and x < 200:
            lines_detected["life_line"] = True
        if h > 100 and x > 150 and x < 300:
            lines_detected["fate_line"] = True
        if y < 150 and w > 100:
            lines_detected["heart_line"] = True
    return lines_detected

def generate_pdf(name, lines_detected, file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Palmistry Report (Web AI)", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Name: {name}", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%B %d, %Y')}", ln=True)
    pdf.ln(10)

    if lines_detected["life_line"]:
        pdf.multi_cell(0, 10, "- Life Line: Strong vitality and energy.")
    else:
        pdf.multi_cell(0, 10, "- Life Line: Not clearly visible.")

    if lines_detected["fate_line"]:
        pdf.multi_cell(0, 10, "- Fate Line: Clear career path, maybe foreign influence.")
    else:
        pdf.multi_cell(0, 10, "- Fate Line: Career may be unconventional.")

    if lines_detected["heart_line"]:
        pdf.multi_cell(0, 10, "- Heart Line: Emotionally mature, values love.")
    else:
        pdf.multi_cell(0, 10, "- Heart Line: Emotion expression may be hidden.")

    pdf.multi_cell(0, 10, "\nThis is an auto-generated AI-based palm reading. For personal insight only.")
    pdf.output(file_path)

# ----------------------------
# Streamlit App Starts Here
# ----------------------------

st.title("🖐️ Palmistry AI - Web Edition")
st.write("Upload your palm image (right hand) to get a palmistry reading and PDF report.")

name = st.text_input("Your Name")
uploaded_file = st.file_uploader("Upload Palm Image", type=["jpg", "jpeg", "png"])

if uploaded_file and name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    st.image(temp_path, caption="Uploaded Palm Image", use_column_width=True)
    with st.spinner("Analyzing your palm..."):
        lines = detect_lines_opencv(temp_path)
        landmarks = analyze_with_mediapipe(temp_path)

    st.success("Analysis complete!")
    st.write("### Your Palmistry Highlights")
    if lines["life_line"]:
        st.write("✅ **Life Line**: Strong vitality")
    else:
        st.write("❌ **Life Line**: Not detected")

    if lines["fate_line"]:
        st.write("✅ **Fate Line**: Clear career direction")
    else:
        st.write("❌ **Fate Line**: May be self-driven")

    if lines["heart_line"]:
        st.write("✅ **Heart Line**: Emotionally balanced")
    else:
        st.write("❌ **Heart Line**: Not clearly visible")

    pdf_path = os.path.join(tempfile.gettempdir(), "Palmistry_Report.pdf")
    generate_pdf(name, lines, pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Your Palmistry Report (PDF)", f, file_name="Palmistry_Report.pdf")