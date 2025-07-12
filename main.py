import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
import speech_recognition as sr


# Setup AI
genai.configure(api_key="AIzaSyBiiwGiqXGdk367nHb065TFAjdZzfzOcGI")
model = genai.GenerativeModel('gemini-2.0-flash')

# Streamlit config
st.set_page_config(layout="wide")

# Title
st.markdown(
    "<h1 style='text-align: center; color: #ff69b4;'>Project Milk</h1>",
    unsafe_allow_html=True
)

# Layout
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    output_text_area = st.empty()
    st.markdown("### Chatbot Assistant 👇")
    chat_input = st.text_input("Type your question here...")
    if st.button("Ask"):
        if chat_input:
            try:
                reply = model.generate_content(chat_input).text
                output_text_area.subheader(reply)
            except:
                output_text_area.subheader("Error with chatbot.")

    # Voice input
    if st.button("🎙️ Speak Instead"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            reply = model.generate_content(text).text
            output_text_area.subheader(reply)
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand your voice.")
        except sr.RequestError:
            st.error("Speech service error.")

# Setup AI
#genai.configure(api_key="AIzaSyBk6tYLtW6U9LXN67EgA8gA7a2YlYqLC7Y")
#model = genai.GenerativeModel('gemini-1.5-flash')

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector
detector = HandDetector(maxHands=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if canvas is None:
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    if fingers == [0, 1, 0, 0, 0]:
        current_pos = tuple(map(int, lmList[8][0:2]))
        if prev_pos is None:
            prev_pos = current_pos
        if isinstance(prev_pos, tuple) and isinstance(current_pos, tuple):
            cv2.line(canvas, prev_pos, current_pos, (255, 0, 255), 10)
    elif fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(img)

    return current_pos, canvas

def sendToAI(canvas):
    pil_image = Image.fromarray(canvas)
    try:
        response = model.generate_content(["Solve this Math Problem:", pil_image])
        if hasattr(response, 'text'):
            return response.text
    except Exception as e:
        print("AI error:", e)
    return None

# Initialize
prev_pos = None
canvas = None
output_text = ""

# Main loop
while run:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)

    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)

        if fingers == [1, 1, 1, 1, 0]:
            output_text = sendToAI(canvas)
            if output_text:
                lower = output_text.lower()
                if any(x in lower for x in ["not", "unable", "sorry", "can't", "i cannot", "no math"]):
                    output_text_area.subheader("Tere bus ki baat nahi hai ja dudh bhech 😜")
                else:
                    output_text_area.subheader(output_text)
            else:
                output_text_area.subheader("Tere bus ki baat nahi hai ja dudh bhech 😜")

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")
    cv2.waitKey(1)
