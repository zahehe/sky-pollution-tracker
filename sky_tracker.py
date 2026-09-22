import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Set page styling
st.set_page_config(page_title="Sky Pollution Tracker", layout="centered")

st.title("🌤️ Sky Pollution Tracker")
st.write(
    "Upload a picture of the sky to analyze haze and estimated pollution levels."
)

# 1. File Uploader Component
uploaded_file = st.file_uploader(
    "Choose a sky image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Sky Image", use_container_width=True)

    # 2. Convert PIL Image to OpenCV format (BGR)
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 3. Smart DIP Logic for Smoke, Dust & Smog Detection

    # A) Gray/Smoke Ratio (Pollution & dark/white smoke has very similar R, G, B channel values)
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    rg_diff = np.abs(r - g)
    gb_diff = np.abs(g - b)
    rb_diff = np.abs(r - b)
    # Low difference between R, G, B channels = Gray/Murky/Polluted pixels
    gray_mask = (rg_diff < 25) & (gb_diff < 25) & (rb_diff < 25)
    gray_percentage = (np.sum(gray_mask) / gray_mask.size) * 100

    # B) Dark Channel Prior for Atmosphere Haze
    min_channel = np.min(img_bgr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)
    haze_intensity = dark_channel.mean()

    # C) Calculate Final Pollution Score (0 to 100)
    # Heavy weight on gray smoke/haze presence
    pollution_score = int(
        np.clip((gray_percentage * 0.75) + (haze_intensity * 0.35), 0, 100)
    )

    # 4. Display Results
    st.subheader("📊 Analysis Results")
    st.write(f"**Estimated Pollution Index:** {pollution_score} / 100")
    st.write(
        f"*(Debug Stats -> Gray/Smoke Pixels: {int(gray_percentage)}% | Haze Intensity: {int(haze_intensity)})*"
    )

    if pollution_score < 35:
        st.success("🟢 **Clean Sky:** Clear air quality detected!")
    elif pollution_score < 60:
        st.warning("🟡 **Moderate Haze:** Mild smog/dust present.")
    else:
        st.error("🔴 **Heavy Smog/Pollution:** High levels of pollution/smoke!")
