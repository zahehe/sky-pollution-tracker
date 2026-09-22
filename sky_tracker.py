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

    # 3. DIP Logic: Haze / Brightness Analysis
    # Convert image to Gray to measure variance/contrast (Haze reduces contrast)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()  # Standard deviation of pixel intensities

    # Convert image to HSV to analyze Saturation (Clean sky = high blue saturation)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()

    # Calculate an estimated Pollution Index (0 to 100)
    # Low contrast + low saturation = Hazy/Polluted sky
    pollution_score = max(0, min(100, int(100 - (contrast + saturation / 2))))

    # 4. Display Results
    st.subheader("📊 Analysis Results")
    st.write(f"**Estimated Pollution Index:** {pollution_score} / 100")

    if pollution_score < 35:
        st.success("🟢 **Clean Sky:** Clear air quality detected!")
    elif pollution_score < 65:
        st.warning("🟡 **Moderate Haze:** Mild atmospheric pollution/smog.")
    else:
        st.error("🔴 **Heavy Smog/Pollution:** High level of atmospheric haze!")