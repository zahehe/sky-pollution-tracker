import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Set page styling
st.set_page_config(page_title="Sky Pollution Tracker", layout="centered")

st.title("🌤️ Sky Pollution Tracker")
st.write(
    "Upload a picture of the sky to analyze weather, sunsets, and air quality."
)

# 1. File Uploader Component
uploaded_file = st.file_uploader(
    "Choose a sky image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert PIL Image to OpenCV format (BGR)
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Focus analysis primarily on the top 60% of the image (the sky region)
    height = img_bgr.shape[0]
    sky_bgr = img_bgr[0 : int(height * 0.6), :]

    # Convert to HSV (Fixed constant name: COLOR_BGR2HSV)
    hsv = cv2.cvtColor(sky_bgr, cv2.COLOR_BGR2HSV)

    # Extract HSV channels (Hue, Saturation, Value/Brightness)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # --- DIP CLASSIFICATION LOGIC ---

    # A. Sunset / Sunrise Check (Orange/Pink hues: Hue 0-25 or 160-180, high saturation & brightness)
    sunset_mask = ((h < 25) | (h > 160)) & (s > 70) & (v > 90)
    sunset_ratio = np.sum(sunset_mask) / h.size

    # B. Dust / Smog / Industrial Pollution Check (Yellowish/Brownish murky haze)
    dust_mask = (h >= 15) & (h <= 35) & (s < 120) & (v > 80)
    dust_ratio = np.sum(dust_mask) / h.size

    # C. Rain Cloud / Overcast Check (Dark gray/black tones: Low brightness, low saturation)
    cloud_mask = (s < 55) & (v < 110)
    cloud_ratio = np.sum(cloud_mask) / h.size

    # D. Atmospheric Haze Intensity (Dark Channel Prior)
    min_channel = np.min(sky_bgr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)
    haze_intensity = dark_channel.mean()

    # --- RESULTS DISPLAY ---

    st.subheader("📊 Analysis Results")

    # 1. Sunset / Sunrise
    if sunset_ratio > 0.12:
        pollution_score = 15
        st.write(f"**Estimated Pollution Index:** {pollution_score} / 100")
        st.success(
            "🌅 **Clear Sky (Sunset/Sunrise):** Vivid sunset colors detected! Air is clear."
        )

    # 2. Heavy Pollution / Smoke / Dust
    elif dust_ratio > 0.15 or haze_intensity > 135:
        pollution_score = int(np.clip(70 + (dust_ratio * 100), 70, 98))
        st.write(f"**Estimated Pollution Index:** {pollution_score} / 100")
        st.error(
            "🔴 **Heavy Pollution / Dust Haze:** Murky dust or industrial smog detected!"
        )

    # 3. Rain / Storm Clouds
    elif cloud_ratio > 0.30:
        pollution_score = 25
        st.write(f"**Estimated Pollution Index:** {pollution_score} / 100")
        st.info(
            "🌧️ **Cloudy Sky:** Dark rain clouds detected. Clean air, but likely to rain!"
        )

    # 4. Normal / Clear Sky
    else:
        pollution_score = int(np.clip(haze_intensity * 0.4, 10, 55))
        st.write(f"**Estimated Pollution Index:** {pollution_score} / 100")
        if pollution_score < 35:
            st.success("🟢 **Clean Sky:** Clear blue sky with healthy air quality.")
        else:
            st.warning("🟡 **Moderate Haze:** Mild atmospheric haze present.")
