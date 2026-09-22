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

    # Focus analysis on the top 70% (sky region)
    height = img_bgr.shape[0]
    sky_bgr = img_bgr[0 : int(height * 0.7), :]
    gray_sky = cv2.cvtColor(sky_bgr, cv2.COLOR_BGR2GRAY)

    # Convert to HSV
    hsv = cv2.cvtColor(sky_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # --- DIP SMOKE & PLUME DETECTION ---

    # 1. Smoke Plume / Factory Chimney Detection (Detects heavy dark vertical structures/smoke)
    # Canny Edge Detector finds sharp smoke borders
    edges = cv2.Canny(gray_sky, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Dark smoke pixels (low brightness pixels in upper sky area)
    dark_smoke_mask = v < 60
    dark_smoke_ratio = np.sum(dark_smoke_mask) / v.size

    # 2. Sunset / Sunrise Check
    sunset_mask = ((h < 25) | (h > 160)) & (s > 80) & (v > 100)
    sunset_ratio = np.sum(sunset_mask) / h.size

    # 3. Dust / Smog Check (Murky yellowish haze)
    dust_mask = (h >= 15) & (h <= 35) & (s < 120) & (v > 80)
    dust_ratio = np.sum(dust_mask) / h.size

    # 4. Rain Cloud Check (Overcast rain clouds)
    cloud_mask = (s < 50) & (v < 110)
    cloud_ratio = np.sum(cloud_mask) / h.size

    # 5. Haze Intensity (Dark Channel Prior)
    min_channel = np.min(sky_bgr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)
    haze_intensity = dark_channel.mean()

    # --- RESULTS & CLASSIFICATION ---

    st.subheader("📊 Analysis Results")

    # Priority 1: Heavy Industrial Smoke / Factory Plumes (OVERWRITES SUNSET)
    if (
        dark_smoke_ratio > 0.08 and edge_density > 0.02
    ) or dust_ratio > 0.18:
        pollution_score = 92
        st.write(f"**Estimated Pollution Index:** {pollution_score} / 100")
        st.error(
            "🔴 **Heavy Industrial Smoke / Pollution:** Thick smoke plumes or toxic smog detected!"
        )

    # Priority 2: Sunset / Sunrise (Only if NO smoke plumes are present)
    elif sunset_ratio > 0.15:
        pollution_score = 15
        st.write(f"**Estimated Pollution Index:** {pollution_score} / 100")
        st.success(
            "🌅 **Clear Sky (Sunset/Sunrise):** Clear air with vibrant sunset colors."
        )

    # Priority 3: Rain Clouds
    elif cloud_ratio > 0.35:
        pollution_score = 25
        st.write(f"**Estimated Pollution Index:** {pollution_score} / 100")
        st.info(
            "🌧️ **Cloudy / Rain Sky:** Dark rain clouds present. Clean air, but likely to rain!"
        )

    # Priority 4: Standard Haze / Clean
    else:
        pollution_score = int(np.clip(haze_intensity * 0.4, 10, 60))
        st.write(f"**Estimated Pollution Index:** {pollution_score} / 100")
        if pollution_score < 35:
            st.success("🟢 **Clean Sky:** Good air quality detected.")
        else:
            st.warning("🟡 **Moderate Haze:** Slight atmospheric haze.")
