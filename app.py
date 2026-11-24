import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="RoadVision AI",
    page_icon="🚧",
    layout="wide"
)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🚧 RoadVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Road Damage Detection System</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="SreekarAditya/yolo-rdd2022-benchmark",
        filename="yolo-rdd2022-benchmark/india_ft_10pct_seed0_best.pt"
    )

    return YOLO(model_path)


model = load_model()

# ---------------- UPLOAD ----------------
st.subheader("📷 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 Original Image")
        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown("### 🔍 Detection")

        if st.button(
            "🚧 Detect Road Damage",
            use_container_width=True
        ):

            # AI detection
            results = model.predict(
                source=image,
                conf=0.10,
                imgsz=800
            )

            result_image = results[0].plot()

            st.image(
                result_image,
                caption="AI Detection Result",
                use_container_width=True
            )

            # ---------------- RESULTS ----------------

            damage_count = len(results[0].boxes)

            st.divider()

            st.subheader("📊 Detection Results")

            # No damage
            if damage_count == 0:

                road_condition = "⚪ Inconclusive"

                st.warning(
                    "No road damage was confidently detected."
                )

                damage_types = {}

            else:

                damage_types = {}

                for box in results[0].boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    class_name = model.names[class_id]

                    if class_name in damage_types:
                        damage_types[class_name] += 1
                    else:
                        damage_types[class_name] = 1

                # ---------------- CONDITION ----------------

                if damage_count >= 6:
                    road_condition = "🔴 Poor"
                elif damage_count >= 3:
                    road_condition = "🟡 Moderate"
                else:
                    road_condition = "🟠 Damaged"

            # ---------------- METRICS ----------------

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "Total Damage",
                    damage_count
                )

            with m2:

                potholes = damage_types.get(
                    "Pothole",
                    0
                )

                st.metric(
                    "Potholes",
                    potholes
                )

            with m3:

                alligator = damage_types.get(
                    "Alligator",
                    0
                )

                st.metric(
                    "Alligator Damage",
                    alligator
                )

            # ---------------- CONDITION ----------------

            st.markdown("### 🛣️ Overall Road Condition")

            st.info(
                f"Road Condition: **{road_condition}**"
            )

            # ---------------- DAMAGE SUMMARY ----------------

            if damage_count > 0:

                st.markdown("### 📋 Damage Summary")

                for damage, count in damage_types.items():

                    st.write(
                        f"🔧 **{damage}** : {count}"
                    )

            # ---------------- REPORT ----------------

            report = "ROAD DAMAGE DETECTION REPORT\n\n"

            report += f"Total Damage Detected: {damage_count}\n\n"

            for damage, count in damage_types.items():

                report += f"{damage}: {count}\n"

            report += (
                f"\nOverall Road Condition: "
                f"{road_condition}\n"
            )

            st.download_button(
                "📥 Download Detection Report",
                data=report,
                file_name="road_damage_report.txt",
                mime="text/plain",
                use_container_width=True
            )

else:

    st.info(
        "👆 Upload a road image above to begin detection."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "RoadVision AI | YOLO-based Road Damage Detection"
)