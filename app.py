"""
app.py
โปรแกรมจำแนกโรค Covid-19 จากภาพ X-Ray
------------------------------------------------------------------
โมเดลที่ใช้ (*.pkcls) ถูกฝึกด้วยโปรแกรม Orange Data Mining โดยใช้ pipeline:
  ภาพ X-Ray --> Image Embedding (แปลงภาพเป็นเวกเตอร์ตัวเลข 2048 มิติ ชื่อ n0..n2047)
            --> โมเดลจำแนก (Tree / SVM)  --> ผลลัพธ์: covid, normal, pneumonia

ดังนั้นแอปนี้จึงรับ "ภาพ X-Ray" เป็น input โดยตรง (ไม่ใช่ตัวเลขที่กรอกเอง)
แล้วเรียกใช้ตัวสกัด embedding ตัวเดียวกับที่ใช้ตอนฝึกโมเดลใน Orange
(ค่าเริ่มต้นคือ "Inception v3" เพราะให้เวกเตอร์ 2048 มิติตรงกับโมเดล)

*** สำคัญ: ต้องเลือก embedder ให้ตรงกับที่ใช้ตอนฝึกโมเดลจริงใน Orange
    (ดูได้จาก widget "Image Embedding" ในไฟล์ .ows ที่ใช้สร้างโมเดล) ***
"""

import os
# ต้องตั้งค่านี้ก่อน import Orange เพื่อให้รันบนเซิร์ฟเวอร์ที่ไม่มีหน้าจอ (headless) ได้
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import streamlit as st
import joblib
import numpy as np
from PIL import Image
from Orange.data import Domain, ContinuousVariable, Table

# ============================================================
# 1) ตั้งค่าหน้าเว็บและหัวข้อแอป
# ============================================================
st.set_page_config(page_title="โปรแกรมจำแนกโรค Covid-19 จากภาพ X-Ray", layout="centered")

# แสดงหัวข้อแอปด้วย st.title
st.title("โปรแกรมจำแนกโรค Covid-19 จากภาพ X-Ray")
st.write("อัปโหลดภาพ X-Ray ปอด แล้วให้โมเดล AI ช่วยจำแนกกลุ่มโรคเบื้องต้น")

# mapping ชื่อคลาสภาษาอังกฤษ -> ข้อความภาษาไทยที่อ่านง่าย
CLASS_TH = {
    "covid": "ผลตรวจเข้าข่าย Covid-19",
    "normal": "ปกติ ไม่พบความผิดปกติ",
    "pneumonia": "เข้าข่ายปอดอักเสบ (Pneumonia)",
}

# ============================================================
# 2) ส่วนเลือกและโหลดไฟล์โมเดล (*.pkcls) ด้วย joblib
# ============================================================
st.sidebar.header("⚙️ เลือกโมเดล")

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# สแกนหาไฟล์ .pkcls ทั้งหมดที่ commit มากับ repo ในโฟลเดอร์ models/
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkcls")]

model = None

if model_files:
    selected_model_file = st.sidebar.selectbox(
        "เลือกไฟล์โมเดลที่ต้องการใช้ทำนาย",
        sorted(model_files)
    )
    model_path = os.path.join(MODEL_DIR, selected_model_file)

    # โหลดโมเดลด้วย joblib (cache ไว้ไม่ให้โหลดซ้ำทุกครั้งที่มีการ interact)
    @st.cache_resource
    def load_model(path):
        return joblib.load(path)

    try:
        model = load_model(model_path)
        n_features = len(model.domain.attributes)
        class_values = model.domain.class_var.values
        st.sidebar.success(
            f"โหลดโมเดล '{selected_model_file}' สำเร็จ\n\n"
            f"- จำนวน features: {n_features}\n"
            f"- คลาสที่ทำนายได้: {', '.join(class_values)}"
        )
    except Exception as e:
        st.sidebar.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
else:
    st.sidebar.warning(
        "ยังไม่พบไฟล์โมเดล .pkcls กรุณานำไฟล์โมเดลไปวางไว้ในโฟลเดอร์ 'models/' "
        "ของโปรเจกต์ (commit ขึ้น GitHub พร้อมกับ app.py)"
    )

# ============================================================
# 2.1) กำหนดตัวสกัด embedding แบบตายตัว (ต้องตรงกับตอนฝึกโมเดลใน Orange)
#      โมเดลนี้มี feature 2048 ตัว (n0..n2047) ซึ่งตรงกับ "Inception v3"
#      จึงล็อกค่าไว้ในโค้ดเลย ไม่ต้องให้ผู้ใช้เลือกเอง
# ============================================================
embedder_name = "inception-v3"

st.divider()

# ============================================================
# 3) ส่วนอัปโหลดภาพ X-Ray
# ============================================================
st.header("อัปโหลดภาพ X-Ray")
uploaded_image = st.file_uploader(
    "เลือกไฟล์ภาพ X-Ray (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="ภาพ X-Ray ที่อัปโหลด", use_column_width=True)

# ============================================================
# 4) ปุ่ม "ทำนายผล"
# ============================================================
if st.button("ทำนายผล"):
    if model is None:
        st.error("ไม่พบโมเดลที่โหลดไว้ กรุณาตรวจสอบว่ามีไฟล์ .pkcls อยู่ในโฟลเดอร์ 'models/'")
    elif uploaded_image is None:
        st.error("กรุณาอัปโหลดภาพ X-Ray ก่อนทำนายผล")
    else:
        with st.spinner("กำลังสกัดคุณลักษณะจากภาพและทำนายผล..."):
            try:
                # --------------------------------------------------
                # 4.1 บันทึกภาพลงไฟล์ชั่วคราว (ImageEmbedder ต้องการ path ของไฟล์)
                # --------------------------------------------------
                temp_img_path = os.path.join("temp_xray.jpg")
                image.save(temp_img_path)

                # --------------------------------------------------
                # 4.2 สกัด embedding ด้วยโมเดล deep learning เดียวกับตอนฝึก
                #     (ต้องมีอินเทอร์เน็ตเชื่อมต่อไปยังเซิร์ฟเวอร์ของ Orange
                #      ยกเว้น embedder 'squeezenet' ที่ทำงานในเครื่องได้)
                # --------------------------------------------------
                from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

                embedder = ImageEmbedder(model=embedder_name)
                embeddings = embedder([temp_img_path])
                embedder.clear_cache is not None  # (ไม่บังคับ) เคลียร์ cache ได้ถ้าต้องการ

                if embeddings is None or embeddings[0] is None:
                    st.error(
                        "ไม่สามารถสกัด embedding จากภาพได้ "
                        "กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต หรือรูปแบบไฟล์ภาพ"
                    )
                else:
                    embedding_vector = np.array(embeddings[0], dtype=float).reshape(1, -1)

                    # --------------------------------------------------
                    # 4.3 ตรวจสอบจำนวนมิติของ embedding ให้ตรงกับโมเดล
                    # --------------------------------------------------
                    n_features = len(model.domain.attributes)
                    if embedding_vector.shape[1] != n_features:
                        st.error(
                            f"จำนวนมิติของ embedding ({embedding_vector.shape[1]}) "
                            f"ไม่ตรงกับที่โมเดลต้องการ ({n_features}) "
                            f"กรุณาเปลี่ยน Embedder ให้ตรงกับตอนฝึกโมเดล"
                        )
                    else:
                        # --------------------------------------------------
                        # 4.4 สร้างตาราง Orange จาก embedding ดิบ
                        #     (ชื่อ feature ต้องตรงกับตอนฝึก คือ n0, n1, ..., n2047)
                        #     Orange จะแปลง/นอร์มัลไลซ์ค่าตาม compute_value
                        #     ที่ฝังอยู่ในโดเมนของโมเดลให้อัตโนมัติ
                        # --------------------------------------------------
                        feature_names = [a.name for a in model.domain.attributes]
                        raw_domain = Domain(
                            [ContinuousVariable(name) for name in feature_names]
                        )
                        raw_table = Table.from_numpy(raw_domain, embedding_vector)

                        # --------------------------------------------------
                        # 4.5 ส่งเข้าโมเดลเพื่อทำนายผล (ทั้ง class และความน่าจะเป็น)
                        # --------------------------------------------------
                        pred_index = model(raw_table)
                        pred_proba = model(raw_table, model.Probs)

                        class_values = model.domain.class_var.values
                        predicted_label = class_values[int(pred_index[0])]
                        confidence = float(np.max(pred_proba[0])) * 100

                        # --------------------------------------------------
                        # 5) แสดงผลการทำนายให้อ่านง่าย
                        # --------------------------------------------------
                        thai_label = CLASS_TH.get(predicted_label, predicted_label)

                        if predicted_label == "covid":
                            st.error(f"⚠️ {thai_label} (ความมั่นใจ {confidence:.2f}%)")
                        elif predicted_label == "pneumonia":
                            st.warning(f"🔶 {thai_label} (ความมั่นใจ {confidence:.2f}%)")
                        else:
                            st.success(f"✅ {thai_label} (ความมั่นใจ {confidence:.2f}%)")

                        # แสดงความน่าจะเป็นของทุกคลาสแบบละเอียด
                        st.write("### รายละเอียดความน่าจะเป็นของแต่ละคลาส")
                        for cls, prob in zip(class_values, pred_proba[0]):
                            st.write(f"- {CLASS_TH.get(cls, cls)}: {prob * 100:.2f}%")

                # ลบไฟล์ภาพชั่วคราวทิ้งหลังใช้งาน
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

            except ImportError:
                st.error(
                    "ไม่พบไลบรารี orange3-imageanalytics กรุณาติดตั้งด้วยคำสั่ง:\n"
                    "pip install orange3-imageanalytics"
                )
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดระหว่างประมวลผล: {e}")

# ============================================================
# หมายเหตุท้ายไฟล์: จุดที่ต้องตรวจสอบ/แก้ไขให้ตรงกับโมเดลจริง
# ============================================================
# 1. Embedder: ต้องเลือกให้ตรงกับตอนฝึกโมเดลใน Orange (widget "Image Embedding")
#    - ถ้าไม่แน่ใจ ให้เปิดไฟล์ .ows ที่ใช้สร้างโมเดล แล้วดูว่าเลือก embedder ตัวไหน
#    - โมเดลนี้มี 2048 features (n0..n2047) ซึ่งตรงกับ "Inception v3" มากที่สุด
# 2. อินเทอร์เน็ต: embedder ส่วนใหญ่ (ยกเว้น squeezenet) ต้องเชื่อมต่อไปยัง
#    เซิร์ฟเวอร์ของ Orange (api.garaza.io) จึงต้องมั่นใจว่าเซิร์ฟเวอร์ที่รันแอป
#    สามารถเข้าถึงอินเทอร์เน็ตได้
# 3. ชื่อคลาส (category values): ปัจจุบันคือ covid / normal / pneumonia
#    ถ้าโมเดลของคุณมีชื่อคลาสอื่น ให้แก้ dict CLASS_TH ให้ตรงกัน
# 4. หากต้องการ deploy บนเซิร์ฟเวอร์ที่ไม่มีจอภาพ (headless) ต้องคง
#    บรรทัด os.environ.setdefault("QT_QPA_PLATFORM", "offscreen") ไว้ที่บนสุดของไฟล์
#    เพราะ Orange3 core import โมดูล Qt เพื่ออ่าน config path แม้ไม่ได้ใช้ GUI จริง
