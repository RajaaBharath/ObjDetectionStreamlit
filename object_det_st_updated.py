import streamlit as st
import cv2
import numpy as np

# Load the model and labels
frozen_model = 'frozen_inference_graph.pb'
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

class_labels = []
file_name = 'labels.txt'

with open(file_name, 'rt') as f:
    class_labels = f.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Streamlit app title and instructions
st.title("Object Detection App")
st.markdown("Upload an image to perform object detection using the pre-trained model.")

# File upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Cache the image to improve performance
    @st.cache_resource
    def detect_objects(image):
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
        ClassIndex, confidence, bbox = model.detect(img)
        return img, ClassIndex, confidence, bbox


    # Perform object detection
    img, ClassIndex, confidence, bbox = detect_objects(uploaded_image)

    # Display detected objects
    st.image(img, channels="BGR", use_column_width=True, caption="Detected Objects")

    # Display object details
    st.subheader("Object Details")
    for classId, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        label = class_labels[classId - 1]
        st.write(f"Class: {label}, Confidence: {round(float(conf), 2)}")
        st.image(img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]], channels="BGR", caption=label)

# Provide instructions for using the app
st.markdown("""
#### Instructions:
1. Upload an image using the file uploader.
2. The app will perform object detection using a pre-trained model.
3. Detected objects and their confidence scores will be displayed.

Note: This example uses a pre-trained model for object detection.
""")
