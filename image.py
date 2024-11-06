import streamlit as st
import cv2
import numpy as np
import tempfile
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
            background-image: url(data:image/{"download.jpeg"};base64,{encoded_string.decode()});
            background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
add_bg_from_local('download.jpeg')

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
helmet_cascade = cv2.CascadeClassifier('haarcascade_helmet.xml')  # Path to your helmet Haar Cascade

# Function to detect helmets and number plates in an image
def detect_helmet_and_number_plate(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect helmets
    helmets = helmet_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
    
    # For number plate detection, you can add a custom function or a trained model
    # Here we will use a simple Haar Cascade for demonstration
    number_plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')  # Adjust this path
    plates = number_plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    # Draw bounding boxes for helmets
    for (x, y, w, h) in helmets:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Helmet", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw bounding boxes for number plates
    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, "Number Plate", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# Streamlit app layout
st.title("Helmet Violation and Number Plate Detection")

# Option for input type: image, video, or webcam
input_type = st.selectbox("Choose input type:", ("Upload Image", "Upload Video", "Use Webcam"))

# Upload image
if input_type == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        
        processed_image = detect_helmet_and_number_plate(image)
        st.image(processed_image, channels="BGR")

# Upload video
elif input_type == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()  # Placeholder for video frame

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Process each frame
            processed_frame = detect_helmet_and_number_plate(frame)

            # Display processed frame
            stframe.image(processed_frame, channels="BGR")

        cap.release()

# Live video stream from webcam
elif input_type == "Use Webcam":
    st.write("Starting webcam...")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Placeholder for video frame

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.write("Error: Unable to read from webcam")
            break

        # Process each frame
        processed_frame = detect_helmet_and_number_plate(frame)

        # Display processed frame
        stframe.image(processed_frame, channels="BGR")

    cap.release()