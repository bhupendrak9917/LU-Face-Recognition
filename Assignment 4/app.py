import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import face_recognition


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

model_face_mesh = mp_face_mesh.FaceMesh()

mp_face_detection = mp.solutions.face_detection
model_detection = mp_face_detection.FaceDetection()

mp_selfie_segmentation = mp.solutions.selfie_segmentation
model = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

st.title("OpenCV Operations")
st.subheader("Image operations")


st.write("This application performs various operations with OpenCV")


add_selectbox = st.sidebar.selectbox(
    "What operations you would like to perform?",
    ("About", "Grayscale", "Red", "Green", "Blue", "Meshing", "Face Recognition", "Face Detection", "Selfie Segmentation")
)


if add_selectbox == "About":
    st.write("This application is a demo for streamlit.")

elif add_selectbox == "Grayscale":
    image_file_path = st.sidebar.file_uploader("Upload image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st.image(gray_image)

elif add_selectbox == "Blue":
    image_file_path = st.sidebar.file_uploader("Upload image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        blue_image = cv2.merge([zeros, zeros, b])
        st.image(blue_image)

elif add_selectbox == "Green":
    image_file_path = st.sidebar.file_uploader("Upload image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        blue_image = cv2.merge([zeros, g, zeros])
        st.image(blue_image)

elif add_selectbox == "Red":
    image_file_path = st.sidebar.file_uploader("Upload image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        blue_image = cv2.merge([r, zeros, zeros])
        st.image(blue_image)

elif add_selectbox == "Meshing":
    image_file_path = st.sidebar.file_uploader("Upload image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        results = model_face_mesh.process(image)

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks)
        st.image(image)

elif add_selectbox == "Face Recognition":
    st.header(add_selectbox)
    image_file_path1 = st.sidebar.file_uploader("Upload image")
    image_file_path2 = st.sidebar.file_uploader("Upload another image")
    name = st.sidebar.text_input("name of the person in image1")
    if image_file_path1 is not None and image_file_path2 is not None :
        image1 = np.array(Image.open(image_file_path1))
        st.sidebar.image(image1)
        image2 = np.array(Image.open(image_file_path2))
        st.sidebar.image(image2)
        
        image1=cv2.resize(image1,(300,400))
        image2=cv2.resize(image2,(300,400))

        #Train Image
        image1_face_encoding = face_recognition.face_encodings(image1)[0]
        image1_location = face_recognition.face_locations(image1)[0]
        # image1_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!
        
        #Test Image
        image2_face_encoding = face_recognition.face_encodings(image2)[0]
        
        # Now we can see the two face encodings are of the same person with `compare_faces`!
        results = face_recognition.compare_faces([image1_face_encoding], image2_face_encoding)
        
        #If both the images are same
        if results[0] == 1:
            cv2.rectangle(image1, 
                (image1_location[3], image1_location[0]),
                (image1_location[1], image1_location[2]),
                (0, 255, 0),
                2)
            cv2.rectangle(image1, 
                (image1_location[3], image1_location[2] +30), 
                (image1_location[1], image1_location[2]), 
                (0, 255 , 0),
                 cv2.FILLED)
            cv2.putText(image1,name,
                (image1_location[3] + 20, image1_location[2]+20),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 0 , 255),
                1)
            st.write("Both the Images are SAME")
            st.image(image1)
            st.image(image2)
        else:
            st.write("Both the Images are not SAME")
            
elif add_selectbox == "Face Detection":
    st.header(add_selectbox)
    image_file_path1 = st.sidebar.file_uploader("Upload image")
    
    if image_file_path1 is not None :
        image1 = np.array(Image.open(image_file_path1))
        st.sidebar.image(image1)

        image1=cv2.resize(image1,(300,400))
        
        results1 = model_detection.process(image1)
        
        if results1.detections :
            for landmark in results1.detections:
                print(mp_face_detection.get_key_point(landmark, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                mp_drawing.draw_detection(image1, landmark)
            st.write("Face Detected")
            st.image(image1)
            
            
        else:
            st.write("No Face Detected")

elif add_selectbox == "Selfie Segmentation":
    st.header(add_selectbox)

    background = st.sidebar.radio("Choose background colour",('Blue','Green','Red'))
    image_file_path = st.sidebar.file_uploader("Upload image")
    
    if image_file_path is not None :
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)

        image=cv2.resize(image,(300,400))

        results = model.process(image)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        
        if background == 'Blue':
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (0, 0 ,255)
            bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
            output_image = np.where(condition, image, bg_image)
            st.image(output_image)

        elif background == 'Green':
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (0, 255, 0)
            bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
            output_image = np.where(condition, image, bg_image)
            st.image(output_image)

        elif background == 'Red':
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (255, 0 ,0)
            bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
            output_image = np.where(condition, image, bg_image)
            st.image(output_image)