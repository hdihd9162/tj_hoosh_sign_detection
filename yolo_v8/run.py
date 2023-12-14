# Import required libraries
import PIL
import streamlit as st
from ultralytics import YOLO

model_path = 'best.pt'

# Setting page layout
st.set_page_config(
    page_title="Object Detection",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
    
)

# Creating sidebar
st.header("Image Config")     # Adding header to sidebar
# Adding file uploader to sidebar for selecting images
source_img = st.file_uploader(
    "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

# Model Options
confidence = float(st.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.button('Detect Objects'):
    with st.spinner('Wait for it...'):
        res = model.predict(uploaded_image,
                            conf=confidence
                            )
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        st.image(res_plotted,
                    caption='Detected Image',
                    use_column_width=True
                    )
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")
