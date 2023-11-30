import cv2
import numpy as np
import streamlit as st
# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in [net.getUnconnectedOutLayers()]]
# Load image
uploaded_files = st.file_uploader("Choose a image file", accept_multiple_files=True)

if st.button('run!!!!!!!!'):
    for uploaded_file in uploaded_files:
        st.write("Filename: ", uploaded_file.name)
        st.write("Filename: ", uploaded_file)
        with st.spinner('Wait for it...'):
            i=str(uploaded_file.name)
            image_list=[i]
            for i in image_list:
                    
                image = cv2.imread(i)
                height, width, channels = image.shape

                # Detect objects
                blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                # Process detections
                class_ids = []
                confidences = []
                boxes = []

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                # Apply non-max suppression
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                # Draw bounding boxes and labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = classes[class_ids[i]]
                        color = (0, 0, 0)  # Green
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), font, 0.5, color, 2)

                # Save result to file
                cv2.imwrite("result.jpg", image)
                st.image(image, caption='detected image')
                st.success('Done!')
