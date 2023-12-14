#train command
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
#test command
yolo task=detect mode=predict model=best.pt conf=0.25 source=IMG_1149.JPG save=True
