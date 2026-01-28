from ultralytics import YOLO

# 1. Load YOUR custom model
model = YOLO(r'C:\Users\Theotime\Documents\GitHub\Cat-Camera\runs\detect\cat_detector_v15\weights\best.pt')

# 2. Export it. 
# Use format='openvino' (Intel) or 'onnx' (AMD/Generic)
# half=True converts it to FP16, which is faster and uses less RAM with negligible accuracy loss.
model.export(format='openvino', half=True)