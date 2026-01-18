import cv2
import time 
import numpy as np 
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    start_time = time.time()
    
    results = model(
        frame,
        #conf=0.7,
        classes = [0]
        )
    latency = (time.time() - start_time) * 1000 
    
    boxes_obj = results[0].boxes
    if boxes_obj is not None and len(boxes_obj) > 0:
        bboxes = boxes_obj.xyxy.cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        classes = boxes_obj.cls.cpu().numpy()
        
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(classes[i])] if hasattr(model, 'names') else str(int(classes[i]))
            label = f'{class_name}{confs[i]:.2f}'
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    masks_obj = results[0].masks
    if masks_obj is not None and len(masks_obj) > 0:
        masks = masks_obj.data.cpu().numpy() if hasattr(masks_obj.data, 'cpu') else masks_obj.data
        for mask in masks:
            mask_bin = (mask > 0.5).astype(np.uint8) * 255
            mask_bin = cv2.resize(mask_bin, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            binary_mask = cv2.threshold(mask_bin, 127, 255, cv2.THRESH_BINARY)[1]
            binary_mask_3c = cv2.merge([binary_mask, binary_mask, binary_mask])
            
            random_color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
            colored_mask = np.full((frame.shape[0], frame.shape[1], 3), random_color, dtype=np.uint8)
            
            output_frame = frame.copy()
            output_frame[binary_mask_3c == 255] = colored_mask[binary_mask_3c == 255]
            
            frame = output_frame
            
        cv2.putText(frame, f'Masks: {len(masks)}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.putText(frame, f'Latency: {latency:.1f}ms', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imshow("YOLO11-Seg - segmentacion en Tiempo Real", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

                        
                        
           