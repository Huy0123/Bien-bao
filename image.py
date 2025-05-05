import cv2

def process_detections(results, names):
    detected_labels = []
    
    if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
        res = results[0]
        for box in res.boxes:
            label_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            text = f"{names[label_id]} {confidence:.2f}"
            
            detected_labels.append(text)
        frame = results[0].plot()
    else:
        frame = None
        
    return frame, detected_labels

class ImageDetector:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def detect(self, image_path, conf_thresh, iou_thresh, img_size):
        results = self.model.predict(
            image_path,
            device=self.device,
            save=False,
            imgsz=img_size,
            conf=conf_thresh,
            iou=iou_thresh
        )
        
        frame, labels = process_detections(results, self.model.names)
        
        # Nếu không có kết quả nhận diện, trả về ảnh gốc
        if frame is None:
            frame = cv2.imread(image_path)
            
        return frame, labels

    def save(self, frame, save_path):
        cv2.imwrite(save_path, frame)
