import cv2
import threading
import time

class CameraDetector:
    def __init__(self, model, device, interval=30):
        self.model = model
        self.device = device
        self.interval = interval
        self.running = False
        self.recording = False
        self.writer = None
        

    def start(self,conf_thresh, iou_thresh, img_size, frame_callback, label_callback ,smooth_mode=True):
        if self.running:
            return
        self.running = True
        
        def _run():
            cap = cv2.VideoCapture(0)
            count = 0
            
            # FPS control
            fps_target = 20
            frame_time = 1.0 / fps_target
            last_time = time.time()
            
            results = None
            
            while self.running:
                # Time control
                current_time = time.time()
                delta = current_time - last_time
                if delta < frame_time:
                    time.sleep(frame_time - delta)
                last_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                should_process = smooth_mode or count == 0 or count % self.interval == 0
                
                # Process frame if needed
                if should_process:
                    try:
                        results = self.model.predict(
                            frame,
                            device=self.device,
                            verbose=False,
                            imgsz=img_size,
                            conf=conf_thresh,
                            iou=iou_thresh
                        )
                    except Exception as e:
                        print(f"Camera processing error: {e}")
                        results = None
                
                detected_labels = []
                if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
                    for box in results[0].boxes:
                        label_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        label = self.model.names[label_id]
                        text = f"{label} {conf:.2f}"
                        detected_labels.append(text)
                
                # Get annotated frame
                if results and len(results) > 0:
                    processed_frame = results[0].plot()
                else:
                    processed_frame = frame.copy()
                
                # Recording logic
                if self.recording and self.writer:
                    self.writer.write(processed_frame)
                
                # Add recording indicator
                if self.recording:
                    cv2.circle(processed_frame, (20, 20), 10, (0, 0, 255), -1)
                    cv2.putText(processed_frame, "REC", (35, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Update UI
                if frame_callback:
                    frame_callback(processed_frame)
                if label_callback:
                    label_callback(detected_labels)
                
                count += 1
            
            # Cleanup
            cap.release()
            if self.writer:
                self.writer.release()
        
        # Start in a separate thread
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread

    def stop(self):
        self.running = False

    def start_record(self, save_path, fps=20, size=(640,480)):
        self.writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            size
        )
        self.recording = True

    def stop_record(self):
        self.recording = False
        if self.writer:
            self.writer.release()
            self.writer = None