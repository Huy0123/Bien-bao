import cv2
import threading
import time

class VideoDetector:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.output_path = None

    def process(self, input_path, output_path, conf_thresh, iou_thresh, img_size,
                progress_callback=None, frame_callback=None, finish_callback=None):
        
        # Kiểm tra đầu vào
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Tạo đối tượng ghi video
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        self.output_path = output_path

        def _run():
            start_time = time.time()
            cap = cv2.VideoCapture(input_path)
            frame_count = 0
            
            while True:
                # Đọc khung hình
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Xử lý khung hình - chỉ dùng predict
                try:
                    results = self.model.predict(
                        frame,
                        device=self.device,
                        classes=None,
                        verbose=False,
                        imgsz=img_size,
                        conf=conf_thresh,
                        iou=iou_thresh
                    )
                        
                    # Trích xuất nhãn cho callback
                    detected_labels = []
                    if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
                        for box in results[0].boxes:
                            label_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            label = self.model.names[label_id]
                            text = f"{label} {conf:.2f}"
                            detected_labels.append(text)
                    
                    # Tạo khung hình với các chú thích
                    if results and len(results) > 0:
                        annotated_frame = results[0].plot()
                    else:
                        annotated_frame = frame.copy()
                        
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    annotated_frame = frame.copy()
                    detected_labels = []
                
                # Ghi khung hình vào video đầu ra
                writer.write(annotated_frame)
                
                # # Cập nhật tiến độ
                if progress_callback:
                    progress_callback(frame_count, total_frames)
                
                # # Cập nhật hiển thị
                if frame_callback:
                    frame_callback(annotated_frame, detected_labels)
                
                frame_count += 1
            
            # Dọn dẹp
            cap.release()
            writer.release()
            
            # Tính toán thống kê cuối cùng
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"Processed {frame_count} frames in {total_time:.2f}s (avg {avg_fps:.2f} FPS)")
            
            if finish_callback:
                finish_callback(output_path)

        # Khởi chạy xử lý trong một luồng riêng biệt
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread
