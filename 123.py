import torch
from ultralytics import YOLO
import cv2
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import threading
import time
from datetime import datetime


# Global variables
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
DETECTION_INTERVAL = 30
SLOW_DELAY = 0.0

# Setup output directories
def setup_output_directories():
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Daily directory
    today_dir = os.path.join(output_dir, time.strftime("%Y-%m-%d"))
    os.makedirs(today_dir, exist_ok=True)

    return today_dir

def load_model(status_label):
    global model
    try:
        status_label.config(text="Đang tải mô hình...")
        model = YOLO('runs/detect/train/weights/best.pt')
        status_label.config(text="Đã tải mô hình thành công")
        return True
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể tải mô hình: {e}")
        status_label.config(text="Không thể tải mô hình")
        return False

def get_unique_filename(path):
    if not os.path.exists(path):
        return path
    base_name, ext = os.path.splitext(path)
    count = 1
    while True:
        new_path = f"{base_name}_{count}{ext}"
        if not os.path.exists(new_path):
            return new_path
        count += 1

def process_detections(frame, results, names):
    """Process detection results and draw bounding boxes on the frame"""
    detected_labels = []
    
    if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
        res = results[0]
        for box in res.boxes:
            # Get bounding box coordinates
            xy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xy)
            
            # Get label and confidence
            label_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            conf_str = f"{confidence:.2f}"
            text = f"{names[label_id]} {conf_str}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Calculate position for text
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x = x1
            text_y = y1 - 10 if y1 > 20 else y1 + 30
            
            # Draw background for text
            cv2.rectangle(frame, (text_x, text_y - text_h - 2),
                          (text_x + text_w, text_y + 2), (128, 128, 128), -1)
            
            # Draw text
            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            detected_labels.append(text)
    
    return frame, detected_labels

def update_result_display(result_label, frame, detected_labels, label_result_text=None):
    """Update the result display with the processed frame"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
    pil_image = Image.fromarray(img_rgb)
    display_img = pil_image.resize((600, 400), resample=Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(display_img)
    
    result_label.config(image=photo)
    result_label.image = photo
    
    # Display detected labels if we have a text label widget
    if label_result_text:
        unique_labels = []
        label_counts = {}
        
        for label in detected_labels:
            class_name = label.split()[0]
            if class_name in label_counts:
                label_counts[class_name] += 1
            else:
                label_counts[class_name] = 1
        
        for name, count in label_counts.items():
            unique_labels.append(f"{name} ({count})")
        
        if unique_labels:
            label_result_text.config(text="Nhận diện: " + ", ".join(unique_labels))
        else:
            label_result_text.config(text="Không nhận diện được đối tượng")

# ====== IMAGE DETECTION FUNCTIONS ======
def upload_image(img_path_label, img_result_label, img_label_text):
    try:
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Tệp hình ảnh", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            img_path_label.config(text=f"Đã tải ảnh: {os.path.basename(file_path)}")
            img_result_label.config(image='')
            img_result_label.image = None
            img_label_text.config(text="")
            
            # Store the path as an attribute of the label
            img_path_label.file_path = file_path
            
            # Display the image preview
            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            display_img = pil_image.resize((600, 400), resample=Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_img)
            img_result_label.config(image=photo)
            img_result_label.image = photo
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể tải ảnh: {e}")

def detect_image(img_path_label, img_result_label, img_label_text, img_status_label, img_progress_bar, conf_threshold, iou_threshold, img_size):
    global model, device
    
    if not hasattr(img_path_label, 'file_path') or not img_path_label.file_path:
        img_status_label.config(text="Vui lòng tải ảnh trước khi nhận diện")
        return
    
    if model is None and not load_model(img_status_label):
        return
    
    try:
        file_path = img_path_label.file_path
        img_progress_bar.pack(fill=X, padx=5, pady=5)
        img_progress_bar["value"] = 10
        
        # Predict
        results = model.predict(
            file_path,
            device=device,
            save=False,
            imgsz=img_size,
            conf=conf_threshold,
            iou=iou_threshold
        )
        
        img_progress_bar["value"] = 60
        
        img = cv2.imread(file_path)
        processed_img, detected_labels = process_detections(img, results, model.names)
        
        img_progress_bar["value"] = 80
        
        # Store processed image for saving
        img_result_label.processed_image = processed_img.copy()
        
        # Update display
        update_result_display(img_result_label, processed_img, detected_labels, img_label_text)
        img_status_label.config(text="Nhận diện thành công")
        
        img_progress_bar["value"] = 100
        img_progress_bar.pack_forget()
    except Exception as e:
        img_progress_bar.pack_forget()
        messagebox.showerror("Lỗi", f"Không thể nhận diện ảnh: {e}")
        img_status_label.config(text=f"Lỗi: {str(e)}")

def save_image_result(img_result_label, img_status_label):
    if not hasattr(img_result_label, 'processed_image'):
        messagebox.showinfo("Thông báo", "Không có ảnh đã xử lý để lưu")
        return
    
    today_dir = setup_output_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"traffic_sign_detected_{timestamp}.jpg"
    
    save_path = filedialog.asksaveasfilename(
        initialdir=today_dir,
        defaultextension=".jpg",
        initialfile=default_name,
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
    )
    
    if save_path:
        try:
            cv2.imwrite(save_path, img_result_label.processed_image)
            messagebox.showinfo("Thành công", f"Đã lưu ảnh tại: {save_path}")
            img_status_label.config(text=f"Đã lưu ảnh tại: {os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {e}")

# ====== VIDEO DETECTION FUNCTIONS ======
def upload_video(vid_path_label, vid_result_label, vid_label_text):
    try:
        file_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Tệp video", "*.mp4 *.avi *.mov *.mkv *.flv")]
        )
        if file_path:
            vid_path_label.config(text=f"Đã tải video: {os.path.basename(file_path)}")
            vid_result_label.config(image='')
            vid_result_label.image = None
            vid_label_text.config(text="")
            
            # Store the path as an attribute of the label
            vid_path_label.file_path = file_path
            
            # Display video first frame as preview
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                display_img = pil_image.resize((600, 400), resample=Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(display_img)
                vid_result_label.config(image=photo)
                vid_result_label.image = photo
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể tải video: {e}")

def process_video(vid_path_label, vid_result_label, vid_label_text, vid_status_label, vid_progress_bar, conf_threshold, iou_threshold, img_size):
    global model, device
    
    if not hasattr(vid_path_label, 'file_path') or not vid_path_label.file_path:
        vid_status_label.config(text="Vui lòng tải video trước khi nhận diện")
        return
    
    if model is None and not load_model(vid_status_label):
        return
    
    file_path = vid_path_label.file_path
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        vid_status_label.config(text="Không thể mở video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Create output path
    today_dir = setup_output_directories()
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(today_dir, f"{base_name}_detected_{timestamp}.mp4")
    output_path = get_unique_filename(output_path)
    
    # Create video writer
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    # Store the output path
    vid_result_label.output_video_path = output_path
    
    def video_thread():
        last_detections = []  # stores bounding boxes and labels
        frame_count = 0
        try:
            cap = cv2.VideoCapture(file_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update progress
                progress = f"Đang xử lý: {frame_count}/{total_frames} frames ({int(frame_count * 100 / total_frames)}%)"
                vid_result_label.after(0, lambda p=progress: vid_status_label.config(text=p))
                vid_progress_bar["value"] = int(frame_count * 100 / total_frames)
                
                # Detect every DETECTION_INTERVAL frames or the first frame
                if frame_count == 0 or frame_count % DETECTION_INTERVAL == 0:
                    results = model.predict(
                        frame,
                        device=device,
                        save=False,
                        imgsz=img_size,
                        conf=conf_threshold,
                        iou=iou_threshold
                    )
                    
                    if results and results[0].boxes is not None:
                        res = results[0]
                        new_detections = []
                        for box in res.boxes:
                            # Get bounding box coordinates
                            xy = box.xyxy[0].tolist()
                            
                            # Get label and confidence
                            label_id = int(box.cls[0].item())
                            confidence = box.conf[0].item()
                            label = model.names[label_id]
                            conf_str = f"{confidence:.2f}"
                            text = f"{label} {conf_str}"
                            
                            new_detections.append((xy, text, (0, 255, 255)))  # Box, text, color
                        last_detections = new_detections
                
                # Draw bounding boxes on current frame
                processed_frame = frame.copy()
                detected_labels = []
                
                for (xy, text, color) in last_detections:
                    x1, y1, x2, y2 = map(int, xy)
                    
                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Calculate position for text
                    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_x = x1
                    text_y = y1 - 10 if y1 > 20 else y1 + 30
                    
                    # Draw background for text
                    cv2.rectangle(processed_frame, (text_x, text_y - text_h - 2),
                                  (text_x + text_w, text_y + 2), (128, 128, 128), -1)
                    
                    # Draw text
                    cv2.putText(processed_frame, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    detected_labels.append(text)
                
                # Write to output video
                video_writer.write(processed_frame)
                
                # Update GUI
                img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img_rgb)
                disp = pil.resize((600, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(disp)
                
                # Display image and detected labels
                unique_labels = list(dict.fromkeys(detected_labels))
                vid_result_label.after(0, lambda p=photo: vid_result_label.config(image=p))
                vid_result_label.after(0, lambda p=photo: setattr(vid_result_label, 'image', p))
                
                if vid_label_text:
                    label_counts = {}
                    for label in detected_labels:
                        class_name = label.split()[0]
                        if class_name in label_counts:
                            label_counts[class_name] += 1
                        else:
                            label_counts[class_name] = 1
                    
                    display_labels = []
                    for name, count in label_counts.items():
                        display_labels.append(f"{name} ({count})")
                    
                    if display_labels:
                        vid_result_label.after(0, lambda labels=", ".join(display_labels): 
                                             vid_label_text.config(text=f"Nhận diện: {labels}"))
                    else:
                        vid_result_label.after(0, lambda: 
                                             vid_label_text.config(text="Không nhận diện được đối tượng"))
                
                time.sleep(SLOW_DELAY)
                frame_count += 1
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xử lý video: {e}")
        finally:
            if cap is not None:
                cap.release()
            if video_writer is not None:
                video_writer.release()
            
            vid_result_label.after(0, lambda: vid_status_label.config(
                text=f"Hoàn thành xử lý video (đã lưu tại: {output_path})"))
            vid_progress_bar["value"] = 100
    
    vid_progress_bar.pack(fill=X, padx=5, pady=5)
    vid_progress_bar["value"] = 0
    threading.Thread(target=video_thread, daemon=True).start()
    vid_status_label.config(text="Đang xử lý video...")

def save_video_result(vid_result_label, vid_status_label):
    if not hasattr(vid_result_label, 'output_video_path'):
        messagebox.showinfo("Thông báo", "Không có video đã xử lý để lưu")
        return
    
    source_video_path = vid_result_label.output_video_path
    if not os.path.exists(source_video_path):
        messagebox.showinfo("Thông báo", "File video đã xử lý không tồn tại")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"traffic_sign_detected_{timestamp}.mp4"
    
    save_path = filedialog.asksaveasfilename(
        defaultextension=".mp4",
        initialfile=default_name,
        filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("All Files", "*.*")]
    )
    
    if save_path:
        try:
            import shutil
            shutil.copy2(source_video_path, save_path)
            messagebox.showinfo("Thành công", f"Đã lưu video tại: {save_path}")
            vid_status_label.config(text=f"Đã lưu video tại: {os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu video: {e}")

# ====== CAMERA DETECTION FUNCTIONS ======
def start_camera(cam_result_label, cam_label_text, cam_status_label, conf_threshold, iou_threshold, img_size):
    global model, device
    
    if model is None and not load_model(cam_status_label):
        return
    
    # Check if camera is already running
    if hasattr(start_camera, 'running') and start_camera.running:
        messagebox.showinfo("Thông báo", "Camera đang chạy")
        return
    
    # Flag to control capture thread
    start_camera.running = True
    start_camera.recording = False
    start_camera.output_video = None
    
    def camera_thread():
        try:
            cap = cv2.VideoCapture(0)  # Use default camera
            if not cap.isOpened():
                cam_status_label.config(text="Không thể kết nối với camera")
                start_camera.running = False
                return
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            frame_count = 0
            last_detections = []
            
            while start_camera.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect every DETECTION_INTERVAL frames or the first frame
                if frame_count == 0 or frame_count % DETECTION_INTERVAL == 0:
                    results = model.predict(
                        frame,
                        device=device,
                        save=False,
                        imgsz=img_size,
                        conf=conf_threshold,
                        iou=iou_threshold
                    )
                    
                    if results and results[0].boxes is not None:
                        res = results[0]
                        new_detections = []
                        for box in res.boxes:
                            # Get bounding box coordinates
                            xy = box.xyxy[0].tolist()
                            
                            # Get label and confidence
                            label_id = int(box.cls[0].item())
                            confidence = box.conf[0].item()
                            label = model.names[label_id]
                            conf_str = f"{confidence:.2f}"
                            text = f"{label} {conf_str}"
                            
                            new_detections.append((xy, text, (0, 255, 255)))  # Box, text, color
                        last_detections = new_detections
                
                # Draw bounding boxes on current frame
                processed_frame = frame.copy()
                detected_labels = []
                
                for (xy, text, color) in last_detections:
                    x1, y1, x2, y2 = map(int, xy)
                    
                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Calculate position for text
                    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_x = x1
                    text_y = y1 - 10 if y1 > 20 else y1 + 30
                    
                    # Draw background for text
                    cv2.rectangle(processed_frame, (text_x, text_y - text_h - 2),
                                  (text_x + text_w, text_y + 2), (128, 128, 128), -1)
                    
                    # Draw text
                    cv2.putText(processed_frame, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    detected_labels.append(text)
                
                # If recording, write to video file
                if start_camera.recording and start_camera.output_video is not None:
                    start_camera.output_video.write(processed_frame)
                
                # Draw recording indicator if recording
                if start_camera.recording:
                    cv2.circle(processed_frame, (20, 20), 10, (0, 0, 255), -1)
                    cv2.putText(processed_frame, "REC", (35, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Update GUI
                img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img_rgb)
                disp = pil.resize((600, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(disp)
                
                cam_result_label.after(0, lambda p=photo: cam_result_label.config(image=p))
                cam_result_label.after(0, lambda p=photo: setattr(cam_result_label, 'image', p))
                
                # Store the latest processed frame
                cam_result_label.current_frame = processed_frame.copy()
                
                if cam_label_text:
                    label_counts = {}
                    for label in detected_labels:
                        class_name = label.split()[0]
                        if class_name in label_counts:
                            label_counts[class_name] += 1
                        else:
                            label_counts[class_name] = 1
                    
                    display_labels = []
                    for name, count in label_counts.items():
                        display_labels.append(f"{name} ({count})")
                    
                    if display_labels:
                        cam_result_label.after(0, lambda labels=", ".join(display_labels): 
                                             cam_label_text.config(text=f"Nhận diện: {labels}"))
                    else:
                        cam_result_label.after(0, lambda: 
                                             cam_label_text.config(text="Không nhận diện được đối tượng"))
                
                frame_count += 1
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi xử lý camera: {e}")
        finally:
            if cap is not None:
                cap.release()
            if start_camera.recording and start_camera.output_video is not None:
                start_camera.output_video.release()
                start_camera.recording = False
                cam_status_label.config(text=f"Đã dừng camera và lưu video")
    
    threading.Thread(target=camera_thread, daemon=True).start()
    cam_status_label.config(text="Camera đang chạy")

def stop_camera(cam_status_label):
    if hasattr(start_camera, 'running'):
        start_camera.running = False
        cam_status_label.config(text="Đã dừng camera")
    else:
        cam_status_label.config(text="Camera chưa được khởi động")

def toggle_recording(cam_result_label, cam_status_label):
    if not hasattr(start_camera, 'running') or not start_camera.running:
        messagebox.showinfo("Thông báo", "Camera chưa được khởi động")
        return
    
    if start_camera.recording:
        # Stop recording
        start_camera.recording = False
        if start_camera.output_video is not None:
            # Close the video writer in the camera thread
            cam_status_label.config(text=f"Đã dừng ghi video, lưu tại: {start_camera.output_path}")
    else:
        # Start recording
        today_dir = setup_output_directories()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(today_dir, f"camera_recording_{timestamp}.mp4")
        output_path = get_unique_filename(output_path)
        
        # Get current frame dimensions
        if hasattr(cam_result_label, 'current_frame'):
            height, width = cam_result_label.current_frame.shape[:2]
        else:
            # Default if no frame available
            width, height = 640, 480
        
        # Create video writer
        start_camera.output_video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            20,  # fps
            (width, height)
        )
        
        start_camera.output_path = output_path
        start_camera.recording = True
        cam_status_label.config(text="Đang ghi video...")

def capture_snapshot(cam_result_label, cam_status_label):
    if not hasattr(cam_result_label, 'current_frame'):
        messagebox.showinfo("Thông báo", "Không có hình ảnh từ camera")
        return
    
    today_dir = setup_output_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(today_dir, f"camera_snapshot_{timestamp}.jpg")
    output_path = get_unique_filename(output_path)
    
    try:
        cv2.imwrite(output_path, cam_result_label.current_frame)
        messagebox.showinfo("Thành công", f"Đã lưu ảnh tại: {output_path}")
        cam_status_label.config(text=f"Đã chụp ảnh: {os.path.basename(output_path)}")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {e}")

def create_gui():
    root = Tk()
    root.title("Nhận diện biển báo giao thông Việt Nam")
    
    # Set window size and position
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    ww, wh = 800, 750
    root.geometry(f"{ww}x{wh}+{(sw - ww) // 2}+{(sh - wh) // 2}")
    
    # Create notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill=BOTH, padx=5, pady=5)
    
    # Create tabs
    img_tab = Frame(notebook, padx=10, pady=10)
    vid_tab = Frame(notebook, padx=10, pady=10)
    cam_tab = Frame(notebook, padx=10, pady=10)
    
    notebook.add(img_tab, text="Nhận diện ảnh")
    notebook.add(vid_tab, text="Nhận diện video")
    notebook.add(cam_tab, text="Camera trực tiếp")
    
    # ====== IMAGE TAB ======
    img_conf_scale = Scale(img_tab, from_=10, to=95, orient=HORIZONTAL, 
                         label="Ngưỡng tin cậy (%)")
    img_conf_scale.set(50)
    img_conf_scale.pack(fill=X, padx=5, pady=2)
    
    img_iou_scale = Scale(img_tab, from_=10, to=95, orient=HORIZONTAL,
                        label="Ngưỡng IoU (%)")
    img_iou_scale.set(50)
    img_iou_scale.pack(fill=X, padx=5, pady=2)
    
    img_size_frame = Frame(img_tab)
    img_size_frame.pack(fill=X, padx=5, pady=5)
    
    Label(img_size_frame, text="Kích thước ảnh: ").pack(side=LEFT)
    img_size_var = IntVar(value=640)
    Spinbox(img_size_frame, from_=320, to=1280, increment=32,
           textvariable=img_size_var, width=5).pack(side=LEFT)
    
    img_control_frame = Frame(img_tab)
    img_control_frame.pack(fill=X, pady=5)
    
    Button(img_control_frame, text="Tải ảnh", width=15, 
          command=lambda: upload_image(img_path_label, img_result_label, img_label_text)).pack(side=LEFT, padx=5)
    
    Button(img_control_frame, text="Nhận diện", width=15,
          command=lambda: detect_image(
              img_path_label, img_result_label, img_label_text, img_status_label, 
              img_progress_bar, img_conf_scale.get()/100, img_iou_scale.get()/100, img_size_var.get()
          )).pack(side=LEFT, padx=5)
    
    Button(img_control_frame, text="Lưu kết quả", width=15,
          command=lambda: save_image_result(img_result_label, img_status_label)).pack(side=LEFT, padx=5)
    
    img_path_label = Label(img_tab, text="Chưa tải ảnh", fg="blue")
    img_path_label.pack(pady=5)
    
    img_progress_bar = ttk.Progressbar(img_tab, orient="horizontal", length=600, mode="determinate")
    
    img_result_label = Label(img_tab, bd=2, relief="solid", width=600, height=400)
    img_result_label.pack(pady=10)
    
    img_label_text = Label(img_tab, text="", font=("Arial", 12))
    img_label_text.pack(pady=5)
    
    img_status_label = Label(img_tab, text="", fg="blue")
    img_status_label.pack(pady=5)
    
    # ====== VIDEO TAB ======
    vid_conf_scale = Scale(vid_tab, from_=10, to=95, orient=HORIZONTAL,
                         label="Ngưỡng tin cậy (%)")
    vid_conf_scale.set(50)
    vid_conf_scale.pack(fill=X, padx=5, pady=2)
    
    vid_iou_scale = Scale(vid_tab, from_=10, to=95, orient=HORIZONTAL,
                        label="Ngưỡng IoU (%)")
    vid_iou_scale.set(50)
    vid_iou_scale.pack(fill=X, padx=5, pady=2)
    
    vid_size_frame = Frame(vid_tab)
    vid_size_frame.pack(fill=X, padx=5, pady=5)
    
    Label(vid_size_frame, text="Kích thước ảnh: ").pack(side=LEFT)
    vid_size_var = IntVar(value=640)
    Spinbox(vid_size_frame, from_=320, to=1280, increment=32,
           textvariable=vid_size_var, width=5).pack(side=LEFT)
    
    vid_control_frame = Frame(vid_tab)
    vid_control_frame.pack(fill=X, pady=5)
    
    Button(vid_control_frame, text="Tải video", width=15,
          command=lambda: upload_video(vid_path_label, vid_result_label, vid_label_text)).pack(side=LEFT, padx=5)
    
    Button(vid_control_frame, text="Xử lý video", width=15,
          command=lambda: process_video(
              vid_path_label, vid_result_label, vid_label_text, vid_status_label,
              vid_progress_bar, vid_conf_scale.get()/100, vid_iou_scale.get()/100, vid_size_var.get()
          )).pack(side=LEFT, padx=5)
    
    Button(vid_control_frame, text="Lưu kết quả", width=15,
          command=lambda: save_video_result(vid_result_label, vid_status_label)).pack(side=LEFT, padx=5)
    
    vid_path_label = Label(vid_tab, text="Chưa tải video", fg="blue")
    vid_path_label.pack(pady=5)
    
    vid_progress_bar = ttk.Progressbar(vid_tab, orient="horizontal", length=600, mode="determinate")
    
    vid_result_label = Label(vid_tab, bd=2, relief="solid", width=600, height=400)
    vid_result_label.pack(pady=10)
    
    vid_label_text = Label(vid_tab, text="", font=("Arial", 12))
    vid_label_text.pack(pady=5)
    
    vid_status_label = Label(vid_tab, text="", fg="blue")
    vid_status_label.pack(pady=5)
    
    # ====== CAMERA TAB ======
    cam_conf_scale = Scale(cam_tab, from_=10, to=95, orient=HORIZONTAL,
                          label="Ngưỡng tin cậy (%)")
    cam_conf_scale.set(50)
    cam_conf_scale.pack(fill=X, padx=5, pady=2)
    
    cam_iou_scale = Scale(cam_tab, from_=10, to=95, orient=HORIZONTAL,
                         label="Ngưỡng IoU (%)")
    cam_iou_scale.set(50)
    cam_iou_scale.pack(fill=X, padx=5, pady=2)
    
    cam_size_frame = Frame(cam_tab)
    cam_size_frame.pack(fill=X, padx=5, pady=5)
    
    Label(cam_size_frame, text="Kích thước ảnh: ").pack(side=LEFT)
    cam_size_var = IntVar(value=640)
    Spinbox(cam_size_frame, from_=320, to=1280, increment=32,
           textvariable=cam_size_var, width=5).pack(side=LEFT)
    
    cam_control_frame = Frame(cam_tab)
    cam_control_frame.pack(fill=X, pady=5)
    
    Button(cam_control_frame, text="Bắt đầu camera", width=15,
          command=lambda: start_camera(
              cam_result_label, cam_label_text, cam_status_label,
              cam_conf_scale.get()/100, cam_iou_scale.get()/100, cam_size_var.get()
          )).pack(side=LEFT, padx=5)
    
    Button(cam_control_frame, text="Dừng camera", width=15,
          command=lambda: stop_camera(cam_status_label)).pack(side=LEFT, padx=5)
    
    Button(cam_control_frame, text="Chụp ảnh", width=15,
          command=lambda: capture_snapshot(cam_result_label, cam_status_label)).pack(side=LEFT, padx=5)
    
    Button(cam_control_frame, text="Ghi/Dừng video", width=15, 
          command=lambda: toggle_recording(cam_result_label, cam_status_label)).pack(side=LEFT, padx=5)
    
    cam_status_label = Label(cam_tab, text="Camera chưa khởi động", fg="blue")
    cam_status_label.pack(pady=5)
    
    cam_result_label = Label(cam_tab, bd=2, relief="solid", width=600, height=400)
    cam_result_label.pack(pady=10)
    
    cam_label_text = Label(cam_tab, text="", font=("Arial", 12))
    cam_label_text.pack(pady=5)
    
    # Load model on startup
    load_model_thread = threading.Thread(
        target=lambda: load_model(img_status_label), 
        daemon=True
    )
    load_model_thread.start()
    
    return root

if __name__ == '__main__':
    root = create_gui()
    root.mainloop()
    
    # Clean up resources when closing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()