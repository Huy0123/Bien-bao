import torch
from ultralytics import YOLO
import cv2
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import threading
import time


# Global variables
uploaded_file_path = None
video_writer = None
current_processed_image = None

# Parameters
DETECTION_INTERVAL = 24
SLOW_DELAY = 0.0

# Setup đường dẫn đầu ra
def setup_output_directories():
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Thư mục theo ngày
    today_dir = os.path.join(output_dir, time.strftime("%Y-%m-%d"))
    os.makedirs(today_dir, exist_ok=True)

    return today_dir

# Upload ảnh hoặc video
def upload_file():
    global uploaded_file_path, video_writer, current_processed_image
    try:
        file_path = filedialog.askopenfilename(
            title="Chọn tệp",
            filetypes=[
                ("Tệp hình ảnh", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("Tệp video", "*.mp4 *.avi *.mov *.mkv *.flv")
            ]
        )
        if file_path:
            uploaded_file_path = file_path
            status_label.config(text=f"Đã tải tệp: {os.path.basename(file_path)}")
            result_label.config(image='', text='')
            result_label.image = None
            label_result_text.config(text="")
            current_processed_image = None
            clean_resources()
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể tải tệp: {e}")

# Hiển thị kết quả
def update_result(photo, detected_labels):
    result_label.config(image=photo)
    result_label.image = photo

    # Hiển thị nhãn đã phát hiện
    # unique_labels = []
    # label_counts = {}
    #
    # for label in detected_labels:
    #     class_name = label.split()[0]
    #     if class_name in label_counts:
    #         label_counts[class_name] += 1
    #     else:
    #         label_counts[class_name] = 1
    #
    # for name, count in label_counts.items():
    #     unique_labels.append(f"{name} ({count})")
    #
    # if unique_labels:
    #     label_result_text.config(text="Nhận diện: " + ", ".join(unique_labels))
    # else:
    #     label_result_text.config(text="Không nhận diện được đối tượng")

def get_unique(path):
    if not os.path.exists(path):
        return path
    base_name, ext = os.path.splitext(path)
    count = 1
    while True:
        new_path = f"{base_name}_{count}{ext}"
        if not os.path.exists(new_path):
            return new_path
        count += 1
# Lưu kết quả
def save_current_result():
    global current_processed_image, uploaded_file_path

    # Kiểm tra xem tập tin đầu vào là ảnh hay video
    if not uploaded_file_path:
        messagebox.showinfo("Thông báo", "Không có kết quả để lưu")
        return

    ext = os.path.splitext(uploaded_file_path)[1].lower()

    # Nếu là ảnh
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
        if current_processed_image is None:
            messagebox.showinfo("Thông báo", "Không có ảnh đã xử lý để lưu")
            return

        today_dir = setup_output_directories()
        base_name = os.path.splitext(os.path.basename(uploaded_file_path))[0]
        default_name = f"{base_name}_detected.jpg"

        save_path = filedialog.asksaveasfilename(
            initialdir=today_dir,
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )

        if save_path:
            try:
                cv2.imwrite(save_path, cv2.cvtColor(current_processed_image, cv2.COLOR_RGB2BGR))
                messagebox.showinfo("Thành công", f"Đã lưu ảnh tại: {save_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {e}")

    # Nếu là video
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
        # Sử dụng biến lưu đường dẫn video đã xử lý
        if not hasattr(save_current_result, "last_video_path"):
            messagebox.showinfo("Thông báo", "Không có video đã xử lý để lưu")
            return

        source_video_path = save_current_result.last_video_path
        if not os.path.exists(source_video_path):
            messagebox.showinfo("Thông báo", "File video đã xử lý không tồn tại")
            return

        # Cho phép người dùng chọn vị trí lưu mới
        base_name = os.path.splitext(os.path.basename(uploaded_file_path))[0]
        default_name = f"{base_name}_detected{ext}"

        save_path = filedialog.asksaveasfilename(
            defaultextension=ext,
            initialfile=default_name,
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("All Files", "*.*")]
        )

        if save_path:
            try:
                import shutil
                shutil.copy2(source_video_path, save_path)
                messagebox.showinfo("Thành công", f"Đã lưu video tại: {save_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu video: {e}")

# Giải phóng tài nguyên
def clean_resources():
    global video_writer
    if video_writer is not None:
        video_writer.release()
        video_writer = None

    # Giải phóng bộ nhớ GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Tiến hành dự đoán
def predict():
    global video_writer, current_processed_image
    if not uploaded_file_path:
        status_label.config(text="Vui lòng tải lên tệp trước khi nhận diện")
        return

    # Sử dụng giá trị từ thanh trượt (conf, iou và kích thước ảnh)
    conf_threshold = conf_scale.get() / 100
    iou_threshold = iou_scale.get() / 100
    img_size = imgsz_var.get()
    # Kiểm tra thiết bị có gpu không
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('runs/detect/train/weights/best.pt')
    names = model.names  # sử dụng tên lớp từ dataset
    ext = os.path.splitext(uploaded_file_path)[1].lower()

    # Chuẩn bị thư mục đầu ra
    today_dir = setup_output_directories()
    base_name = os.path.splitext(os.path.basename(uploaded_file_path))[0]

    try:
        # Xử lý ảnh
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            progress_bar.pack(fill=X, padx=5, pady=5)
            progress_bar["value"] = 10
            root.update_idletasks()
            # Dự đoán
            results = model.predict(
                uploaded_file_path,
                device=device,
                save=False,
                imgsz=img_size,
                conf=conf_threshold,
                iou=iou_threshold
            )

            progress_bar["value"] = 100
            progress_bar.pack_forget()

            if results:
                res = results[0]
                img = cv2.imread(uploaded_file_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                detected_labels = []
                if hasattr(res, "boxes") and res.boxes is not None:
                    for box in res.boxes:
                        # Lấy tọa độ khung bounding box
                        xy = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, xy)

                        # Lấy nhãn và độ tin cậy
                        label_id = int(box.cls[0].item())
                        confidence = box.conf[0].item()
                        conf_str = f"{confidence:.2f}"
                        text = f"{names[label_id]} {conf_str}"

                        # Vẽ khung bao
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 255), 2)

                        # Tính toán vị trí để hiển thị chữ
                        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        text_x = x1
                        text_y = y1 - 10 if y1 > 20 else y1 + 30

                        # Vẽ nền cho chữ để dễ đọc hơn
                        cv2.rectangle(img_rgb, (text_x, text_y - text_h - 2),
                                      (text_x + text_w, text_y + 2), (128, 128, 128), -1)

                        # Vẽ chữ
                        cv2.putText(img_rgb, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        detected_labels.append(text)

                progress_bar["value"] = 80
                root.update_idletasks()

                current_processed_image = img_rgb.copy()
                pil_image = Image.fromarray(img_rgb)
                display_img = pil_image.resize((600, 400), resample=Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(display_img)
                update_result(photo, detected_labels)
                status_label.config(text="Nhận diện thành công")
            else:
                status_label.config(text="Không nhận diện được đối tượng")

            progress_bar["value"] = 100
            progress_bar.grid_forget()

        # Xử lý video với detect định kỳ và giữ bounding box
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
            cap = cv2.VideoCapture(uploaded_file_path)
            if not cap.isOpened():
                status_label.config(text="Không thể mở video")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Tạo tên file đầu ra
            output_path = os.path.join(today_dir, f"{base_name}_detected{ext}")
            output_path = get_unique(output_path)
            video_writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )

            def video_thread():
                last_detections = []  # lưu bounding boxes và labels
                frame_count = 0
                try:
                    cap2 = cv2.VideoCapture(uploaded_file_path)
                    while True:
                        ret, frame = cap2.read()
                        if not ret:
                            break

                        # Cập nhật progress
                        progress = f"Đang xử lý: {frame_count}/{total_frames} frames ({int(frame_count * 100 / total_frames)}%)"
                        result_label.after(0, lambda p=progress: status_label.config(text=p))
                        video_progress["value"] = int(frame_count * 100 / total_frames)

                        # Detect mỗi DETECTION_INTERVAL frame hoặc frame đầu tiên
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
                                    # Lấy tọa độ khung bounding box
                                    xy = box.xyxy[0].tolist()

                                    # Lấy nhãn và độ tin cậy
                                    label_id = int(box.cls[0].item())
                                    confidence = box.conf[0].item()
                                    label = names[label_id]
                                    conf_str = f"{confidence:.2f}"
                                    text = f"{label} {conf_str}"

                                    new_detections.append((xy, text, (0, 255, 255)))  # Box, text, color
                                last_detections = new_detections

                        # Vẽ bounding box trên khung hiện tại
                        processed_frame = frame.copy()
                        detected_labels = []

                        for (xy, text, color) in last_detections:
                            x1, y1, x2, y2 = map(int, xy)

                            # Vẽ khung bao
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)

                            # Tính toán vị trí để hiển thị chữ
                            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            text_x = x1
                            text_y = y1 - 10 if y1 > 20 else y1 + 30

                            # Vẽ nền cho chữ
                            cv2.rectangle(processed_frame, (text_x, text_y - text_h - 2),
                                          (text_x + text_w, text_y + 2), (128, 128, 128), -1)

                            # Vẽ chữ
                            cv2.putText(processed_frame, text, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            detected_labels.append(text)

                        # Ghi video
                        if video_writer is not None:
                            video_writer.write(processed_frame)

                        # Cập nhật GUI
                        img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(img_rgb)
                        disp = pil.resize((600, 400), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(disp)

                        # Hiển thị ảnh và nhãn đã phát hiện (loại bỏ trùng lặp)
                        unique_labels = list(dict.fromkeys(detected_labels))
                        result_label.after(0, lambda p=photo, d=unique_labels: update_result(p, d))

                        time.sleep(SLOW_DELAY)
                        frame_count += 1

                except Exception as e:
                    messagebox.showerror("Lỗi", f"Không thể xử lý video: {e}")
                finally:
                    if cap2 is not None:
                        cap2.release()
                    if video_writer is not None:
                        video_writer.release()

                    save_current_result.last_video_path = output_path
                    result_label.after(0, lambda: status_label.config(
                        text=f"Hoàn thành xử lý video (đã lưu tại: {output_path})"))
                    video_progress["value"] = 100

            video_progress.pack(fill=X, padx=5, pady=5)
            video_progress["value"] = 0
            threading.Thread(target=video_thread, daemon=True).start()
            status_label.config(text="Đang xử lý video...")

    except Exception as e:
        progress_bar.pack_forget()
        video_progress.pack_forget()
        messagebox.showerror("Lỗi", f"Không thể thực hiện dự đoán: {e}")
        clean_resources()


def create_gui():
    global result_label, status_label, label_result_text, conf_scale, iou_scale, imgsz_var, progress_bar, video_progress
    root = Tk()
    root.title("Nhận diện biển báo giao thông Việt Nam")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    ww, wh = 800, 750
    root.geometry(f"{ww}x{wh}+{(sw - ww) // 2}+{(sh - wh) // 2}")

    frame = Frame(root, padx=10, pady=10)
    frame.pack(expand=True, fill=BOTH)

    # Tạo frame cho nút điều khiển
    control_frame = Frame(frame)
    control_frame.pack(fill=X, pady=5)

    Button(control_frame, text="Tải tệp", command=upload_file).pack(side=LEFT, padx=5)
    Button(control_frame, text="Nhận diện", command=predict).pack(side=LEFT, padx=5)
    Button(control_frame, text="Lưu kết quả", command=save_current_result).pack(side=LEFT, padx=5)

    # Frame cho tham số
    param_frame = LabelFrame(frame, text="Tham số phát hiện")
    param_frame.pack(fill=X, pady=5)

    # Thanh trượt cho threshold
    conf_scale = Scale(param_frame, from_=10, to=95, orient=HORIZONTAL,
                       label="Ngưỡng tin cậy (%)")
    conf_scale.set(50)  # Mặc định 0.5
    conf_scale.pack(fill=X, padx=5, pady=2)

    iou_scale = Scale(param_frame, from_=10, to=95, orient=HORIZONTAL,
                      label="Ngưỡng IoU (%)")
    iou_scale.set(50)  # Mặc định 0.5
    iou_scale.pack(fill=X, padx=5, pady=2)

    # Frame cho kích thước ảnh
    imgsz_frame = Frame(param_frame)
    imgsz_frame.pack(fill=X, padx=5, pady=5)

    Label(imgsz_frame, text="Kích thước ảnh: ").pack(side=LEFT)
    imgsz_var = IntVar(value=640)
    Spinbox(imgsz_frame, from_=320, to=1280, increment=32,
            textvariable=imgsz_var, width=5).pack(side=LEFT)

    status_label = Label(frame, text="Chưa tải tệp", fg="blue")
    status_label.pack(pady=5)

    # Thanh tiến trình
    progress_bar = ttk.Progressbar(frame, orient="horizontal", length=600, mode="determinate")
    video_progress = ttk.Progressbar(frame, orient="horizontal", length=600, mode="determinate")

    result_label = Label(frame, bd=2, relief="solid", width=600, height=400)
    result_label.pack(pady=10)

    label_result_text = Label(frame, text="", font=("Arial", 12))
    label_result_text.pack(pady=5)

    return root


if __name__ == '__main__':
    root = create_gui()
    root.mainloop()