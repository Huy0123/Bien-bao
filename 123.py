import torch
from ultralytics import YOLO
import cv2
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading
import time

# Global variables
uploaded_file_path = None
video_writer = None

# Parameters
DETECTION_INTERVAL = 12
SLOW_DELAY = 0.03


def upload_file():
    global uploaded_file_path, video_writer
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
            if video_writer:
                video_writer.release()
                video_writer = None
    except Exception as e:
        messagebox.showerror("Error", f"Failed to upload file: {e}")


def update_result(photo, detected_labels):
    result_label.config(image=photo)
    result_label.image = photo
    if detected_labels:
        label_result_text.config(text="Nhận diện: " + ", ".join(detected_labels))
    else:
        label_result_text.config(text="Không nhận diện được đối tượng")


def predict():
    global video_writer
    if not uploaded_file_path:
        status_label.config(text="Vui lòng tải lên tệp trước khi nhận diện")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('runs/detect/train/weights/best.pt')
    names = model.names  # sử dụng tên lớp từ dataset
    ext = os.path.splitext(uploaded_file_path)[1].lower()

    try:
        # Xử lý ảnh tĩnh
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            results = model.predict(uploaded_file_path, device=device, save=False, imgsz=640, conf=0.5, iou=0.5)
            if results:
                res = results[0]
                img = cv2.imread(uploaded_file_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                detected_labels = []
                for box in res.boxes:
                    # Lấy tọa độ khung bounding box
                    xy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xy)

                    # Lấy nhãn và độ tin cậy
                    label = names[int(box.cls[0].item())]
                    confidence = box.conf[0].item()
                    conf_str = f"{confidence:.2f}"
                    text = f"{label} {conf_str}"

                    # Vẽ khung bao
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    # Tính toán vị trí để hiển thị chữ nằm giữa khung
                    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_x = x1 + (x2 - x1 - text_w) // 2
                    text_y = y1- 10

                    # Vẽ chữ
                    cv2.rectangle(img_rgb, (text_x - 2, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2),
                                  (128, 128, 128), -1)
                    cv2.putText(img_rgb, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


                    detected_labels.append(text)

                pil_image = Image.fromarray(img_rgb)
                display_img = pil_image.resize((600, 400), resample=Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(display_img)
                update_result(photo, detected_labels)
                status_label.config(text="Nhận diện thành công")
            else:
                status_label.config(text="Không nhận diện được đối tượng")


        # Xử lý video với detect định kỳ và giữ bounding box
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
            cap = cv2.VideoCapture(uploaded_file_path)
            if not cap.isOpened():
                status_label.config(text="Không thể mở video")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


            video_writer = cv2.VideoWriter(
                "output_video.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )

            def video_thread():
                last_boxes = []  # lưu bounding box và label string
                frame_count = 0
                try:
                    cap2 = cv2.VideoCapture(uploaded_file_path)
                    while True:
                        ret, frame = cap2.read()
                        if not ret:
                            break

                        # Detect mỗi DETECTION_INTERVAL frame
                        if frame_count % DETECTION_INTERVAL == 0:
                            results = model.predict(frame, device=device, save=False, imgsz=640, conf=0.5, iou=0.5)
                            if results and results[0].boxes is not None:
                                res = results[0]
                                new_boxes = []
                                for box in res.boxes:
                                    xy = box.xyxy[0].tolist()
                                    label = names[int(box.cls[0].item())]
                                    new_boxes.append((xy, label))
                                last_boxes = new_boxes
                            else:
                                last_boxes = []

                        # Vẽ bounding box lưu trữ
                        detected_labels = []
                        for (xy, label) in last_boxes:
                            x1, y1, x2, y2 = map(int, xy)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                            cv2.putText(frame, label, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                            detected_labels.append(label)

                        # Ghi video (vẫn ở tốc độ gốc)
                        video_writer.write(frame)

                        # Cập nhật GUI (chậm lại bằng time.sleep)
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(img_rgb)
                        disp = pil.resize((600,400), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(disp)
                        result_label.after(0, lambda: update_result(photo, list(dict.fromkeys(detected_labels))))
                        time.sleep(SLOW_DELAY)

                        frame_count += 1
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to process video: {e}")
                finally:
                    video_writer.release()
                    result_label.after(0, lambda: status_label.config(text="Hoàn thành xử lý video (kết quả được lưu tự động)"))

            threading.Thread(target=video_thread, daemon=True).start()
            status_label.config(text="Đang xử lý video...")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict: {e}")


def create_gui():
    global result_label, status_label, label_result_text
    root = Tk()
    root.title("Nhận diện biển báo giao thông Việt Nam")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    ww, wh = 800, 750
    root.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{(sh-wh)//2}")

    frame = Frame(root, padx=10, pady=10)
    frame.pack(expand=True)

    Button(frame, text="Upload File", command=upload_file).pack(pady=10)
    Button(frame, text="Nhận diện", command=predict).pack(pady=10)

    status_label = Label(frame, text="Chưa tải tệp", fg="blue")
    status_label.pack(pady=5)

    result_label = Label(frame, bd=2, relief="solid", width=600, height=400)
    result_label.pack(pady=10)

    label_result_text = Label(frame, text="", font=("Arial", 12))
    label_result_text.pack(pady=5)

    root.mainloop()

if __name__ == '__main__':
    create_gui()
