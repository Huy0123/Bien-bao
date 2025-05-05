import os
import shutil
from datetime import datetime
import torch
from ultralytics import YOLO
import cv2
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np

from image import ImageDetector
from video import VideoDetector
from camera import CameraDetector


def setup_output_directories():
    base = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(base, exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    d = os.path.join(base, today)
    os.makedirs(d, exist_ok=True)
    return d

def get_unique_filename(path):
    if not os.path.exists(path): return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        p = f"{base}_{i}{ext}"
        if not os.path.exists(p): return p
        i += 1

class MainApp:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.label_map = {
            "Ben xe buyt": "Bến xe buýt",
            "Cam con lai": "Biển báo cấm", 
            "Cam do": "Cấm đỗ",
            "Cam dung va do": "Cấm dừng và đỗ",
            "Cam nguoc chieu": "Cấm ngược chiều",
            "Cam quay dau xe": "Cấm quay đầu xe",
            "Cam re phai": "Cấm rẽ phải",
            "Cam re trai": "Cấm rẽ trái",
            "Chi dan": "Biển báo chỉ dẫn",
            "Chieu cao toi da": "Chiều cao tối đa",
            "Duong giao nhau": "Đường giao nhau",
            "Giao voi duong khong uu tien": "Giao với đường không ưu tiên",
            "Giao voi duong uu tien": "Giao với đường ưu tiên",
            "Het tat ca lenh cam": "Hết tất cả lệnh cấm",
            "Hieu lenh": "Biển báo hiệu lệnh",
            "Nguoi di bo cat ngang": "Người đi bộ cắt ngang",
            "Nguy hiem": "Nguy hiểm",
            "Re phai": "Rẽ phải",
            "Re trai": "Rẽ trái", 
            "Toc do toi da": "Tốc độ tối đa",
            "Trong tai toi da": "Trọng tải tối đa",
            "Vong xuyen": "Vòng xuyến"
        }
        self.model = YOLO('runs/detect/train2/weights/best.pt')
        self._configure_model()
        
        # Khởi tạo các detector
        self.img_det = ImageDetector(self.model, self.device)
        self.vid_det = VideoDetector(self.model, self.device)
        self.cam_det = CameraDetector(self.model, self.device)
        
        self.root = None
        self.build_ui()

    def _configure_model(self):
        if hasattr(self.model, 'overrides'):
            self.model.overrides['conf'] = 0.5
            self.model.overrides['line_width'] = None
            self.model.overrides['box'] = True
   
    def build_ui(self):
        r = Tk(); self.root = r
        r.title('Nhận diện biển báo giao thông Việt Nam')
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        ww, wh = 800, 750
        r.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{(sh-wh)//2}")
        
        # Tạo các tab điều khiển
        nb = ttk.Notebook(r)
        nb.pack(expand=True, fill=BOTH, padx=5, pady=5)
        nb.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Tạo các tab và nút điều khiển của chúng
        self._build_image_tab(nb)
        self._build_video_tab(nb)
        self._build_camera_tab(nb)
        
        r.mainloop()

    # Xây dựng tab nhận diện ảnh
    def _build_image_tab(self, notebook):
        self.create_detection_tab(notebook, 'Nhận diện ảnh', 'img')
        buttons = [
            ('Tải ảnh', self.on_image_upload),
            ('Nhận diện', self.on_image_detect),
            ('Lưu', self.on_image_save)
        ]
        self._create_buttons(self.img_button_frame, buttons)

    # Xây dựng tab nhận diện video
    def _build_video_tab(self, notebook):
        self.create_detection_tab(notebook, 'Nhận diện video', 'vid')
        buttons = [
            ('Tải video', self.on_video_upload),
            ('Xử lý', self.on_video_process),
            ('Lưu', self.on_video_save)
        ]
        self._create_buttons(self.vid_button_frame, buttons)

    # Xây dựng tab camera
    def _build_camera_tab(self, notebook):
        self.create_detection_tab(notebook, 'Camera trực tiếp', 'cam')
        buttons = [
            ('Bắt đầu', self.on_cam_start),
            ('Dừng', self.on_cam_stop),
            ('Chụp', self.on_cam_snapshot),
            ('Ghi/Dừng', self.on_cam_toggle)
        ]
        self._create_buttons(self.cam_button_frame, buttons)

    def _create_buttons(self, frame, button_configs):
        for text, cmd in button_configs:
            Button(frame, text=text, command=cmd, width=15).pack(side=LEFT, padx=5)

    # Tạo các điều khiển cho tab nhận diện ảnh và video
    def create_detection_controls(self, tab, prefix):
        # Confidence slider
        setattr(self, f"{prefix}_conf", Scale(tab, from_=10, to=95, orient=HORIZONTAL, label='Tin cậy (%)'))
        getattr(self, f"{prefix}_conf").set(50)
        getattr(self, f"{prefix}_conf").pack(fill=X)
        
        # IoU slider
        setattr(self, f"{prefix}_iou", Scale(tab, from_=10, to=95, orient=HORIZONTAL, label='IoU (%)'))
        getattr(self, f"{prefix}_iou").set(50)
        getattr(self, f"{prefix}_iou").pack(fill=X)
        
        # Kich thước ảnh
        f = Frame(tab); f.pack(fill=X)
        Label(f, text='Kích thước:').pack(side=LEFT)
        setattr(self, f"{prefix}_size", IntVar(value=640))
        Spinbox(f, from_=320, to=1280, increment=32, 
                textvariable=getattr(self, f"{prefix}_size"), width=5).pack(side=LEFT)
        
        # Tạo frame nút điều khiển ở đây
        setattr(self, f"{prefix}_button_frame", Frame(tab))
        getattr(self, f"{prefix}_button_frame").pack(fill=X, pady=5)
        
        # Path and progress indicators
        path_text = "" if prefix == 'cam' else "Chưa tải"
        setattr(self, f"{prefix}_path", Label(tab, text=path_text, fg='blue'))
        getattr(self, f"{prefix}_path").pack(pady=5)
        setattr(self, f"{prefix}_prog", ttk.Progressbar(tab, orient='horizontal', length=600, mode='determinate'))


    def create_detection_tab(self, notebook, title, prefix):
        tab = Frame(notebook)
        notebook.add(tab, text=title)
        
        # Tạo các điều khiển chung cho ảnh và video
        self.create_detection_controls(tab, prefix)
        
        # Khung hiển thị
        setattr(self, f"{prefix}_disp", Label(tab, bd=2, relief='solid', width=600, height=400))
        getattr(self, f"{prefix}_disp").pack(pady=10)
        
        # Text labels
        setattr(self, f"{prefix}_txt", Label(tab, font=('Arial', 12)))
        getattr(self, f"{prefix}_txt").pack()
        
        # Status labels
        setattr(self, f"{prefix}_status", Label(tab, fg='blue'))
        getattr(self, f"{prefix}_status").pack(pady=5)
        
        return tab

    # Tạo hình ảnh trống khi không có file nào được tải lên
    def _create_blank_image(self):
        blank = Image.new('RGB', (600, 400), color='white')
        return ImageTk.PhotoImage(blank)

    def on_tab_changed(self, event):
        tab_id = event.widget.select()
        tab_name = event.widget.tab(tab_id, "text")
        
        if tab_name == 'Nhận diện ảnh':
            self._reset_tab('img')
        elif tab_name == 'Nhận diện video':
            self._reset_tab('vid')
        elif tab_name == 'Camera trực tiếp':
            self._reset_tab('cam')

    def _reset_tab(self, prefix):
        blank_photo = self._create_blank_image()
        disp = getattr(self, f"{prefix}_disp")
        disp.config(image=blank_photo)
        disp.image = blank_photo
        
        # Reset các nhãn và trạng thái
        getattr(self, f"{prefix}_txt").config(text='')
        getattr(self, f"{prefix}_status").config(text='' if prefix != 'cam' else 'Chưa khởi động')
        
        if prefix != 'cam':
            getattr(self, f"{prefix}_path").config(text='Chưa tải')
        else:
            getattr(self, f"{prefix}_path").config(text='')
        # Xóa dữ liệu liên kết
        path_attr = getattr(self, f"{prefix}_path")
        if hasattr(path_attr, 'file_path'):
            delattr(path_attr, 'file_path')
        
        # Xử lý đặc biệt cho từng tab
        if prefix == 'img' and hasattr(disp, 'processed'):
            delattr(disp, 'processed')
        elif prefix == 'vid' and hasattr(self.vid_det, 'output_path'):
            delattr(self.vid_det, 'output_path')
        elif prefix == 'cam' and hasattr(self.cam_det, 'running') and self.cam_det.running:
            self.on_cam_stop()

    def display_image(self, image, display_widget, text_widget, labels=None):
        # Xử lý hình ảnh một cách nhất quán
        if isinstance(image, np.ndarray):
            # Chuyển đổi từ OpenCV BGR sang RGB
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            display_widget.current = image  # Lưu frame hiện tại
        else:
            img = image
            
        # Điều chỉnh kích thước và hiển thị
        img = img.resize((600, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        display_widget.config(image=photo)
        display_widget.image = photo
        
        # Cập nhật thông tin nhận diện
        if labels is not None:
            self._update_detection_labels(labels, text_widget)

    # Cập nhật tên biển báo
    def _update_detection_labels(self, labels, text_widget):
        counts = {}
        for t in labels:
            # Tách nhãn và độ tin cậy
            parts = t.split()
            confidence_part = parts[-1] if parts and parts[-1][0].isdigit() else ""
            name_parts = parts[:-1] if confidence_part else parts
            full_name = " ".join(name_parts)
                
            # Tìm tên tiếng Việt
            vietnamese_name = self.label_map.get(full_name, full_name)
            counts[vietnamese_name] = counts.get(vietnamese_name, 0) + 1
        
        text = ', '.join(f"{k} ({v})" for k, v in counts.items())
        text_widget.config(text='Nhận diện: ' + (text or 'Không có'))

    # Chức năng upload ảnh
    def on_image_upload(self):
        path = filedialog.askopenfilename(filetypes=[('Images','*.jpg *.jpeg *.png')])
        if not path: return
        self.img_path.file_path = path
        self.img_path.config(text=os.path.basename(path))
        img = cv2.imread(path)
        self.display_image(img, self.img_disp, self.img_txt)

    #Cấu hình nhận diện ảnh
    def on_image_detect(self):
        if not hasattr(self.img_path, 'file_path'):
            self.img_status.config(text='Vui lòng tải ảnh')
            return
            
        frame, labels = self.img_det.detect(
            self.img_path.file_path,
            self.img_conf.get()/100,
            self.img_iou.get()/100,
            self.img_size.get()
        )
        self.display_image(frame, self.img_disp, self.img_txt, labels)
        self.img_disp.processed = frame
        self.img_status.config(text='Xong')

    # Lưu ảnh đã xử lý
    def on_image_save(self):
        if not hasattr(self.img_disp, 'processed'):
            messagebox.showinfo('Thông báo','Chưa có kết quả')
            return
            
        d = setup_output_directories()
        fname = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        p = filedialog.asksaveasfilename(initialdir=d, initialfile=fname, defaultextension='.jpg')
        if p:
            cv2.imwrite(p, self.img_disp.processed)
            messagebox.showinfo('Thành công',f'Lưu tại {p}')

    # Xử lý video
    def on_video_upload(self):
        path = filedialog.askopenfilename(filetypes=[('Videos','*.mp4 *.avi')])
        if not path: return
        self.vid_path.file_path = path
        self.vid_path.config(text=os.path.basename(path))
        # Đọc video đầu tiên để hiển thị
        cap = cv2.VideoCapture(path)
        ret, frm = cap.read()
        cap.release()
        if ret:
            self.display_image(frm, self.vid_disp, self.vid_txt)

    def on_video_process(self):
        if not hasattr(self.vid_path, 'file_path'):
            self.vid_status.config(text='Vui lòng tải video')
            return
            
        # Hiển thị thanh tiến trình
        self.vid_prog.pack(fill=X, padx=5, pady=5)
        self.vid_prog['value'] = 0
        self.vid_status.config(text='Đang xử lý...')
        
        # Tạo đường dẫn đầu ra
        d = setup_output_directories()
        fname = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(d, fname)
        
        # Xử lý video với các callback
        self.vid_det.process(
            self.vid_path.file_path, 
            output_path,
            self.vid_conf.get()/100, 
            self.vid_iou.get()/100, 
            self.vid_size.get(),
            lambda i, total: self._update_progress(i, total),
            lambda frm, labels: self.display_image(frm, self.vid_disp, self.vid_txt, labels),
            lambda p: self._video_processing_completed(p)
        )

    def _update_progress(self, i, total):
        self.vid_prog['value'] = i*100/total
        self.vid_status.config(text=f"Đang xử lý: {i}/{total} frames ({int(i*100/total)}%)")

    def _video_processing_completed(self, p):
        self.vid_prog['value'] = 100
        self.vid_prog.pack_forget()
        self.vid_status.config(text=f'Xong: {os.path.basename(p)}')
        self.vid_det.output_path = p

    def on_video_save(self):
        if not hasattr(self.vid_det, 'output_path') or not self.vid_det.output_path:
            messagebox.showinfo('Thông báo','Chưa có video xử lý')
            return
            
        p = filedialog.asksaveasfilename(
            defaultextension='.mp4',
            initialfile=os.path.basename(self.vid_det.output_path)
        )
        
        if p: 
            shutil.copy2(self.vid_det.output_path, p)
            messagebox.showinfo('Thành công', f'Lưu tại {p}')

    # CAMERA TAB FUNCTIONS
    def on_cam_start(self):
        self.cam_status.config(text='Camera đang chạy')
        
        def frame_cb(frm):
            self.display_image(frm, self.cam_disp, self.cam_txt)
            
        def label_cb(labels):
            self._update_detection_labels(labels, self.cam_txt)
            
        self.cam_det.start(conf_thresh=self.cam_conf.get()/100,
                            iou_thresh=self.cam_iou.get()/100,
                            img_size=self.cam_size.get(),
                            frame_callback = frame_cb, label_callback=label_cb)
    def on_cam_stop(self):
        # Dừng camera trước
        self.cam_det.stop()
        # Đảm bảo camera thực sự dừng
        self.root.after(100, self._complete_camera_stop)

    def _complete_camera_stop(self):
        # Phương thức này sẽ chạy sau khi camera đã thực sự dừng
        self.cam_status.config(text='Đã dừng camera')
        
        # Tạo hình ảnh trống
        blank_photo = self._create_blank_image()
        
        # Cập nhật UI
        self.cam_disp.config(image=blank_photo)
        self.cam_disp.image = blank_photo
        
        # Xóa dữ liệu hình ảnh cũ
        if hasattr(self.cam_disp, 'current'):
            delattr(self.cam_disp, 'current')
        
        # Xóa văn bản nhận diện
        self.cam_txt.config(text='')
        
        # Đảm bảo UI được cập nhật
        self.root.update()

    # Chụp ảnh từ camera
    def on_cam_snapshot(self):
        # Kiểm tra xem camera có đang chạy không
        if not hasattr(self.cam_det, 'running') or not self.cam_det.running:
            messagebox.showwarning('Cảnh báo', 'Camera chưa được bật. Vui lòng bấm "Bắt đầu" trước khi chụp ảnh.')
            return
            
        # Kiểm tra xem có khung hình hiện tại không    
        if not hasattr(self.cam_disp, 'current'):
            messagebox.showinfo('Thông báo', 'Không có khung hình hiện tại để chụp')
            return
                
        d = setup_output_directories()
        fname = f"cam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        p = os.path.join(d, fname)
        cv2.imwrite(p, self.cam_disp.current)
        messagebox.showinfo('Chụp ảnh', 'Đã lưu tại ' + p)

    # Ghi video từ camera
    def on_cam_toggle(self):
        # Kiểm tra xem camera có đang chạy không
        if not self.cam_det.recording:
            # Nếu chưa ghi video, kiểm tra camera trước khi bắt đầu ghi
            if not hasattr(self.cam_det, 'running') or not self.cam_det.running:
                messagebox.showwarning('Cảnh báo', 'Camera chưa được bật. Vui lòng bấm "Bắt đầu" trước khi ghi video.')
                return
                
            d = setup_output_directories()
            fname = f"cam_vid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            p = os.path.join(d, fname)
            self.cam_det.start_record(p, fps=20, size=(640, 480))
            self.cam_det.video_path = p
            self.cam_status.config(text='Đang ghi video')
        else:
            self.cam_det.stop_record()
            self.cam_status.config(text='Đã dừng ghi video')
            # Thêm thông báo về vị trí lưu video
            if hasattr(self.cam_det, 'video_path'):
                messagebox.showinfo('Thành công', f'Video đã được lưu tại:\n{self.cam_det.video_path}')

if __name__ == '__main__': 
    MainApp()