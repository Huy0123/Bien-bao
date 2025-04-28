import os
import threading
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

# Utils: Thiết lập thư mục output và tạo tên file
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
        print(f"Using device: {self.device}")
        
        # Thêm bảng ánh xạ nhãn sang tiếng Việt đầy đủ
        self.label_map = {
            "Ben_xe_buyt": "Bến xe buýt",
            "Cam_con_lai": "Biển báo cấm", 
            "Cam_do": "Cấm đỗ",
            "Cam_dung_va_do": "Cấm dừng và đỗ",
            "Cam_nguoc_chieu": "Cấm ngược chiều",
            "Cam_quay_dau_xe": "Cấm quay đầu xe",
            "Cam_re_phai": "Cấm rẽ phải",
            "Cam_re_trai": "Cấm rẽ trái",
            "Chi_dan": "Biển báo chỉ dẫn",
            "Chieu_cao_toi_da": "Chiều cao tối đa",
            "Duong_giao_nhau": "Đường giao nhau",
            "Giao_voi_duong_khong_uu_tien": "Giao với đường không ưu tiên",
            "Giao_voi_duong_uu_tien": "Giao với đường ưu tiên",
            "Het_tat_ca_lenh_cam": "Hết tất cả lệnh cấm",
            "Hieu_lenh": "Biển báo hiệu lệnh",
            "Nguoi_di_bo_cat_ngang": "Người đi bộ cắt ngang",
            "Nguy_hiem": "Nguy hiểm",
            "Re_phai": "Rẽ phải",
            "Re_trai": "Rẽ trái", 
            "Toc_do_toi_da": "Tốc độ tối đa",
            "Trong_tai_toi_da": "Trọng tải tối đa",
            "Vong_xuyen": "Vòng xuyến"
        }
        
        # Thêm các biến thể không có dấu gạch dưới
        variants = {}
        for key, value in self.label_map.items():
            # Thêm biến thể không có gạch dưới
            variants[key.replace("_", "")] = value
            variants[key.replace("_", " ")] = value
            # Thêm biến thể chữ thường và chữ hoa
            variants[key.lower()] = value 
            variants[key.upper()] = value
        
        # Cập nhật bảng ánh xạ với các biến thể
        self.label_map.update(variants)
        
        try:
            print("Loading YOLO model...")
            self.model = YOLO('runs/detect/train2/weights/best.pt')
            self._configure_model()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Model Error", f"Failed to load YOLO model: {e}")
            exit(1)
        
        # Khởi tạo các detector
        self.img_det = ImageDetector(self.model, self.device)
        self.vid_det = VideoDetector(self.model, self.device)
        self.cam_det = CameraDetector(self.model, self.device)
        
        self.root = None
        self.build_ui()

    def _configure_model(self):
        """Cấu hình mô hình và kiểm tra các tính năng"""
        # Configure model settings for better visualization
        if hasattr(self.model, 'overrides'):
            self.model.overrides['conf'] = 0.5
            self.model.overrides['line_width'] = None
            self.model.overrides['box'] = True
        
        # Kiểm tra xem mô hình có hỗ trợ track không
        has_tracking = hasattr(self.model, 'track')
        available_trackers = []
        
        if has_tracking:
            try:
                from ultralytics.trackers import BOTSORT, BYTETracker
                
                try:
                    _ = BYTETracker(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
                    available_trackers.append("bytetrack")
                except: pass
                    
                try:
                    _ = BOTSORT(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
                    available_trackers.append("botsort")
                except: pass
            except ImportError: pass
                
        if available_trackers:
            print(f"Available trackers: {', '.join(available_trackers)}")
        else:
            print("No object tracking available. Install ultralytics tracking dependencies for smoother results.")

    def build_ui(self):
        """Xây dựng giao diện người dùng chính"""
        r = Tk(); self.root = r
        r.title('Nhận diện biển báo giao thông Việt Nam')
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        ww, wh = 800, 750
        r.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{(sh-wh)//2}")
        
        # Tạo notebook chứa các tab
        nb = ttk.Notebook(r)
        nb.pack(expand=True, fill=BOTH, padx=5, pady=5)
        nb.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Tạo các tab và nút điều khiển của chúng
        self._build_image_tab(nb)
        self._build_video_tab(nb)
        self._build_camera_tab(nb)
        
        r.mainloop()

    def _build_image_tab(self, notebook):
        """Xây dựng tab nhận diện ảnh"""
        tab = self.create_detection_tab(notebook, 'Nhận diện ảnh', 'img')
        buttons = [
            ('Tải ảnh', self.on_image_upload),
            ('Nhận diện', self.on_image_detect),
            ('Lưu', self.on_image_save)
        ]
        self._create_buttons(self.img_button_frame, buttons)

    def _build_video_tab(self, notebook):
        """Xây dựng tab nhận diện video"""
        tab = self.create_detection_tab(notebook, 'Nhận diện video', 'vid')
        buttons = [
            ('Tải video', self.on_video_upload),
            ('Xử lý', self.on_video_process),
            ('Lưu', self.on_video_save)
        ]
        
        self._create_buttons(self.vid_button_frame, buttons)

    def _build_camera_tab(self, notebook):
        """Xây dựng tab camera trực tiếp"""
        tab = self.create_detection_tab(notebook, 'Camera trực tiếp', 'cam')
        buttons = [
            ('Bắt đầu', self.on_cam_start),
            ('Dừng', self.on_cam_stop),
            ('Chụp', self.on_cam_snapshot),
            ('Ghi/Dừng', self.on_cam_toggle)
        ]
        self._create_buttons(self.cam_button_frame, buttons)

    def _create_buttons(self, frame, button_configs):
        """Tạo các nút từ cấu hình"""
        for text, cmd in button_configs:
            Button(frame, text=text, command=cmd, width=15).pack(side=LEFT, padx=5)


    def create_detection_controls(self, tab, prefix):
        """Tạo các điều khiển chung cho detection tab"""
        # Confidence slider
        setattr(self, f"{prefix}_conf", Scale(tab, from_=10, to=95, orient=HORIZONTAL, label='Tin cậy (%)'))
        getattr(self, f"{prefix}_conf").set(50)
        getattr(self, f"{prefix}_conf").pack(fill=X)
        
        # IoU slider
        setattr(self, f"{prefix}_iou", Scale(tab, from_=10, to=95, orient=HORIZONTAL, label='IoU (%)'))
        getattr(self, f"{prefix}_iou").set(50)
        getattr(self, f"{prefix}_iou").pack(fill=X)
        
        # Size control
        f = Frame(tab); f.pack(fill=X)
        Label(f, text='Kích thước:').pack(side=LEFT)
        setattr(self, f"{prefix}_size", IntVar(value=640))
        Spinbox(f, from_=320, to=1280, increment=32, 
                textvariable=getattr(self, f"{prefix}_size"), width=5).pack(side=LEFT)
        
        # Tạo frame nút điều khiển ở đây
        setattr(self, f"{prefix}_button_frame", Frame(tab))
        getattr(self, f"{prefix}_button_frame").pack(fill=X, pady=5)
        
        # Path and progress indicators (sau khi tạo frame nút)
        setattr(self, f"{prefix}_path", Label(tab, text='Chưa tải', fg='blue'))
        getattr(self, f"{prefix}_path").pack(pady=5)
        setattr(self, f"{prefix}_prog", ttk.Progressbar(tab, orient='horizontal', length=600, mode='determinate'))

    def create_detection_tab(self, notebook, title, prefix):
        """Tạo cấu trúc cơ bản cho tab nhận diện"""
        tab = Frame(notebook)
        notebook.add(tab, text=title)
        
        # Tạo các điều khiển chung (bao gồm cả frame nút)
        self.create_detection_controls(tab, prefix)
        
        # Khung hiển thị
        setattr(self, f"{prefix}_disp", Label(tab, bd=2, relief='solid', width=600, height=400))
        getattr(self, f"{prefix}_disp").pack(pady=10)
        
        # Text labels
        setattr(self, f"{prefix}_txt", Label(tab, font=('Arial', 12)))
        getattr(self, f"{prefix}_txt").pack()
        
        setattr(self, f"{prefix}_status", Label(tab, fg='blue'))
        getattr(self, f"{prefix}_status").pack(pady=5)
        
        return tab

    def _create_blank_image(self):
        """Tạo hình ảnh trống cho hiển thị"""
        blank = Image.new('RGB', (600, 400), color='white')
        return ImageTk.PhotoImage(blank)

    def on_tab_changed(self, event):
        """Xử lý sự kiện khi người dùng chuyển tab"""
        tab_id = event.widget.select()
        tab_name = event.widget.tab(tab_id, "text")
        
        # Reset trạng thái dựa vào tab được chọn
        if tab_name == 'Nhận diện ảnh':
            self._reset_tab('img')
        elif tab_name == 'Nhận diện video':
            self._reset_tab('vid')
        elif tab_name == 'Camera trực tiếp':
            self._reset_camera_tab()

    def _reset_tab(self, prefix):
        """Reset trạng thái của tab"""
        blank_photo = self._create_blank_image()
        disp = getattr(self, f"{prefix}_disp")
        disp.config(image=blank_photo)
        disp.image = blank_photo
        
        # Reset các nhãn và trạng thái
        getattr(self, f"{prefix}_txt").config(text='')
        getattr(self, f"{prefix}_status").config(text='')
        getattr(self, f"{prefix}_path").config(text='Chưa tải')
        
        # Xóa dữ liệu liên kết
        path_attr = getattr(self, f"{prefix}_path")
        if hasattr(path_attr, 'file_path'):
            delattr(path_attr, 'file_path')
        
        # Xóa kết quả xử lý nếu có
        if prefix == 'img' and hasattr(disp, 'processed'):
            delattr(disp, 'processed')
        elif prefix == 'vid' and hasattr(self.vid_det, 'output_path'):
            delattr(self.vid_det, 'output_path')

    def _reset_camera_tab(self):
        """Reset tab camera với xử lý đặc biệt"""
        # Dừng camera nếu đang chạy
        if hasattr(self.cam_det, 'running') and self.cam_det.running:
            self.on_cam_stop()
        else:
            # Nếu camera đã dừng, vẫn cần reset UI
            blank_photo = self._create_blank_image()
            self.cam_disp.config(image=blank_photo)
            self.cam_disp.image = blank_photo
            self.cam_txt.config(text='')
            self.cam_status.config(text='Chưa khởi động')

    def display_image(self, image, display_widget, text_widget, labels=None):
        """Hiển thị hình ảnh và cập nhật thông tin nhận diện"""
        # Xử lý hình ảnh
        if isinstance(image, np.ndarray):
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:
            img = image
            
        img = img.resize((600, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        display_widget.config(image=photo)
        display_widget.image = photo
        
        # Lưu frame hiện tại vào display_widget để sử dụng sau này
        if isinstance(image, np.ndarray):
            display_widget.current = image
        
        # Cập nhật thông tin nhận diện nếu có
        if labels is not None:
            self._update_detection_labels(labels, text_widget)

    def _update_detection_labels(self, labels, text_widget):
        """Cập nhật nhãn hiển thị kết quả nhận diện với tên tiếng Việt đầy đủ"""
        counts = {}
        for t in labels:
            # Tách riêng tên nhãn và độ tin cậy
            parts = t.split()
            
            # Nếu có độ tin cậy, tách nó ra
            confidence_part = parts[-1] if parts and parts[-1][0].isdigit() else ""
            name_parts = parts[:-1] if confidence_part else parts
            full_name = " ".join(name_parts)
            
            
            # Tìm kiếm trong bảng ánh xạ
            vietnamese_name = self.label_map.get(full_name, None)
            
            # Nếu không tìm thấy, thử các biến thể khác
            if not vietnamese_name:
                # Thử tìm kiếm không phân biệt chữ hoa/thường
                for key, value in self.label_map.items():
                    if key.lower() == full_name.lower():
                        vietnamese_name = value
                        break
            
            # Nếu vẫn không tìm thấy, giữ nguyên tên gốc
            if not vietnamese_name:
                vietnamese_name = full_name
            
            # Cập nhật số lượng
            counts[vietnamese_name] = counts.get(vietnamese_name, 0) + 1
        
        text = ', '.join(f"{k} ({v})" for k, v in counts.items())
        text_widget.config(text='Nhận diện: ' + (text or 'Không có'))

    # IMAGE TAB FUNCTIONS
    def on_image_upload(self):
        path = filedialog.askopenfilename(filetypes=[('Images','*.jpg *.jpeg *.png')])
        if not path: return
        self.img_path.file_path = path
        self.img_path.config(text=os.path.basename(path))
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.display_image(img, self.img_disp, self.img_txt)

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

    # VIDEO TAB FUNCTIONS
    def on_video_upload(self):
        path = filedialog.askopenfilename(filetypes=[('Videos','*.mp4 *.avi')])
        if not path: return
        self.vid_path.file_path = path
        self.vid_path.config(text=os.path.basename(path))
        
        # Show first frame
        cap = cv2.VideoCapture(path)
        ret, frm = cap.read()
        cap.release()
        if ret:
            self.display_image(frm, self.vid_disp, self.vid_txt)

    def on_video_process(self):
        if not hasattr(self.vid_path,'file_path'):
            self.vid_status.config(text='Vui lòng tải video')
            return
            
        # Setup UI
        self.vid_prog.pack(fill=X, padx=5, pady=5)
        self.vid_prog['value'] = 0
        self.vid_status.config(text='Đang xử lý...')
        
        # Setup output path
        d = setup_output_directories()
        fname = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(d, fname)
        
        # Setup callbacks
        def prog(i, total):
            self.vid_prog['value'] = i*100/total
            self.vid_status.config(text=f"Đang xử lý: {i}/{total} frames ({int(i*100/total)}%)")
            
        def frame_cb(frm, labels):
            self.display_image(frm, self.vid_disp, self.vid_txt, labels)
            
        def finish(p):
            self.vid_prog['value'] = 100
            self.vid_prog.pack_forget()
            self.vid_status.config(text=f'Xong: {os.path.basename(p)}')
            self.vid_det.output_path = p  # Store for saving later
            
        # Start video processing
        self.vid_det.process(
            self.vid_path.file_path, 
            output_path,
            self.vid_conf.get()/100, 
            self.vid_iou.get()/100, 
            self.vid_size.get(),
            prog, frame_cb, finish
        )

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
            
        self.cam_det.start(frame_cb, label_cb)

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