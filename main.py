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

from image import ImageDetector
from video import VideoDetector
from camera import CameraDetector

# Utils

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
        p = f"{base}_{i}{ext}";
        if not os.path.exists(p): return p
        i += 1

class MainApp:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        try:
            print("Loading YOLO model...")
            self.model = YOLO('runs/detect/train/weights/best.pt')
            
            # Configure model settings for better visualization
            if hasattr(self.model, 'overrides'):
                # Set some visualization preferences
                self.model.overrides['conf'] = 0.25  # Default confidence
                self.model.overrides['line_width'] = None  # Auto line width
                self.model.overrides['box'] = True  # Show boxes
            
            # Check if tracking is available
            has_tracking = hasattr(self.model, 'track')
            available_trackers = []
            
            if has_tracking:
                try:
                    from ultralytics.trackers import BOTSORT, BYTETracker
                    
                    try:
                        _ = BYTETracker(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
                        available_trackers.append("bytetrack")
                    except:
                        pass
                        
                    try:
                        _ = BOTSORT(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
                        available_trackers.append("botsort")
                    except:
                        pass
                except ImportError:
                    pass
                    
            if available_trackers:
                print(f"Available trackers: {', '.join(available_trackers)}")
            else:
                print("No object tracking available. Install ultralytics tracking dependencies for smoother results.")
            
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Model Error", f"Failed to load YOLO model: {e}")
            exit(1)
        
        # Initialize detectors
        self.img_det = ImageDetector(self.model, self.device)
        self.vid_det = VideoDetector(self.model, self.device)
        self.cam_det = CameraDetector(self.model, self.device)
        
        self.root = None
        self.build_ui()

    def build_ui(self):
        r = Tk(); self.root = r
        r.title('Nhận diện biển báo giao thông Việt Nam')
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight(); ww, wh = 800,750
        r.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{(sh-wh)//2}")
        nb = ttk.Notebook(r); nb.pack(expand=True, fill=BOTH, padx=5, pady=5)

        # Image Tab
        it = Frame(nb); nb.add(it, text='Nhận diện ảnh')
        self.img_conf = Scale(it, from_=10,to=95,orient=HORIZONTAL,label='Tin cậy (%)'); self.img_conf.set(50); self.img_conf.pack(fill=X)
        self.img_iou = Scale(it, from_=10,to=95,orient=HORIZONTAL,label='IoU (%)'); self.img_iou.set(50); self.img_iou.pack(fill=X)
        f = Frame(it); f.pack(fill=X)
        Label(f, text='Kích thước:').pack(side=LEFT)
        self.img_size = IntVar(value=640)
        Spinbox(f, from_=320,to=1280,increment=32,textvariable=self.img_size,width=5).pack(side=LEFT)
        cf = Frame(it); cf.pack(fill=X,pady=5)
        Button(cf, text='Tải ảnh',command=self.on_image_upload,width=15).pack(side=LEFT,padx=5)
        Button(cf, text='Nhận diện',command=self.on_image_detect,width=15).pack(side=LEFT,padx=5)
        Button(cf, text='Lưu',command=self.on_image_save,width=15).pack(side=LEFT,padx=5)
        self.img_path = Label(it,text='Chưa tải',fg='blue'); self.img_path.pack(pady=5)
        self.img_prog = ttk.Progressbar(it,orient='horizontal',length=600,mode='determinate')
        self.img_disp = Label(it,bd=2,relief='solid',width=600,height=400); self.img_disp.pack(pady=10)
        self.img_txt = Label(it,font=('Arial',12)); self.img_txt.pack()
        self.img_status = Label(it,fg='blue'); self.img_status.pack(pady=5)

        # Video Tab
        vt = Frame(nb); nb.add(vt, text='Nhận diện video')
        self.vid_conf = Scale(vt, from_=10,to=95,orient=HORIZONTAL,label='Tin cậy (%)'); self.vid_conf.set(50); self.vid_conf.pack(fill=X)
        self.vid_iou = Scale(vt, from_=10,to=95,orient=HORIZONTAL,label='IoU (%)'); self.vid_iou.set(50); self.vid_iou.pack(fill=X)
        f2 = Frame(vt); f2.pack(fill=X)
        Label(f2,text='Kích thước:').pack(side=LEFT)
        self.vid_size = IntVar(value=640)
        Spinbox(f2,from_=320,to=1280,increment=32,textvariable=self.vid_size,width=5).pack(side=LEFT)
        cv2b = Frame(vt); cv2b.pack(fill=X,pady=5)
        Button(cv2b,text='Tải video',command=self.on_video_upload,width=15).pack(side=LEFT,padx=5)
        Button(cv2b,text='Xử lý',command=self.on_video_process,width=15).pack(side=LEFT,padx=5)
        Button(cv2b,text='Lưu',command=self.on_video_save,width=15).pack(side=LEFT,padx=5)
        self.vid_path=Label(vt,text='Chưa tải',fg='blue');self.vid_path.pack(pady=5)
        self.vid_prog=ttk.Progressbar(vt,orient='horizontal',length=600,mode='determinate')
        self.vid_disp=Label(vt,bd=2,relief='solid',width=600,height=400);self.vid_disp.pack(pady=10)
        self.vid_txt=Label(vt,font=('Arial',12));self.vid_txt.pack()
        self.vid_status=Label(vt,fg='blue');self.vid_status.pack(pady=5)

        # Camera Tab
        ct=Frame(nb);nb.add(ct,text='Camera trực tiếp')
        self.cam_conf=Scale(ct,from_=10,to=95,orient=HORIZONTAL,label='Tin cậy (%)');self.cam_conf.set(50);self.cam_conf.pack(fill=X)
        self.cam_iou=Scale(ct,from_=10,to=95,orient=HORIZONTAL,label='IoU (%)');self.cam_iou.set(50);self.cam_iou.pack(fill=X)
        f3=Frame(ct);f3.pack(fill=X)
        Label(f3,text='Kích thước:').pack(side=LEFT)
        self.cam_size=IntVar(value=640)
        Spinbox(f3,from_=320,to=1280,increment=32,textvariable=self.cam_size,width=5).pack(side=LEFT)
        cb=Frame(ct);cb.pack(fill=X,pady=5)
        Button(cb,text='Bắt đầu',command=self.on_cam_start,width=15).pack(side=LEFT,padx=5)
        Button(cb,text='Dừng',command=self.on_cam_stop,width=15).pack(side=LEFT,padx=5)
        Button(cb,text='Chụp',command=self.on_cam_snapshot,width=15).pack(side=LEFT,padx=5)
        Button(cb,text='Ghi/Dừng',command=self.on_cam_toggle,width=15).pack(side=LEFT,padx=5)
        self.cam_status=Label(ct,text='Chưa khởi động',fg='blue');self.cam_status.pack(pady=5)
        self.cam_disp=Label(ct,bd=2,relief='solid',width=600,height=400);self.cam_disp.pack(pady=10)
        self.cam_txt=Label(ct,font=('Arial',12));self.cam_txt.pack()

        r.mainloop()

    # --- Callbacks ---
    def on_image_upload(self):
        path = filedialog.askopenfilename(filetypes=[('Images','*.jpg *.jpeg *.png')])
        if not path: return
        self.img_path.file_path = path
        self.img_path.config(text=os.path.basename(path))
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((600,400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.img_disp.config(image=photo); self.img_disp.image=photo
        self.img_txt.config(text=''); self.img_status.config(text='')

    def on_image_detect(self):
        if not hasattr(self.img_path, 'file_path'):
            self.img_status.config(text='Vui lòng tải ảnh'); return
        frame, labels = self.img_det.detect(
            self.img_path.file_path,
            self.img_conf.get()/100,
            self.img_iou.get()/100,
            self.img_size.get()
        )
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((600,400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.img_disp.config(image=photo); self.img_disp.image=photo
        counts = {}
        for t in labels:
            name = t.split()[0]
            counts[name] = counts.get(name,0)+1
        text = ', '.join(f"{k} ({v})" for k,v in counts.items())
        self.img_txt.config(text='Nhận diện: '+(text or 'Không có'))
        self.img_disp.processed=frame
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

    def on_video_upload(self):
        path = filedialog.askopenfilename(filetypes=[('Videos','*.mp4 *.avi')])
        if not path: return
        self.vid_path.file_path = path
        self.vid_path.config(text=os.path.basename(path))
        cap=cv2.VideoCapture(path);ret,frm=cap.read();cap.release()
        if ret:
            frm=cv2.cvtColor(frm,cv2.COLOR_BGR2RGB)
            img=Image.fromarray(frm).resize((600,400),Image.Resampling.LANCZOS)
            ph=ImageTk.PhotoImage(img)
            self.vid_disp.config(image=ph);self.vid_disp.image=ph
            self.vid_txt.config(text='');self.vid_status.config(text='')

    def on_video_process(self):
        if not hasattr(self.vid_path,'file_path'):
            self.vid_status.config(text='Vui lòng tải video'); 
            return
            
        self.vid_prog.pack(fill=X, padx=5, pady=5)
        self.vid_prog['value']=0
        self.vid_status.config(text='Đang xử lý...')
        
        # Setup output path
        d = setup_output_directories()
        fname = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(d, fname)
        
        def prog(i,total):
            self.vid_prog['value']=i*100/total
            self.vid_status.config(text=f"Đang xử lý: {i}/{total} frames ({int(i*100/total)}%)")
            
        def frame_cb(frm, labels):
            # Convert to RGB for display
            img=cv2.cvtColor(frm,cv2.COLOR_BGR2RGB)
            img=Image.fromarray(img).resize((600,400), Image.Resampling.LANCZOS)
            ph=ImageTk.PhotoImage(img)
            self.vid_disp.config(image=ph)
            self.vid_disp.image=ph
            
            # Update detection text
            counts = {} 
            for t in labels:
                # Extract class name (first part of the label)
                name = t.split()[0]
                counts[name] = counts.get(name,0)+1
            
            text = ', '.join(f"{k} ({v})" for k,v in counts.items())
            self.vid_txt.config(text='Nhận diện: '+(text or 'Không có'))
            
        def finish(p):
            self.vid_prog['value']=100
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

    def on_cam_start(self):
        self.cam_status.config(text='Camera đang chạy')
        
        def frame_cb(frm):
            img = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((600,400), Image.Resampling.LANCZOS)
            ph = ImageTk.PhotoImage(img)
            self.cam_disp.config(image=ph)
            self.cam_disp.image = ph
            self.cam_disp.current = frm
            
        def label_cb(labels):
            counts = {} 
            for t in labels:
                name = t.split()[0]
                counts[name] = counts.get(name,0)+1
                
            text = ', '.join(f"{k} ({v})" for k,v in counts.items())
            self.cam_txt.config(text='Nhận diện: '+(text or 'Không có'))
            
        self.cam_det.start(frame_cb, label_cb)

    def on_cam_stop(self):
        self.cam_det.stop()
        self.cam_status.config(text='Đã dừng camera')

    def on_cam_snapshot(self):
        if not hasattr(self.cam_disp,'current'):
            messagebox.showinfo('Thông báo','Camera chưa chạy')
            return
            
        d = setup_output_directories()
        fname = f"cam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        p = os.path.join(d, fname)
        cv2.imwrite(p, self.cam_disp.current)
        messagebox.showinfo('Chụp ảnh','Đã lưu tại ' + p)

    def on_cam_toggle(self):
        if not self.cam_det.recording:
            d = setup_output_directories()
            fname = f"cam_vid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            p = os.path.join(d, fname)
            self.cam_det.start_record(
                p, 
                fps=20, 
                size=(640, 480)  # Adjust size if needed
            )
            self.cam_status.config(text='Đang ghi video')
        else:
            self.cam_det.stop_record()
            self.cam_status.config(text='Đã dừng ghi video')

if __name__ == '__main__': MainApp()