import cv2
import threading
import time

class CameraDetector:
    def __init__(self, model, device, interval=30, slow_delay=0.0):
        self.model = model
        self.device = device
        self.interval = interval
        self.slow_delay = slow_delay
        self.running = False
        self.recording = False
        self.writer = None
        
        # Check if tracker is available and which trackers are supported
        self.has_tracker = hasattr(model, 'track')
        self.available_trackers = self._get_available_trackers()
        
        # Store track history for visualization
        self.track_history = {}
        # Maximum history to keep
        self.max_history = 30
    
    def _get_available_trackers(self):
        """Check which trackers are available in the current YOLO installation"""
        if not self.has_tracker:
            return []
            
        try:
            # Try importing tracker modules to see what's available
            from ultralytics.trackers import BOTSORT, BYTETracker
            trackers = []
            
            try:
                _ = BYTETracker(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
                trackers.append("bytetrack")
            except:
                pass
                
            try:
                _ = BOTSORT(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
                trackers.append("botsort")
            except:
                pass
                
            return trackers
        except ImportError:
            return []

    def start(self, frame_callback=None, label_callback=None, smooth_mode=True):
        """
        Start camera detection
        
        Parameters:
        -----------
        smooth_mode : bool
            If True, uses the 1234.py style smooth tracking on every frame
        """
        if self.running:
            return
        self.running = True
        
        def _run():
            cap = cv2.VideoCapture(0)
            count = 0
            
            # FPS control
            fps_target = 15  # Target FPS for smooth operation
            frame_time = 1.0 / fps_target
            last_time = time.time()
            
            # Determine if we should use tracking
            use_tracking = self.has_tracker and len(self.available_trackers) > 0
            selected_tracker = self.available_trackers[0] if use_tracking else None
            
            while self.running:
                # Time control
                current_time = time.time()
                delta = current_time - last_time
                
                if delta < frame_time:
                    # Wait to maintain consistent frame rate
                    time.sleep(frame_time - delta)
                
                # Capture new timestamp
                last_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame using smooth approach
                if smooth_mode:
                    try:
                        # Use tracking if available (like 1234.py)
                        if use_tracking:
                            try:
                                results = self.model.track(
                                    frame,
                                    device=self.device,
                                    persist=True,
                                    tracker=selected_tracker,
                                    verbose=False
                                )
                            except Exception as e:
                                print(f"Tracking failed: {e}")
                                results = self.model.predict(
                                    frame,
                                    device=self.device,
                                    verbose=False
                                )
                        else:
                            results = self.model.predict(
                                frame,
                                device=self.device,
                                verbose=False
                            )
                            
                        # Extract labels
                        detected_labels = []
                        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
                            for box in results[0].boxes:
                                label_id = int(box.cls[0].item())
                                conf = float(box.conf[0].item())
                                label = self.model.names[label_id]
                                
                                # Check for track ID
                                track_id = None
                                if hasattr(box, 'id') and box.id is not None:
                                    track_id = int(box.id[0].item())
                                    text = f"{label} {conf:.2f} ID:{track_id}"
                                else:
                                    text = f"{label} {conf:.2f}"
                                    
                                detected_labels.append(text)
                        
                        # Get annotated frame
                        if results and len(results) > 0:
                            processed_frame = results[0].plot()
                        else:
                            processed_frame = frame.copy()
                            
                    except Exception as e:
                        print(f"Camera processing error: {e}")
                        processed_frame = frame.copy()
                        detected_labels = []
                        
                else:
                    # Original interval-based detection (existing code)
                    if count == 0 or count % self.interval == 0:
                        try:
                            # Use tracking for smoother results if available
                            if use_tracking:
                                try:
                                    results = self.model.track(
                                        frame,
                                        device=self.device,
                                        persist=True,
                                        tracker=selected_tracker,
                                        verbose=False
                                    )
                                except Exception as e:
                                    print(f"Tracking failed: {e}")
                                    results = self.model.predict(
                                        frame,
                                        device=self.device,
                                        verbose=False
                                    )
                            else:
                                results = self.model.predict(
                                    frame,
                                    device=self.device,
                                    verbose=False
                                )
                        except Exception as e:
                            print(f"Camera detection error: {e}")
                    
                    # Rest of the original processing code...
                    # ... (kept for compatibility)
                    
                    # Use YOLO's built-in plotting for visualization
                    if results:
                        processed_frame = results[0].plot()
                    else:
                        processed_frame = frame
                
                # Recording logic (same for both modes)
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
        self.track_history = {}  # Clear tracking history

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