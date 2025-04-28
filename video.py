import cv2
import threading
import time
import os

class VideoDetector:
    def __init__(self, model, device, interval=5, slow_delay=0.0):
        self.model = model
        self.device = device
        self.interval = interval
        self.slow_delay = slow_delay
        self.output_path = None
        
        # Check if tracker is available and which trackers are supported
        self.has_tracker = hasattr(model, 'track')
        self.available_trackers = self._get_available_trackers()
        
    def _get_available_trackers(self):
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

    def process(self, input_path, output_path, conf_thresh, iou_thresh, img_size,
                progress_callback=None, frame_callback=None, finish_callback=None):
        """Process video using smooth frame-by-frame tracking (like 1234.py)"""
        
        # First phase: analyze video properties
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 17
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Create output writer
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        self.output_path = output_path

        def _run():
            # Track processing speed
            start_time = time.time()
            frames_processed = 0
            processing_fps = 0
            
            # Open video for processing
            cap = cv2.VideoCapture(input_path)
            frame_count = 0
            
            # Determine if we should use tracking
            use_tracking = self.has_tracker and len(self.available_trackers) > 0
            selected_tracker = self.available_trackers[0] if use_tracking else None
            
            # For controlling speed
            target_delay = 1.0 / (fps * 2.0)  # Half speed playback by default
            
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                try:
                    # Use tracking for ALL frames (no skipping)
                    if use_tracking:
                        try:
                            results = self.model.track(
                                frame, 
                                device=self.device,
                                persist=True,
                                tracker=selected_tracker,
                                classes=None,
                                verbose=False,
                                imgsz=img_size, 
                                conf=conf_thresh, 
                                iou=iou_thresh
                            )
                        except Exception as e:
                            print(f"Tracking failed, falling back to prediction: {e}")
                            results = self.model.predict(
                                frame,
                                device=self.device,
                                classes=None,
                                verbose=False,
                                imgsz=img_size,
                                conf=conf_thresh,
                                iou=iou_thresh
                            )
                    else:
                        results = self.model.predict(
                            frame,
                            device=self.device,
                            classes=None,
                            verbose=False,
                            imgsz=img_size,
                            conf=conf_thresh,
                            iou=iou_thresh
                        )
                        
                    # Extract labels for callback
                    detected_labels = []
                    if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
                        for box in results[0].boxes:
                            label_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            label = self.model.names[label_id]
                            
                            # Check for tracking ID
                            track_id = None
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = int(box.id[0].item())
                                text = f"{label} {conf:.2f} ID:{track_id}"
                            else:
                                text = f"{label} {conf:.2f}"
                                
                            detected_labels.append(text)
                    
                    # Get annotated frame using YOLO's built-in visualization
                    if results and len(results) > 0:
                        annotated_frame = results[0].plot()
                    else:
                        annotated_frame = frame.copy()
                        
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    annotated_frame = frame.copy()
                    detected_labels = []
                
                # Write frame to output video
                writer.write(annotated_frame)
                
                # Update processing stats
                frames_processed += 1
                if frames_processed % 10 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    if elapsed > 0:
                        processing_fps = frames_processed / elapsed
                
                # Update progress
                if progress_callback and frame_count % 5 == 0:
                    progress_callback(frame_count, total_frames)
                
                # Update display
                if frame_callback and frame_count % 2 == 0:  # Update UI every other frame to avoid lag
                    frame_callback(annotated_frame, detected_labels)
                
                # Control processing speed (optional)
                time.sleep(target_delay)
                
                frame_count += 1
            
            # Cleanup
            cap.release()
            writer.release()
            
            # Calculate final stats
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"Processed {frame_count} frames in {total_time:.2f}s (avg {avg_fps:.2f} FPS)")
            
            if finish_callback:
                finish_callback(output_path)

        # Start processing in a separate thread
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread
        
    def play_processed_video(self, video_path, frame_callback=None, speed=0.5):
        """Play a processed video at controlled speed"""
        
        def _play():
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / (fps * speed))  # Convert to ms
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_callback:
                    frame_callback(frame, [])
                
                # Wait to maintain proper playback speed
                key = cv2.waitKey(delay)
                if key == 27:  # ESC
                    break
            
            cap.release()
            
        thread = threading.Thread(target=_play, daemon=True)
        thread.start()
        return thread