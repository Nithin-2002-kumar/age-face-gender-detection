import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import tkinter as tk
from tkinter import Label, Button, Toplevel, filedialog, ttk, messagebox
from datetime import datetime
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import torch
import threading
from collections import deque
import time
import gc
from typing import List, Dict, Optional, Tuple
import queue


class AdvancedDetector:
    def __init__(self):
        # Initialize models
        self.object_model: Optional[YOLO] = None
        self.face_model: Optional[YOLO] = None
        self.age_model = None
        self.gender_model = None
        self.face_cascade = None
        self.models_loaded = False

        # Hardware optimization flags
        self.use_cuda = torch.cuda.is_available()
        self.use_half_precision = self.use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'

        # Model paths (should be configurable)
        self.model_paths = {
            'object': 'E:\\Vamshi\\vamshi\\yolov8n.pt',
            'face': 'E:\\Vamshi\\vamshi\\face_yolov8m.pt',
            'age': 'E:\\Vamshi\\vamshi\\age_model.h5',
            'gender': 'E:\\Vamshi\\vamshi\\gender_model.h5'
        }

        # Detection parameters
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                         '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        self.current_mode = "object"

        # Performance tracking
        self.inference_times = deque(maxlen=30)
        self.last_inference_time = 0

        # Load models
        self._load_models()

    def _load_models(self) -> None:
        """Load models with optimizations and fallbacks"""
        try:
            # Load object detection model (fastest first)
            self.object_model = YOLO(self.model_paths['object']).to(self.device)
            if self.use_half_precision:
                self.object_model = self.object_model.half()

            # Load face detection model
            try:
                self.face_model = YOLO(self.model_paths['face']).to(self.device)
                if self.use_half_precision:
                    self.face_model = self.face_model.half()
            except Exception as e:
                print(f"Using object model for face detection: {e}")
                self.face_model = self.object_model

            # Load age/gender models
            self.age_model = load_model(self.model_paths['age'], compile=False)
            self.gender_model = load_model(self.model_paths['gender'], compile=False)

            # Haar cascade fallback
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            ) if cv2.data.haarcascades else None

            # Set models to eval mode
            for model in [self.object_model, self.face_model]:
                if model is not None:
                    model.eval()

            torch.set_grad_enabled(False)
            if self.use_cuda:
                torch.backends.cudnn.benchmark = True

            self.models_loaded = True
            print("Models loaded successfully")

        except Exception as e:
            print(f"Model loading failed: {e}")
            self.models_loaded = False

    def set_mode(self, mode: str) -> bool:
        """Set detection mode (object/face)"""
        if mode in ["object", "face"] and self.models_loaded:
            self.current_mode = mode
            return True
        return False

    def predict_frame(self, frame: np.ndarray) -> List[Dict]:
        """Main detection method with performance optimizations"""
        if not self.models_loaded or frame is None or frame.size == 0:
            return []

        start_time = time.perf_counter()

        try:
            # Convert frame to RGB and normalize if needed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.current_mode == "object":
                detections = self._detect_objects(frame_rgb)
            else:
                detections = self._detect_faces(frame_rgb)

        except Exception as e:
            print(f"Detection error: {e}")
            return []

        # Track performance
        self.last_inference_time = time.perf_counter() - start_time
        self.inference_times.append(self.last_inference_time)

        return detections

    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Optimized object detection"""
        with torch.no_grad():
            results = self.object_model(frame, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes.cpu() if self.use_cuda else result.boxes

            for box in boxes:
                detections.append({
                    'class_name': result.names[int(box.cls.item())],
                    'confidence': box.conf.item(),
                    'bbox': box.xyxy[0].tolist(),
                    'type': 'object'
                })

        return detections

    def _detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Optimized face detection with age/gender prediction"""
        detections = []
        model = self.face_model if self.face_model else self.object_model

        # Primary detection with YOLO
        with torch.no_grad():
            results = model(frame, verbose=False)

        for result in results:
            boxes = result.boxes.cpu() if self.use_cuda else result.boxes

            for box in boxes:
                # Skip if using object model and not a person
                if model == self.object_model and result.names[int(box.cls.item())] != 'person':
                    continue

                bbox = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, bbox)

                # Boundary checks
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                face_img = frame[y1:y2, x1:x2]

                # Predict age/gender
                age, gender, face_conf = self._predict_age_gender(face_img)
                combined_conf = (box.conf.item() + face_conf) / 2

                detections.append({
                    'class_name': f"{age} {gender}",
                    'confidence': combined_conf,
                    'bbox': bbox,
                    'type': 'face'
                })

        # Fallback to Haar if no detections
        if not detections and self.face_cascade:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                age, gender, conf = self._predict_age_gender(face_img)

                detections.append({
                    'class_name': f"{age} {gender}",
                    'confidence': conf,
                    'bbox': [x, y, x + w, y + h],
                    'type': 'face'
                })

        return detections

    def _predict_age_gender(self, face_img: np.ndarray) -> Tuple[str, str, float]:
        """Optimized age/gender prediction"""
        if face_img.size == 0 or min(face_img.shape[:2]) < 20:
            return '(25-32)', 'Male', 0.7

        try:
            # Convert and resize
            if face_img.shape[2] == 4:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)
            elif face_img.shape[2] == 1:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)

            face_img = cv2.resize(face_img, (64, 64))
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            # Batch predictions
            age_pred = self.age_model.predict(face_img, verbose=0)
            gender_pred = self.gender_model.predict(face_img, verbose=0)

            age = self.age_list[np.argmax(age_pred)]
            gender = self.gender_list[int(gender_pred[0][0] > 0.5)]
            confidence = (age_pred.max() + gender_pred.max()) / 2

            return age, gender, confidence

        except Exception as e:
            print(f"Age/gender error: {e}")
            return '(25-32)', 'Male', 0.7

    def get_performance_stats(self) -> Dict:
        """Get current performance metrics"""
        avg_time = np.mean(self.inference_times) if self.inference_times else 0
        fps = 1 / avg_time if avg_time > 0 else 0

        return {
            'inference_time': self.last_inference_time,
            'avg_inference': avg_time,
            'fps': fps,
            'mode': self.current_mode,
            'device': self.device
        }


class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimized Face & Object Detection")
        self.root.geometry("1200x800")
        self._setup_dpi()

        # Performance tracking
        self.frame_queue = queue.Queue(maxsize=1)
        self.detection_queue = queue.Queue(maxsize=1)
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

        # Initialize detector
        self.detector = AdvancedDetector()
        if not self.detector.models_loaded:
            messagebox.showerror("Error", "Failed to load models")
            self.root.destroy()
            return

        # Camera settings
        self.cap = self._init_camera()
        self.is_capturing = False
        self.stop_event = threading.Event()

        # Detection settings
        self.current_detections = []
        self.save_path = os.path.join(os.path.expanduser("~"), "detection_captures")
        os.makedirs(self.save_path, exist_ok=True)

        # UI setup
        self._create_ui()
        self.stop_camera()

    def _setup_dpi(self):
        """Configure for high DPI displays"""
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass

    def _init_camera(self) -> cv2.VideoCapture:
        """Initialize camera with optimal settings"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Camera initialization failed")
            self.root.destroy()
            return None

        # Set optimal parameters
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        return cap

    def _create_ui(self):
        """Create the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        # Mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="Detection Mode")
        mode_frame.pack(side=tk.LEFT, padx=5)

        self.mode_var = tk.StringVar(value="object")
        ttk.Radiobutton(mode_frame, text="Objects", variable=self.mode_var,
                        value="object", command=self._change_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Faces", variable=self.mode_var,
                        value="face", command=self._change_mode).pack(side=tk.LEFT)

        # Confidence threshold
        conf_frame = ttk.LabelFrame(control_frame, text="Confidence")
        conf_frame.pack(side=tk.LEFT, padx=5)

        self.conf_thresh = tk.DoubleVar(value=0.5)
        ttk.Scale(conf_frame, from_=0.1, to=0.9, variable=self.conf_thresh,
                  orient=tk.HORIZONTAL, length=100).pack()

        # Buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT, padx=5)

        self.capture_btn = ttk.Button(btn_frame, text="Capture",
                                      command=self.capture_image, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(btn_frame, text="Start", command=self.start_camera).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Stop", command=self.stop_camera).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Gallery", command=self._open_gallery).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Stats", command=self.show_stats).pack(side=tk.LEFT, padx=2)

        # Info display
        info_frame = ttk.LabelFrame(main_frame, text="Detection Info")
        info_frame.pack(fill=tk.X, pady=5)

        self.detection_var = tk.StringVar(value="No detection")
        self.confidence_var = tk.StringVar(value="0%")
        self.fps_var = tk.StringVar(value="0 FPS")

        ttk.Label(info_frame, text="Detection:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(info_frame, textvariable=self.detection_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(info_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(info_frame, textvariable=self.confidence_var).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(info_frame, text="FPS:").grid(row=0, column=2, sticky=tk.W)
        ttk.Label(info_frame, textvariable=self.fps_var).grid(row=0, column=3, sticky=tk.W)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var,
                  relief=tk.SUNKEN).pack(fill=tk.X, pady=5)

    def _change_mode(self):
        """Handle mode change"""
        if self.detector.set_mode(self.mode_var.get()):
            self.status_var.set(f"Mode: {self.mode_var.get()}")
            self.current_detections = []
        else:
            self.status_var.set("Mode change failed")

    def start_camera(self):
        """Start camera capture and processing"""
        if not self.is_capturing and self.cap is not None:
            self.is_capturing = True
            self.capture_btn.config(state=tk.NORMAL)
            self.stop_event.clear()

            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._process_frames,
                daemon=True
            )
            self.processing_thread.start()

            # Start UI update
            self._update_ui()

    def stop_camera(self):
        """Stop camera capture"""
        self.is_capturing = False
        self.stop_event.set()
        self.capture_btn.config(state=tk.DISABLED)

        # Clear display
        blank_img = Image.new('RGB', (100, 100), (0, 0, 0))
        blank_tk = ImageTk.PhotoImage(blank_img)
        self.image_label.imgtk = blank_tk
        self.image_label.config(image=blank_tk)

        self.detection_var.set("No detection")
        self.confidence_var.set("0%")
        self.fps_var.set("0 FPS")

    def _process_frames(self):
        """Frame processing thread"""
        while not self.stop_event.is_set() and self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, lambda: self.status_var.set("Camera error"))
                break

            try:
                # Put frame in queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put_nowait(frame.copy())

                # Process frame if queue not full
                if not self.detection_queue.full():
                    detections = self.detector.predict_frame(frame)
                    filtered = [d for d in detections if d['confidence'] >= self.conf_thresh.get()]
                    self.detection_queue.put_nowait((frame.copy(), filtered))

            except queue.Full:
                pass
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))

    def _update_ui(self):
        """Update UI with latest frame and detections"""
        if not self.is_capturing:
            return

        start_time = time.perf_counter()

        # Get latest detections if available
        try:
            frame, detections = self.detection_queue.get_nowait()
            self.current_detections = detections
        except queue.Empty:
            frame = None

        # Get latest frame if no new detections
        if frame is None:
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                frame = None

        # Update display if we have a frame
        if frame is not None:
            display_frame = frame.copy()

            # Draw detections
            if self.current_detections:
                best = max(self.current_detections, key=lambda x: x['confidence'])
                self.detection_var.set(best['class_name'])
                self.confidence_var.set(f"{best['confidence'] * 100:.1f}%")

                for det in self.current_detections:
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    color = (0, 255, 0) if det['type'] == 'face' else (255, 0, 0)
                    thickness = 2 if det == best else 1

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                    label = f"{det['class_name']} {det['confidence'] * 100:.1f}%"
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            else:
                self.detection_var.set("No detection")
                self.confidence_var.set("0%")

            # Convert to PIL Image and display
            img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

            # Maintain aspect ratio
            window_width = self.image_label.winfo_width()
            window_height = self.image_label.winfo_height()

            if window_width > 1 and window_height > 1:
                img.thumbnail((window_width, window_height), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.config(image=imgtk)

        # Update FPS counter
        self.frame_count += 1
        elapsed = time.perf_counter() - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.fps_var.set(f"{self.fps:.1f} FPS")
            self.frame_count = 0
            self.last_fps_time = time.perf_counter()

        # Calculate next update delay
        process_time = time.perf_counter() - start_time
        delay = max(1, int(1000 / 30 - process_time * 1000))

        self.root.after(delay, self._update_ui)

    def capture_image(self):
        """Capture and save current frame"""
        if not self.current_detections:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_path, f"capture_{timestamp}.jpg")

        # Get the most recent frame with detections
        try:
            frame, _ = self.detection_queue.get_nowait()
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.status_var.set(f"Saved: {filename}")
        except queue.Empty:
            self.status_var.set("No frame to capture")

    def _open_gallery(self):
        """Open image gallery viewer"""
        try:
            images = sorted(
                [f for f in os.listdir(self.save_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
                key=lambda x: os.path.getmtime(os.path.join(self.save_path, x)),
                reverse=True
            )

            if not images:
                messagebox.showinfo("Info", "No images found")
                return

            gallery = Toplevel(self.root)
            gallery.title("Image Gallery")
            gallery.geometry("900x700")

            # Image display
            img_label = ttk.Label(gallery)
            img_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

            # Navigation controls
            control_frame = ttk.Frame(gallery)
            control_frame.pack(fill=tk.X, pady=5)

            self.gallery_index = 0
            self.gallery_images = images

            prev_btn = ttk.Button(control_frame, text="Previous",
                                  command=lambda: self._navigate_gallery(gallery, img_label, -1))
            prev_btn.pack(side=tk.LEFT, padx=5)

            next_btn = ttk.Button(control_frame, text="Next",
                                  command=lambda: self._navigate_gallery(gallery, img_label, 1))
            next_btn.pack(side=tk.LEFT, padx=5)

            ttk.Label(control_frame, text=f"1 of {len(images)}").pack(side=tk.LEFT, padx=5)

            delete_btn = ttk.Button(control_frame, text="Delete",
                                    command=lambda: self._delete_image(gallery, img_label))
            delete_btn.pack(side=tk.RIGHT, padx=5)

            # Load first image
            self._load_gallery_image(gallery, img_label, 0)

        except Exception as e:
            messagebox.showerror("Error", f"Gallery error: {str(e)}")

    def _load_gallery_image(self, gallery, label, index):
        """Load image at specified index into gallery"""
        if 0 <= index < len(self.gallery_images):
            self.gallery_index = index
            img_path = os.path.join(self.save_path, self.gallery_images[index])

            try:
                img = Image.open(img_path)

                # Resize to fit window
                gallery.update_idletasks()
                max_width = gallery.winfo_width() - 40
                max_height = gallery.winfo_height() - 100
                img.thumbnail((max_width, max_height), Image.LANCZOS)

                img_tk = ImageTk.PhotoImage(img)
                label.config(image=img_tk)
                label.image = img_tk
                gallery.title(f"Gallery: {self.gallery_images[index]}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def _navigate_gallery(self, gallery, label, direction):
        """Navigate through gallery images"""
        new_index = max(0, min(self.gallery_index + direction, len(self.gallery_images) - 1))
        if new_index != self.gallery_index:
            self._load_gallery_image(gallery, label, new_index)

    def _delete_image(self, gallery, label):
        """Delete current gallery image"""
        if 0 <= self.gallery_index < len(self.gallery_images):
            img_path = os.path.join(self.save_path, self.gallery_images[self.gallery_index])

            if messagebox.askyesno("Confirm", "Delete this image?"):
                try:
                    os.remove(img_path)
                    self.gallery_images.pop(self.gallery_index)

                    if not self.gallery_images:
                        gallery.destroy()
                        messagebox.showinfo("Info", "No more images")
                    else:
                        new_index = min(self.gallery_index, len(self.gallery_images) - 1)
                        self._load_gallery_image(gallery, label, new_index)

                except Exception as e:
                    messagebox.showerror("Error", f"Delete failed: {str(e)}")

    def show_stats(self):
        """Show performance statistics"""
        stats = self.detector.get_performance_stats()

        stats_win = Toplevel(self.root)
        stats_win.title("Performance Stats")
        stats_win.geometry("300x200")

        ttk.Label(stats_win, text=f"Mode: {stats['mode']}").pack(pady=5)
        ttk.Label(stats_win, text=f"Device: {stats['device']}").pack(pady=5)
        ttk.Label(stats_win, text=f"Last Inference: {stats['inference_time'] * 1000:.1f} ms").pack(pady=5)
        ttk.Label(stats_win, text=f"Avg Inference: {stats['avg_inference'] * 1000:.1f} ms").pack(pady=5)
        ttk.Label(stats_win, text=f"Est. FPS: {stats['fps']:.1f}").pack(pady=5)

        ttk.Button(stats_win, text="Close", command=stats_win.destroy).pack(pady=10)

    def quit_app(self):
        """Clean up and exit"""
        self.stop_camera()

        if self.cap is not None:
            self.cap.release()

        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.stop_event.set()
            self.processing_thread.join(timeout=1)

        gc.collect()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = Application(root)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}")
        messagebox.showerror("Error", f"Application crashed: {str(e)}")