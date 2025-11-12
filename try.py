# robust_meter_ocr_flexible.py
# pip install opencv-python easyocr numpy requests inference-sdk

import cv2
import numpy as np
import requests
import easyocr
import time
import csv
import sys
import re   
import os
from inference_sdk import InferenceHTTPClient
from mail import send_quick_email
from twilio_try import send_reading_sms
import datetime
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# ====== FLEXIBLE CONFIG WITH ADJUSTABLE BOUNDING BOX ======
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "meter_results")
DEBUG_DIR = os.path.join(RESULTS_DIR, "debug_crops")
CSV_PATH = os.path.join(RESULTS_DIR, "readings.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

CAM_IP = "http://10.226.81.183"
STREAM_URL = f"{CAM_IP}/stream"
JPG_URL = f"{CAM_IP}/jpg"
ROOT_URL = CAM_IP

# FLEXIBLE ROI SETTINGS - ADJUST THESE AS NEEDED
ROI_EXPANSION_FACTOR = 0.3  # Expand ROI by 30% in all directions (0.3 = 30%)
MIN_ROI_WIDTH = 200         # Minimum width for ROI in pixels
MIN_ROI_HEIGHT = 100        # Minimum height for ROI in pixels
MAX_ROI_WIDTH = 800         # Maximum width for ROI in pixels  
MAX_ROI_HEIGHT = 400        # Maximum height for ROI in pixels

# Multiple fallback ROIs for different meter positions
FALLBACK_ROIS = [
    (0.20, 0.60, 0.10, 0.90),  # Center (default)
    (0.50, 0.90, 0.10, 0.90),  # Bottom
    (0.10, 0.50, 0.10, 0.90),  # Top
    (0.30, 0.70, 0.05, 0.95),  # Wider horizontal
    (0.40, 0.80, 0.15, 0.85),  # Bottom-heavy
]

# Roboflow API configuration
ROBOFLOW_API_KEY = "fuOy83MqlC1LtKZ25J0p"
ROBOFLOW_API_URL = "https://detect.roboflow.com"
DIGIT_DETECTION_MODEL = "number-detection-for-v9/3"
METER_DISPLAY_MODEL = "meter-display-yqxh9/1"   

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url=ROBOFLOW_API_URL,
    api_key=ROBOFLOW_API_KEY
)

# EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

print(f"[i] Results will be stored in: {RESULTS_DIR}")
print(f"[i] ROI Expansion: {ROI_EXPANSION_FACTOR*100}%")
print(f"[i] Using {len(FALLBACK_ROIS)} fallback ROI positions")

def should_send_email_today():
    """Check if we should send email today (once per day)"""
    email_log_file = os.path.join(RESULTS_DIR, "email_sent.log")
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Create directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check if we already sent email today
    if os.path.exists(email_log_file):
        with open(email_log_file, 'r') as f:
            last_sent = f.read().strip()
        
        if last_sent == today:
            print(f"⏰ Email already sent today ({today})")
            return False
    
    # Update log with today's date
    with open(email_log_file, 'w') as f:
        f.write(today)
    
    print(f"📧 Will send email for {today}")
    return True

def format_reading(raw_reading):
    """Format reading - take only first 5 digits, ignore 6th decimal"""
    if not raw_reading:
        return "00000"
    
    # Remove all non-digit characters
    digits_only = re.sub(r'[^0-9]', '', raw_reading)
    
    # Always take first 5 digits only
    if len(digits_only) >= 5:
        return digits_only[:5]
    else:
        # If less than 5, pad with zeros
        return digits_only.zfill(5)

def should_send_sms_today():
    """Check if we should send SMS today (once per day)."""
    sms_log_file = os.path.join(RESULTS_DIR, "sms_sent.log")
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if os.path.exists(sms_log_file):
        try:
            with open(sms_log_file, 'r', encoding='utf-8') as f:
                last_sent = f.read().strip()
            if last_sent == today:
                print(f"⏰ SMS already sent today ({today})")
                return False
        except Exception:
            pass  # Ignore file read errors and allow SMS to be sent again

    print(f"📩 Will send SMS for {today}")
    return True


def record_sms_sent():
    """Record today's date to prevent multiple SMS in one day"""
    sms_log_file = os.path.join(RESULTS_DIR, "sms_sent.log")
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        with open(sms_log_file, 'w', encoding='utf-8') as f:
            f.write(today)
    except Exception as e:
        print(f"[!] Failed to write SMS log: {e}")

# ====== IMPROVED BOUNDING BOX FUNCTIONS ======
def expand_roi(roi, frame_shape, expansion_factor=ROI_EXPANSION_FACTOR):
    """Expand ROI with bounds checking"""
    x0, y0, x1, y1 = roi
    h, w = frame_shape[:2]
    
    # Calculate expansion amounts
    width_expansion = int((x1 - x0) * expansion_factor)
    height_expansion = int((y1 - y0) * expansion_factor)
    
    # Expand ROI
    x0_expanded = max(0, x0 - width_expansion)
    y0_expanded = max(0, y0 - height_expansion)
    x1_expanded = min(w, x1 + width_expansion)
    y1_expanded = min(h, y1 + height_expansion)
    
    # Ensure minimum size
    if (x1_expanded - x0_expanded) < MIN_ROI_WIDTH:
        center_x = (x0_expanded + x1_expanded) // 2
        x0_expanded = max(0, center_x - MIN_ROI_WIDTH // 2)
        x1_expanded = min(w, center_x + MIN_ROI_WIDTH // 2)
    
    if (y1_expanded - y0_expanded) < MIN_ROI_HEIGHT:
        center_y = (y0_expanded + y1_expanded) // 2
        y0_expanded = max(0, center_y - MIN_ROI_HEIGHT // 2)
        y1_expanded = min(h, center_y + MIN_ROI_HEIGHT // 2)
    
    # Limit maximum size
    if (x1_expanded - x0_expanded) > MAX_ROI_WIDTH:
        center_x = (x0_expanded + x1_expanded) // 2
        x0_expanded = max(0, center_x - MAX_ROI_WIDTH // 2)
        x1_expanded = min(w, center_x + MAX_ROI_WIDTH // 2)
    
    if (y1_expanded - y0_expanded) > MAX_ROI_HEIGHT:
        center_y = (y0_expanded + y1_expanded) // 2
        y0_expanded = max(0, center_y - MAX_ROI_HEIGHT // 2)
        y1_expanded = min(h, center_y + MAX_ROI_HEIGHT // 2)
    
    return (x0_expanded, y0_expanded, x1_expanded, y1_expanded)

def get_fallback_rois(frame_shape):
    """Generate multiple fallback ROIs based on frame size"""
    h, w = frame_shape[:2]
    rois = []
    
    for y0_frac, y1_frac, x0_frac, x1_frac in FALLBACK_ROIS:
        y0 = int(h * y0_frac)
        y1 = int(h * y1_frac)
        x0 = int(w * x0_frac)
        x1 = int(w * x1_frac)
        rois.append((x0, y0, x1, y1))
    
    return rois

def save_reading_csv(reading):
    """Safely save reading to CSV with error handling - ONLY 5 DIGITS"""
    try:
        # Format the reading to only 5 digits
        formatted_reading = format_reading(reading)
        
        file_exists = os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Unix Time", "Reading"])
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                int(time.time()),
                formatted_reading  # Use formatted reading (5 digits only)
            ])
        print(f"[✓] Saved to CSV: {formatted_reading} (original: {reading})")
    except Exception as e:
        print(f"[!] CSV save error: {e}")
        
def save_debug_image(image, prefix="crop"):
    """Save debug image with timestamp"""
    try:
        timestamp = int(time.time())
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(DEBUG_DIR, filename)
        cv2.imwrite(filepath, image)
        return filepath
    except Exception as e:
        print(f"[!] Debug image save error: {e}")
        return None

# ====== REST OF YOUR FUNCTIONS ======
def fetch_root_image():
    try:
        resp = requests.get(ROOT_URL, timeout=30)
        if b"<html" in resp.content.lower():
            m = re.search(r'<img\s+src="([^"]+)"', resp.text)
            if m:
                img_url = m.group(1)
                if not img_url.startswith("http"):
                    img_url = f"{CAM_IP.rstrip('/')}/{img_url.lstrip('/')}"
                resp = requests.get(img_url, timeout=30)
            else:
                return None
        arr = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print("[!] root fetch error:", e)
        return None

def fetch_root_image_dir():
    """
    Fetch image locally instead of from ESP32 or URL.
    Reads 'image.png' from the current working directory (or specify path).
    """
    try:
        # Construct the full path to image.png
        img_path = os.path.join(os.getcwd(), "testing_dataset/meter1.png")

        # Check if the file exists
        if not os.path.exists(img_path):
            print("[!] image.png not found in directory:", os.getcwd())
            return None

        # Read the image using OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # Validate successful read
        if img is None:
            print("[!] Failed to read image.png (possibly corrupted)")
            return None

        return img

    except Exception as e:
        print("[!] local image fetch error:", e)
        return None

def read_frame_via_stream():
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        return None
    return cap

def read_frame_via_jpg():
    try:
        resp = requests.get(JPG_URL, timeout=30)
        arr = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print("[!] jpg fetch error:", e)
        return None

def preprocess(image):
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    adjusted = cv2.convertScaleAbs(sharpen, alpha=1.5, beta=10)
    return adjusted

def to_rgb_uint8(img):
    if img is None:
        return None
    img8 = img.astype('uint8') if img.dtype != 'uint8' else img
    if len(img8.shape) == 2:
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    return img8

def preprocess_variants(img):
    img_rgb = to_rgb_uint8(img)
    h, w = img_rgb.shape[:2]
    scale = max(1, int(1000 / max(w, h)))
    if scale > 1:
        img_rgb = cv2.resize(img_rgb, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    th = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    th_inv = cv2.bitwise_not(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    
    return {
        'orig_rgb': img_rgb, 'gray': gray, 'clahe': clahe,
        'th': th, 'th_inv': th_inv, 'closed': closed,
    }

def detect_digits_and_create_white_bg(frame):
    try:
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
            
        temp_path = os.path.join(RESULTS_DIR, "temp_digit_frame.jpg")
        cv2.imwrite(temp_path, gray_frame)
        
        results = CLIENT.infer(temp_path, model_id=DIGIT_DETECTION_MODEL)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        predictions = results.get('predictions', [])
        white_background = np.full_like(gray_frame, 255, dtype=np.uint8)
        
        for prediction in predictions:
            x = int(prediction['x'])
            y = int(prediction['y'])
            class_id = prediction['class']
            cv2.putText(white_background, class_id, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return white_background, predictions
        
    except Exception as e:
        print("[!] Digit detection error:", e)
        if len(frame.shape) == 3:
            h, w = frame.shape[:2]
        else:
            h, w = frame.shape
        return np.full((h, w), 255, dtype=np.uint8), []

def detect_display_roi_roboflow(frame):
    try:
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
            
        temp_path = os.path.join(RESULTS_DIR, "temp_display_frame.jpg")
        cv2.imwrite(temp_path, gray_frame)
        
        crop = CLIENT.infer(temp_path, model_id=METER_DISPLAY_MODEL)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        predictions = crop.get('predictions', [])
        if not predictions:
            return None
        
        jd = predictions[0]
        x = int(jd['x'])
        y = int(jd['y'])
        height = int(jd['height'])
        width = int(jd['width'])
        
        x0 = int(x - width / 2)
        x1 = int(x + width / 2)
        y0 = int(y - height / 2)
        y1 = int(y + height / 2)
        
        # Expand the detected ROI
        expanded_roi = expand_roi((x0, y0, x1, y1), frame.shape)
        return expanded_roi
    
    except Exception as e:
        print("[!] Display detection error:", e)
        return None

def digit_count(s):
    return len(re.findall(r'\d', s))

def run_ocr_pipeline(white_bg_crop, debug_name=""):
    if white_bg_crop.size == 0:
        return ""
    
    # Save debug crop
    debug_path = save_debug_image(white_bg_crop, f"crop_{debug_name}")
    if debug_path:
        print(f"[i] Saved debug crop: {debug_path}")
    
    variants = preprocess_variants(white_bg_crop)
    ocr_results = {}
    
    for name, var_img in variants.items():
        try:
            out = reader.readtext(var_img, detail=1)
        except Exception as e:
            print(f'[!] OCR error on variant {name}:', e)
            out = []
        
        texts = []
        for item in out:
            if len(item) >= 2:
                bbox, txt = item[0], item[1]
                xs = [p[0] for p in bbox]
                x_center = sum(xs) / len(xs)
                texts.append((x_center, txt))
        
        texts_sorted = sorted(texts, key=lambda t: t[0])
        joined = ''.join([t[1] for t in texts_sorted])
        ocr_results[name] = { 'joined': joined, 'items': texts_sorted }
        
        print(f'OCR raw ({name}): {joined} (boxes: {len(texts_sorted)})')
    
    best_name = None
    best_score = -1
    for name, info in ocr_results.items():
        score = digit_count(info['joined'])
        if score > best_score:
            best_score = score
            best_name = name
    
    if best_name is None or best_score == 0:
        if ocr_results:
            best_name = max(ocr_results.items(), key=lambda kv: len(kv[1]['joined']))[0]
    
    if best_name is None:
        print('No OCR candidates found.')
        return ""
    
    best_joined = ocr_results[best_name]['joined']
    cleaned = re.sub(r'\s+', '', best_joined)
    cleaned = re.sub(r'[^0-9\.]', '', cleaned)
    
    print(f'Selected variant: {best_name}, raw: {repr(best_joined)}')
    print(f'Processed OCR output: {repr(cleaned)}')
    
    return cleaned

def try_multiple_rois(frame, white_bg):
    """Try multiple ROI positions and return the best reading"""
    fallback_rois = get_fallback_rois(frame.shape)
    best_reading = ""
    best_roi = fallback_rois[0]
    best_digit_count = -1
    
    for i, roi in enumerate(fallback_rois):
        x0, y0, x1, y1 = roi
        white_bg_crop = white_bg[y0:y1, x0:x1]
        
        if white_bg_crop.size == 0:
            continue
            
        print(f"[i] Trying ROI {i+1}: {roi}")
        reading = run_ocr_pipeline(white_bg_crop, f"roi_{i}")
        
        digit_count_val = digit_count(reading)
        if digit_count_val > best_digit_count:
            best_digit_count = digit_count_val
            best_reading = reading
            best_roi = roi
            
        if digit_count_val >= 4:  # Good enough reading
            break
    
    return best_reading, best_roi

def process_frame(frame):
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame
    
    preprocessed_frame = preprocess(gray_frame)
    white_bg, digit_predictions = detect_digits_and_create_white_bg(gray_frame)
    
    # Try to detect display with Roboflow first
    roi = detect_display_roi_roboflow(gray_frame)
    
    if roi is None:
        print("[!] Roboflow detection failed, trying multiple fallback ROIs...")
        reading, roi = try_multiple_rois(frame, white_bg)
    else:
        print("[✓] Using Roboflow-detected ROI (expanded)")
        x0, y0, x1, y1 = roi
        white_bg_crop = white_bg[y0:y1, x0:x1]
        
        if white_bg_crop.size == 0:
            print("[!] Empty crop after ROI application, trying fallback ROIs")
            reading, roi = try_multiple_rois(frame, white_bg)
        else:
            reading = run_ocr_pipeline(white_bg_crop, "roboflow")
    
    return reading, roi

def draw_results(frame, roi, reading):
    out = frame.copy()
    x0, y0, x1, y1 = roi
    
    # Draw ROI rectangle
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)
    
    # Add reading text
    label = f"Reading: {reading if reading else '—'}"
    cv2.putText(out, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add ROI info
    roi_info = f"ROI: {x1-x0}x{y1-y0}"
    cv2.putText(out, roi_info, (x0, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return out

# ====== MAIN LOOP WITH ADJUSTABLE SETTINGS ======
def main():
    # Declare global variables at the TOP of the function
    global ROI_EXPANSION_FACTOR
    
    print(f"[i] Starting flexible meter reading pipeline...")
    print(f"[i] ROI Expansion: {ROI_EXPANSION_FACTOR*100}%")
    print(f"[i] Min ROI: {MIN_ROI_WIDTH}x{MIN_ROI_HEIGHT}")
    print(f"[i] Max ROI: {MAX_ROI_WIDTH}x{MAX_ROI_HEIGHT}")
    print(f"[i] All results will be saved in: {RESULTS_DIR}")
    
    # Allow user to adjust settings
    adjust = input("Adjust settings? (y/n): ").lower().strip()
    if adjust == 'y':
        try:
            expansion = float(input(f"ROI expansion factor (current: {ROI_EXPANSION_FACTOR}): ") or ROI_EXPANSION_FACTOR)
            ROI_EXPANSION_FACTOR = max(0.1, min(1.0, expansion))  # Limit between 10% and 100%
            print(f"[i] ROI expansion set to: {ROI_EXPANSION_FACTOR*100}%")
        except ValueError:
            print("[!] Invalid input, using default settings")
    
    # cap = read_frame_via_stream()
    cap = None  # Disable stream for local testing
    fallback = False
    if cap is None:
        print("[i] Stream unavailable - falling back to JPG/root.")
        fallback = True

    last_print = 0.0
    roi_cache = None
    roi_cache_time = 0
    ROI_CACHE_DURATION = 15.0  # Longer cache since we're trying multiple ROIs
    try:
        while True:
            if fallback:
                # frame = read_frame_via_jpg() or fetch_root_image()
                frame = fetch_root_image_dir()  # Local image fetch for testing
                if frame is None:
                    print("[!] No frame; retrying in 1s...")
                    time.sleep(1)
                    continue
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[!] Stream failed; switching to fallback.")
                    fallback = True
                    continue

            current_time = time.time()
            if roi_cache is None or (current_time - roi_cache_time) > ROI_CACHE_DURATION:
                reading, roi = process_frame(frame)
                roi_cache = (roi, reading)
                roi_cache_time = current_time
            else:
                roi, reading = roi_cache

            if reading:
                now = time.time()
                if now - last_print > 1.0:
                    # Format reading to 5 digits before using
                    formatted_reading = format_reading(reading)
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Meter Reading: {formatted_reading} (raw: {reading})")
                    save_reading_csv(reading)  # This function now uses format_reading internally
                    
                    # Send email once per day with debug info
                    print(f"[i] Checking if we should send email today...")
                    if should_send_email_today():
                        try:
                            print("[i] Attempting to send email...")
                            send_quick_email(formatted_reading)  # Send formatted reading
                        except Exception as e:
                            print(f"[!] Email error: {e}")
                    else:
                        print("[i] Skipping email - already sent today")
                        
                    # attempt SMS once per day
                    if should_send_sms_today():
                        try:
                            sid = send_reading_sms(formatted_reading)  # Send formatted reading
                            if sid:
                                record_sms_sent()
                                print("[i] SMS sent successfully ✅ SID:", sid)
                        except Exception as e:
                            print(f"[!] SMS error: {e}")
                    else:
                        print("[i] Skipping SMS - already sent today")   
                                
            disp = draw_results(frame, roi, reading)
            cv2.imshow("Flexible Meter OCR - q=quit / s=save / r=refresh / +/-=adjust ROI", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                timestamp = int(time.time())
                snapshot_path = os.path.join(RESULTS_DIR, f"snapshot_{timestamp}.jpg")
                cv2.imwrite(snapshot_path, disp)
                print(f"[i] Saved snapshot: {snapshot_path}")
            if key == ord('r'):
                roi_cache = None
                print("[i] Cache cleared - reprocessing next frame")
            if key == ord('+') or key == ord('='):
                # Increase ROI expansion
                ROI_EXPANSION_FACTOR = min(1.0, ROI_EXPANSION_FACTOR + 0.1)
                roi_cache = None
                print(f"[i] ROI expansion increased to: {ROI_EXPANSION_FACTOR*100}%")
            if key == ord('-') or key == ord('_'):
                # Decrease ROI expansion
                ROI_EXPANSION_FACTOR = max(0.1, ROI_EXPANSION_FACTOR - 0.1)
                roi_cache = None
                print(f"[i] ROI expansion decreased to: {ROI_EXPANSION_FACTOR*100}%")

    except KeyboardInterrupt:
        print("\n[i] Script interrupted by user")
    except Exception as e:
        print(f"[!] Unexpected error: {e}")
    finally:
        # Cleanup
        if not fallback and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(f"[i] Script stopped. Check results in: {RESULTS_DIR}")
        print(f"[i] Final ROI expansion: {ROI_EXPANSION_FACTOR*100}%")

if __name__ == "__main__":
    main()
