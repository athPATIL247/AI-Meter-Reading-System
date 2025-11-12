# robust_meter_ocr_flexible.py
import cv2
import numpy as np
import requests
import easyocr
import time
import csv
import re
import os
from inference_sdk import InferenceHTTPClient
from mail import send_quick_email
from twilio_try import send_reading_sms
import datetime
from dotenv import load_dotenv

load_dotenv()

# ====== CONFIGURATION ======
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
SNAPSHOT_URL = f"{CAM_IP}/capture"

# ROI Settings
ROI_EXPANSION_FACTOR = 0.3
MIN_ROI_WIDTH = 200
MIN_ROI_HEIGHT = 100
MAX_ROI_WIDTH = 800
MAX_ROI_HEIGHT = 400

FALLBACK_ROIS = [
    (0.20, 0.60, 0.10, 0.90),
    (0.50, 0.90, 0.10, 0.90),
    (0.10, 0.50, 0.10, 0.90),
    (0.30, 0.70, 0.05, 0.95),
    (0.40, 0.80, 0.15, 0.85),
]

# Roboflow API
ROBOFLOW_API_KEY = "fuOy83MqlC1LtKZ25J0p"
ROBOFLOW_API_URL = "https://detect.roboflow.com"
DIGIT_DETECTION_MODEL = "number-detection-for-v9/3"
METER_DISPLAY_MODEL = "meter-display-yqxh9/1"

CLIENT = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)
reader = easyocr.Reader(['en'], gpu=False)

print(f"[i] Starting Meter Reading System")
print(f"[i] Results stored in: {RESULTS_DIR}")

# ====== UTILITY FUNCTIONS ======
def should_send_email_today():
    """Check if email was already sent today"""
    email_log_file = os.path.join(RESULTS_DIR, "email_sent.log")
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if os.path.exists(email_log_file):
        try:
            with open(email_log_file, 'r') as f:
                if f.read().strip() == today:
                    return False
        except Exception:
            pass
    
    try:
        with open(email_log_file, 'w') as f:
            f.write(today)
        return True
    except Exception:
        return False

def should_send_sms_today():
    """Check if SMS was already sent today"""
    sms_log_file = os.path.join(RESULTS_DIR, "sms_sent.log")
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(sms_log_file):
        try:
            with open(sms_log_file, 'r') as f:
                if f.read().strip() == today:
                    return False
        except Exception:
            pass

    return True

def record_sms_sent():
    """Record that SMS was sent today"""
    sms_log_file = os.path.join(RESULTS_DIR, "sms_sent.log")
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        with open(sms_log_file, 'w') as f:
            f.write(today)
    except Exception as e:
        print(f"[!] Failed to write SMS log: {e}")

def expand_roi(roi, frame_shape, expansion_factor=ROI_EXPANSION_FACTOR):
    """Expand ROI with constraints"""
    x0, y0, x1, y1 = roi
    h, w = frame_shape[:2]
    
    width_expansion = int((x1 - x0) * expansion_factor)
    height_expansion = int((y1 - y0) * expansion_factor)
    
    x0_expanded = max(0, x0 - width_expansion)
    y0_expanded = max(0, y0 - height_expansion)
    x1_expanded = min(w, x1 + width_expansion)
    y1_expanded = min(h, y1 + height_expansion)
    
    # Ensure minimum dimensions
    if (x1_expanded - x0_expanded) < MIN_ROI_WIDTH:
        center_x = (x0_expanded + x1_expanded) // 2
        x0_expanded = max(0, center_x - MIN_ROI_WIDTH // 2)
        x1_expanded = min(w, center_x + MIN_ROI_WIDTH // 2)
    
    if (y1_expanded - y0_expanded) < MIN_ROI_HEIGHT:
        center_y = (y0_expanded + y1_expanded) // 2
        y0_expanded = max(0, center_y - MIN_ROI_HEIGHT // 2)
        y1_expanded = min(h, center_y + MIN_ROI_HEIGHT // 2)
    
    # Ensure maximum dimensions
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
    """Generate fallback ROIs based on frame dimensions"""
    h, w = frame_shape[:2]
    rois = []
    for y0_frac, y1_frac, x0_frac, x1_frac in FALLBACK_ROIS:
        y0 = int(h * y0_frac)
        y1 = int(h * y1_frac)
        x0 = int(w * x0_frac)
        x1 = int(w * x1_frac)
        rois.append((x0, y0, x1, y1))
    return rois

def calculate_daily_usage(current_reading):
    """Calculate units used in past 24 hours"""
    try:
        if not os.path.exists(CSV_PATH):
            return "0.00"
        
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        if len(rows) < 2:
            return "0.00"
        
        # Get reading from 24 hours ago (approximate)
        twenty_four_hours_ago = datetime.datetime.now() - datetime.timedelta(hours=24)
        past_readings = []
        
        for row in rows[1:]:  # Skip header
            if len(row) >= 2:
                try:
                    row_time = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                    if row_time <= twenty_four_hours_ago:
                        past_readings.append((row_time, float(row[2])))
                except:
                    continue
        
        if past_readings:
            # Get the closest reading to 24 hours ago
            past_readings.sort(key=lambda x: abs((x[0] - twenty_four_hours_ago).total_seconds()))
            past_reading = past_readings[0][1]
            daily_usage = float(current_reading) - past_reading
            return f"{daily_usage:.2f}"
        else:
            return "0.00"
            
    except Exception as e:
        print(f"[!] Error calculating daily usage: {e}")
        return "0.00"

def format_reading(raw_reading):
    """Format reading - 5 digits normally, 6th digit is decimal if present"""
    if not raw_reading:
        return "00000"
    
    # Remove all non-digit characters
    digits_only = re.sub(r'[^0-9]', '', raw_reading)
    
    if len(digits_only) == 6:
        # If 6 digits, treat last digit as decimal: 014547 -> 01454.7
        main_digits = digits_only[:5]
        decimal_digit = digits_only[5]
        return f"{main_digits}.{decimal_digit}"
    elif len(digits_only) == 5:
        # If exactly 5 digits, use as is
        return digits_only
    elif len(digits_only) > 6:
        # If more than 6 digits, take first 5
        return digits_only[:5]
    else:
        # If less than 5, pad with zeros
        return digits_only.zfill(5)

def save_reading_csv(reading):
    """Save comprehensive reading data to CSV"""
    try:
        # Format the reading properly
        formatted_reading = format_reading(reading)
        
        file_exists = os.path.exists(CSV_PATH)
        
        # Calculate additional metrics
        current_time = datetime.datetime.now()
        voltage = "220"  # Standard voltage
        frequency = "50"  # Standard frequency
        daily_usage = calculate_daily_usage(formatted_reading)
        power_factor = "0.95"  # Estimated
        status = "Normal"
        
        with open(CSV_PATH, "a", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Unix_Time", "Reading_kWh", "Voltage_V", 
                    "Frequency_Hz", "Daily_Usage_kWh", "Power_Factor", "Status"
                ])
            writer.writerow([
                current_time.strftime("%Y-%m-%d %H:%M:%S"),
                int(time.time()),
                formatted_reading,
                voltage,
                frequency,
                daily_usage,
                power_factor,
                status
            ])
        
        # Print only essential information
        print(f"[{current_time.strftime('%H:%M:%S')}] Reading: {formatted_reading} kWh | Daily: {daily_usage} kWh")
        
    except Exception as e:
        print(f"[!] CSV save error: {e}")

# ====== CORE PROCESSING FUNCTIONS ======
def fetch_root_image_dir():
    """
    Fetch image locally instead of from ESP32 or URL.
    Reads 'image.png' from the current working directory (or specify path).
    """
    try:
        # Construct the full path to image.png
        img_path = os.path.join(os.getcwd(), "image.png")

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


def fetch_root_image():
    """Fetch image from root URL with HTML parsing fallback"""
    try:
        resp = requests.get(ROOT_URL, timeout=30)
        if b"<html" in resp.content.lower():
            # Try to find img tag in HTML
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
        print(f"[!] Root image fetch error: {e}")
        return None

def fetch_snapshot_image():
    """Try to get image from snapshot endpoint"""
    try:
        resp = requests.get(SNAPSHOT_URL, timeout=30)
        arr = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[!] Snapshot fetch error: {e}")
        return None

def read_frame_via_stream():
    """Initialize video capture from stream"""
    try:
        cap = cv2.VideoCapture(STREAM_URL)
        # Set timeout properties
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            else:
                cap.release()
        return None
    except Exception as e:
        print(f"[!] Stream initialization error: {e}")
        return None

def read_frame_via_jpg():
    """Fetch single JPEG frame"""
    try:
        resp = requests.get(JPG_URL, timeout=30)
        arr = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[!] JPG fetch error: {e}")
        return None

def get_frame_any_method():
    """Try all available methods to get a frame"""
    # Try snapshot first
    frame = fetch_snapshot_image()
    if frame is not None:
        return frame, "snapshot"
    
    # Try JPG URL
    frame = read_frame_via_jpg()
    if frame is not None:
        return frame, "jpg"
    
    # Try root URL with HTML parsing
    frame = fetch_root_image()
    if frame is not None:
        return frame, "root"
    
    return None, "none"

def preprocess(image):
    """Apply preprocessing to enhance image quality"""
    if image is None:
        return None
        
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply sharpening
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    
    # Enhance contrast
    enhanced = cv2.convertScaleAbs(sharpen, alpha=1.5, beta=10)
    
    return enhanced

def detect_digits_and_create_white_bg(frame):
    """Detect digits using Roboflow and create white background image"""
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
        print(f"[!] Digit detection error: {e}")
        if len(frame.shape) == 3:
            h, w = frame.shape[:2]
        else:
            h, w = frame.shape
        return np.full((h, w), 255, dtype=np.uint8), []

def detect_display_roi_roboflow(frame):
    """Detect meter display ROI using Roboflow"""
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
        
        return expand_roi((x0, y0, x1, y1), frame.shape)
    
    except Exception as e:
        print(f"[!] Display ROI detection error: {e}")
        return None

def run_ocr_pipeline(white_bg_crop):
    """Run OCR pipeline with multiple preprocessing variants"""
    if white_bg_crop is None or white_bg_crop.size == 0:
        return ""
    
    def to_rgb_uint8(img):
        if img is None: 
            return None
        img8 = img.astype('uint8') if img.dtype != 'uint8' else img
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR) if len(img8.shape) == 2 else img8

    def preprocess_variants(img):
        img_rgb = to_rgb_uint8(img)
        if img_rgb is None:
            return {}
            
        h, w = img_rgb.shape[:2]
        scale = max(1, int(1000 / max(w, h)))
        if scale > 1:
            img_rgb = cv2.resize(img_rgb, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        th = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        
        return {'orig_rgb': img_rgb, 'gray': gray, 'clahe': clahe, 'th': th}
    
    variants = preprocess_variants(white_bg_crop)
    if not variants:
        return ""
        
    ocr_results = {}
    
    for name, var_img in variants.items():
        try:
            out = reader.readtext(var_img, detail=1)
        except Exception as e:
            print(f"[!] OCR error for {name}: {e}")
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
        ocr_results[name] = joined
    
    def digit_count(s): 
        return len(re.findall(r'\d', s))
    
    best_name, best_score = None, -1
    for name, joined in ocr_results.items():
        score = digit_count(joined)
        if score > best_score:
            best_score, best_name = score, name
    
    if best_name is None or best_score == 0:
        if ocr_results:
            best_name = max(ocr_results.items(), key=lambda kv: len(kv[1]))[0]
    
    if best_name is None:
        return ""
    
    best_joined = ocr_results[best_name]
    cleaned = re.sub(r'\s+', '', best_joined)
    cleaned = re.sub(r'[^0-9]', '', cleaned)  # Only digits, no decimal
    
    return cleaned

def try_multiple_rois(frame, white_bg):
    """Try multiple fallback ROIs to find best reading"""
    fallback_rois = get_fallback_rois(frame.shape)
    best_reading, best_roi, best_digit_count = "", fallback_rois[0], -1
    
    for roi in fallback_rois:
        x0, y0, x1, y1 = roi
        white_bg_crop = white_bg[y0:y1, x0:x1]
        
        if white_bg_crop.size == 0:
            continue
            
        reading = run_ocr_pipeline(white_bg_crop)
        digit_count_val = len(re.findall(r'\d', reading))
        
        if digit_count_val > best_digit_count:
            best_digit_count, best_reading, best_roi = digit_count_val, reading, roi
            
        if digit_count_val >= 4:
            break
    
    return best_reading, best_roi

def process_frame(frame):
    """Main frame processing pipeline"""
    if frame is None:
        return "", (0, 0, 100, 100)
        
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame
    
    preprocessed_frame = preprocess(gray_frame)
    if preprocessed_frame is None:
        return "", (0, 0, 100, 100)
        
    white_bg, digit_predictions = detect_digits_and_create_white_bg(gray_frame)
    
    roi = detect_display_roi_roboflow(gray_frame)
    
    if roi is None:
        reading, roi = try_multiple_rois(frame, white_bg)
    else:
        x0, y0, x1, y1 = roi
        white_bg_crop = white_bg[y0:y1, x0:x1]
        
        if white_bg_crop.size == 0:
            reading, roi = try_multiple_rois(frame, white_bg)
        else:
            reading = run_ocr_pipeline(white_bg_crop)
    
    return reading, roi

def draw_results(frame, roi, reading):
    """Draw results on frame for display"""
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
        
    out = frame.copy()
    x0, y0, x1, y1 = roi
    
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)
    label = f"Reading: {reading if reading else '—'}"
    cv2.putText(out, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return out

# ====== MAIN LOOP ======
def main():
    global ROI_EXPANSION_FACTOR
    
    print("[i] Initializing camera connection...")
    
    # Try stream first
    # cap = read_frame_via_stream()
    cap = None  # Disable stream for local testing
    fallback = False if cap else True

    if fallback:
        print("[i] Using fallback image capture methods")
    else:
        print("[i] Using stream capture")

    last_print, roi_cache, roi_cache_time = 0.0, None, 0
    ROI_CACHE_DURATION = 15.0
    last_reading = None
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    try:
        while True:
            frame = None
            method = "unknown"
            
            if not fallback:
                # Try stream
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print("[!] Stream failing, switching to fallback mode")
                        fallback = True
                        consecutive_failures = 0
                    continue
                else:
                    consecutive_failures = 0
                    method = "stream"
            else:
                # Use fallback methods
                # frame, method = get_frame_any_method()
                frame = fetch_root_image_dir()  # Local image fetch for testing
                method = "local"
                if frame is None:
                    print(f"[!] All capture methods failed, retrying...")
                    time.sleep(2)
                    continue
            
            current_time = time.time()
            if roi_cache is None or (current_time - roi_cache_time) > ROI_CACHE_DURATION:
                reading, roi = process_frame(frame)
                roi_cache, roi_cache_time = (roi, reading), current_time
            else:
                roi, reading = roi_cache

            if reading and reading != last_reading:  # Only process new readings
                now = time.time()
                if now - last_print > 1.0:
                    save_reading_csv(reading)
                    last_reading = reading
                    last_print = now
                    
                    # Send notifications once per day
                    if should_send_email_today():
                        try:
                            formatted_reading = format_reading(reading)
                            print(f"[i] Sending email with reading: {formatted_reading}")
                            send_quick_email(formatted_reading)
                        except Exception as e:
                            print(f"[!] Email sending failed: {e}")
                        
                    if should_send_sms_today():
                        try:
                            formatted_reading = format_reading(reading)
                            print(f"[i] Sending SMS with reading: {formatted_reading}")
                            sid = send_reading_sms(formatted_reading)
                            if sid:
                                record_sms_sent()
                        except Exception as e:
                            print(f"[!] SMS sending failed: {e}")
                        
            disp = draw_results(frame, roi, reading)
            cv2.imshow("Meter Reader - Q:Quit S:Save R:Refresh", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                timestamp = int(time.time())
                filename = os.path.join(RESULTS_DIR, f"snapshot_{timestamp}.jpg")
                cv2.imwrite(filename, disp)
                print(f"[i] Saved snapshot: {filename}")
            if key == ord('r'):
                roi_cache = None
                print("[i] ROI cache cleared")

    except KeyboardInterrupt:
        print("\n[i] System stopped by user")
    except Exception as e:
        print(f"[!] System error: {e}")
    finally:
        if not fallback and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(f"[i] System stopped. Data in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()