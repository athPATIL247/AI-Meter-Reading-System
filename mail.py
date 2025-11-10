# mail.py - Enhanced with better error handling
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time

def send_quick_email(reading):
    """Enhanced email sender with detailed error reporting"""
    
    # === UPDATE THESE VALUES ===
    SENDER_EMAIL = "smsintegrationtest@gmail.com"
    SENDER_PASSWORD = "ovjs wyfx ktdz xmqt"  # Gmail App Password
    RECEIVER_EMAIL = "rathodkartik293@gmail.com"
    # ===========================
    
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create message
        subject = f"⚡ Meter Reading: {reading} kWh"
        body = f"""
Electricity Meter Reading Update:

📊 Reading: {reading} kWh
⏰ Time: {current_time}

This is an automated notification from your meter monitoring system.

---
Sent via Python Meter Reader
"""
        
        print(f"[i] Preparing email...")
        print(f"   From: {SENDER_EMAIL}")
        print(f"   To: {RECEIVER_EMAIL}")
        print(f"   Subject: {subject}")
        
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        print("[i] Connecting to Gmail SMTP...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.set_debuglevel(1)  # Enable debug output
        
        print("[i] Starting TLS...")
        server.starttls()
        
        print("[i] Logging in...")
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        print("[i] Sending email...")
        server.send_message(msg)
        
        print("[i] Closing connection...")
        server.quit()
        
        print(f"✅ Email successfully sent to {RECEIVER_EMAIL}")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ Gmail authentication failed: {e}")
        print("   Please check:")
        print("   1. Gmail App Password is correct")
        print("   2. 2-Factor Authentication is enabled")
        print("   3. You're using App Password, not regular password")
        return False
        
    except smtplib.SMTPException as e:
        print(f"❌ SMTP error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

# Test function
def test_email():
    """Test email functionality"""
    print("🧪 Testing email functionality...")
    test_reading = "01454"
    success = send_quick_email(test_reading)
    
    if success:
        print("🎉 Email test successful! Check your inbox (and spam folder).")
    else:
        print("❌ Email test failed. Check the error messages above.")

if __name__ == "__main__":
    test_email()