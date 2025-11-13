import os
import csv
import datetime
from twilio.rest import Client


def get_latest_meter_data(csv_path="meter_results/readings.csv"):
    """Get the latest meter reading data from CSV"""
    try:
        if not os.path.exists(csv_path):
            return None
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        if len(rows) < 2:  # Only header or empty
            return None
            
        # Get the last row (most recent reading)
        last_row = rows[-1]
        
        # Expected columns: Timestamp, Unix_Time, Reading_kWh, Voltage_V, Frequency_Hz, Daily_Usage_kWh, Power_Factor, Status
        if len(last_row) >= 7:
            return {
                'timestamp': last_row[0],
                'reading': last_row[2],
                'voltage': last_row[3],
                'frequency': last_row[4],
                'daily_usage': last_row[5],
                'power_factor': last_row[6],
                'status': last_row[7] if len(last_row) > 7 else 'Normal'
            }
        else:
            return None
            
    except Exception as e:
        print(f"[!] Error reading meter data from CSV: {e}")
        return None


def format_reading_for_sms(reading, meter_data=None):
    """Format the reading and meter data into a comprehensive SMS message"""
    
    if meter_data:
        # Full message with all details
        message = f"""🔋 ELECTRICITY METER READING

📊 Current: {meter_data['reading']} kWh
📅 Time: {meter_data['timestamp']}
⚡ 24h Usage: {meter_data['daily_usage']} kWh

📈 Power Stats:
   Voltage: {meter_data['voltage']}V
   Frequency: {meter_data['frequency']}Hz
   Power Factor: {meter_data['power_factor']}

✅ Status: {meter_data['status']}

Automated Meter Reader System"""
    else:
        # Fallback message if CSV data not available
        message = f"""🔋 METER READING: {reading} kWh

Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Automated Reading System"""

    return message


def send_reading_sms(reading, meter_data=None, account_sid=None, auth_token=None, from_number=None, to_number=None, csv_path=None):
    try:
        # Allow overrides, otherwise read from environment (support multiple var names)
        account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        from_number = from_number or os.getenv("TWILIO_FROM_NUMBER")
        to_number = to_number or os.getenv("SMS_TO_NUMBER") or os.getenv("TWILIO_TO_NUMBER") or os.getenv("TWILIO_TO")

        missing = [name for name, val in (
            ("TWILIO_ACCOUNT_SID", account_sid),
            ("TWILIO_AUTH_TOKEN", auth_token),
            ("TWILIO_FROM_NUMBER", from_number),
            ("SMS_TO_NUMBER or TWILIO_TO_NUMBER", to_number),
        ) if not val]

        if missing:
            print("[!] Missing Twilio environment/argument values:", ", ".join(missing))
            print("    Set env vars TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER and SMS_TO_NUMBER (or pass overrides).")
            return None

        # If caller didn't provide meter_data but a csv_path exists, try to pull latest data
        if meter_data is None and csv_path:
            try:
                latest = get_latest_meter_data(csv_path)
                if latest:
                    meter_data = latest
                    # If reading passed is None or empty, use CSV reading
                    if not reading:
                        reading = latest.get('reading')
            except Exception:
                meter_data = None

        client = Client(account_sid, auth_token)

        # Build message with meter data if available
        if meter_data:
            message_body = (
                f"🔢 Meter Reading: {reading} kWh\n"
                f"⚡ Voltage: {meter_data.get('voltage', 'N/A')}V\n"
                f"📊 Frequency: {meter_data.get('frequency', 'N/A')}Hz\n"
                f"🔋 Units Used: {meter_data.get('units_used', '0.00')} kWh\n"
                f"🕒 Time: {meter_data.get('timestamp', 'N/A')}"
            )
        else:
            message_body = f"🔢 Meter Reading: {reading} kWh"

        message = client.messages.create(
            body=message_body,
            from_=from_number,
            to=to_number
        )

        return message.sid

    except Exception as e:
        print(f"[!] SMS sending failed: {e}")
        return None

def send_test_sms():
    """Send a test SMS with sample data"""
    # Create sample meter data for testing
    sample_data = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'reading': '01454.7',
        'voltage': '220',
        'frequency': '50', 
        'daily_usage': '12.45',
        'power_factor': '0.95',
        'status': 'Normal'
    }
    
    body = format_reading_for_sms('01454.7', sample_data)
    
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    from_number = os.environ.get('TWILIO_FROM_NUMBER')
    to_number = os.environ.get('SMS_TO_NUMBER') or os.environ.get('TWILIO_TO_NUMBER') or os.environ.get('TWILIO_TO')
    
    if not all([account_sid, auth_token, from_number, to_number]):
        print('[!] Missing Twilio environment variables')
        return False
        
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=body,
            from_=from_number,
            to=to_number,
        )
        print(f"[✓] Test SMS sent (sid={message.sid})")
        print("Message content:")
        print(body)
        return message.sid
    except Exception as e:
        print(f"[!] Failed to send test SMS: {e}")
        return False


if __name__ == '__main__':
    # Quick manual test when running the script directly.
    import argparse

    parser = argparse.ArgumentParser(description='Send a meter reading via Twilio')
    parser.add_argument('reading', nargs='?', default=None, help='Reading to send')
    parser.add_argument('--csv', dest='csv_path', default='meter_results/readings.csv', help='Path to CSV file')
    parser.add_argument('--from', dest='from_number', help='Twilio from number')
    parser.add_argument('--to', dest='to_number', help='Destination number')
    parser.add_argument('--sid', dest='sid', help='Twilio Account SID')
    parser.add_argument('--token', dest='token', help='Twilio Auth Token')
    parser.add_argument('--test', action='store_true', help='Send test SMS with sample data')
    args = parser.parse_args()

    if args.test:
        result = send_test_sms()
    else:
        result = send_reading_sms(
            args.reading or 'TEST-12345',
            csv_path=args.csv_path,
            account_sid=args.sid,
            auth_token=args.token,
            from_number=args.from_number,
            to_number=args.to_number,
        )

    if result:
        print('Done:', result)
    else:
        print('SMS not sent')