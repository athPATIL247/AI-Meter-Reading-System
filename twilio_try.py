"""Simple Twilio helper: send a meter reading as an SMS.

Usage:
 - Import `send_reading_sms(reading, ...)` from this file and call it.
 - Or run this file directly to send a test message using environment variables.

Environment variables used if parameters are not provided:
 - TWILIO_ACCOUNT_SID
 - TWILIO_AUTH_TOKEN
 - TWILIO_FROM_NUMBER
 - SMS_TO_NUMBER
"""

import os
from twilio.rest import Client


def send_reading_sms(reading, account_sid=None, auth_token=None, from_number=None, to_number=None):
    """Send an SMS containing the reading via Twilio.

    Parameters:
        reading (str): The meter reading text to send.
        account_sid (str): Twilio Account SID. If None, read from TWILIO_ACCOUNT_SID env var.
        auth_token (str): Twilio Auth Token. If None, read from TWILIO_AUTH_TOKEN env var.
        from_number (str): Twilio phone number sending the SMS. If None, read from TWILIO_FROM_NUMBER env var.
        to_number (str): Recipient phone number. If None, read from SMS_TO_NUMBER env var.

    Returns:
        str|bool: Message SID on success, False on failure.
    """
    account_sid = account_sid or os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = auth_token or os.environ.get('TWILIO_AUTH_TOKEN')
    from_number = from_number or os.environ.get('TWILIO_FROM_NUMBER')
    to_number = to_number or os.environ.get('SMS_TO_NUMBER')

    if not all([account_sid, auth_token, from_number, to_number]):
        print('[!] Twilio credentials or phone numbers are not fully set.\n'
              '    Required: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, SMS_TO_NUMBER')
        return False

    try:
        client = Client(account_sid, auth_token)
        body = f"Your readings are: {reading}"
        message = client.messages.create(
            body=body,
            from_=from_number,
            to=to_number,
        )
        print(f"[✓] SMS sent (sid={message.sid})")
        return message.sid
    except Exception as e:
        print(f"[!] Failed to send SMS via Twilio: {e}")
        return False


if __name__ == '__main__':
    # Quick manual test when running the script directly.
    import argparse

    parser = argparse.ArgumentParser(description='Send a meter reading via Twilio')
    parser.add_argument('reading', nargs='?', default='12345', help='Reading to send')
    parser.add_argument('--from', dest='from_number', help='Twilio from number')
    parser.add_argument('--to', dest='to_number', help='Destination number')
    parser.add_argument('--sid', dest='sid', help='Twilio Account SID')
    parser.add_argument('--token', dest='token', help='Twilio Auth Token')
    args = parser.parse_args()

    result = send_reading_sms(
        args.reading,
        account_sid=args.sid,
        auth_token=args.token,
        from_number=args.from_number,
        to_number=args.to_number,
    )

    if result:
        print('Done:', result)
    else:
        print('SMS not sent')