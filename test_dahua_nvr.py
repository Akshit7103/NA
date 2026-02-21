"""
Dahua NVR RTSP Connection Test
Tests connection to Dahua NVR cameras with proper URL formatting.
"""
import cv2
import sys
import urllib.parse


def test_dahua(ip, username, password, channel, subtype=1):
    """
    Test Dahua NVR RTSP connection.

    Dahua URL format:
      rtsp://user:pass@IP:554/cam/realmonitor?channel=X&subtype=Y
        subtype=0 -> Main stream
        subtype=1 -> Sub stream
    """
    # URL-encode password to handle special characters like @
    encoded_password = urllib.parse.quote(password, safe='')

    url = f"rtsp://{username}:{encoded_password}@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}"

    stream_type = "Main stream" if subtype == 0 else "Sub stream"
    print(f"\nTesting Dahua NVR - Channel {channel} ({stream_type})")
    print(f"URL: rtsp://{username}:****@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}")
    print("Connecting...")

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("FAILED - Could not open stream")
        cap.release()
        return False

    ret, frame = cap.read()
    if not ret or frame is None:
        print("FAILED - Connected but could not read frame")
        cap.release()
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"SUCCESS - Resolution: {w}x{h}, FPS: {fps}")
    print(f"\nShowing live preview. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            break

        cv2.putText(frame, f"Dahua NVR Ch{channel} ({stream_type}) {w}x{h}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(f"Dahua NVR - Channel {channel}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True


def main():
    print("=" * 50)
    print("  Dahua NVR RTSP Connection Test")
    print("=" * 50)

    ip = input("\nEnter NVR IP (e.g., 172.16.11.89): ").strip()
    username = input("Enter Username (usually 'admin'): ").strip()
    password = input("Enter Password: ").strip()
    channel = input("Enter Camera Channel (1, 2, 3...): ").strip()

    print("\nStream type:")
    print("  0 = Main stream (high quality)")
    print("  1 = Sub stream (recommended for analytics)")
    subtype = input("Enter stream type (0 or 1) [default: 1]: ").strip()
    subtype = int(subtype) if subtype in ('0', '1') else 1

    success = test_dahua(ip, username, password, int(channel), subtype)

    if not success:
        print("\n--- Troubleshooting ---")
        print("1. Make sure the NVR IP is reachable: ping", ip)
        print("2. Verify credentials are correct")
        print(f"3. Verify channel {channel} exists and has a camera connected")
        print("4. Check if RTSP is enabled on the NVR (port 554)")
        print("5. Try the other stream type (main=0, sub=1)")
        print("\nCommon Dahua RTSP formats:")
        print(f"  rtsp://admin:pass@{ip}:554/cam/realmonitor?channel={channel}&subtype=0")
        print(f"  rtsp://admin:pass@{ip}:554/cam/realmonitor?channel={channel}&subtype=1")

        # Try the other subtype
        other = 0 if subtype == 1 else 1
        retry = input(f"\nTry {'main' if other == 0 else 'sub'} stream instead? (y/n): ").strip().lower()
        if retry == 'y':
            test_dahua(ip, username, password, int(channel), other)


if __name__ == "__main__":
    main()
