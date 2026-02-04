"""
Simple Hikvision Stream Test
Connect your laptop to their LAN, run this, enter details
"""

import cv2

print("\n" + "="*60)
print("HIKVISION STREAM TEST")
print("="*60)

# Get details
nvr_ip = input("\nEnter NVR IP (e.g., 192.168.1.101): ").strip()
username = input("Enter Username (usually 'admin'): ").strip()
password = input("Enter Password: ").strip()
channel = input("Enter Camera Channel (1, 2, 3...): ").strip()

# Build RTSP URL
# For Hikvision: Channel 1 = 102 (sub-stream), Channel 2 = 202, etc.
channel_code = str(int(channel) * 100 + 2)  # Converts 1→102, 2→202, 3→302
rtsp_url = f"rtsp://{username}:{password}@{nvr_ip}:554/Streaming/Channels/{channel_code}"

print(f"\nTesting: {rtsp_url}")
print("Connecting...")

# Try to connect
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("\n❌ FAILED - Cannot connect!")
    print("\nTry:")
    print("1. Check if IP is correct (ping the IP)")
    print("2. Check username/password")
    print("3. Try channel 1 if unsure")
    print("4. Make sure you're on same LAN network")
else:
    print("✅ Connected!\n")
    print("Reading video... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection")
            break

        # Show info
        cv2.putText(frame, f"Channel {channel} - Press Q to quit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Hikvision Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completed!")
    print(f"\nYour working RTSP URL:")
    print(rtsp_url)
    print("\nCopy this to config.yaml line 14")
