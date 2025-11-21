#!/usr/bin/env python3
"""
Camera detection and testing script
Run this to check if cameras are available on your system
"""

import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_camera_backends():
    """Test different camera backends"""
    backends = {
        cv2.CAP_V4L2: "V4L2 (Linux)",
        cv2.CAP_DSHOW: "DirectShow (Windows)",
        cv2.CAP_AVFOUNDATION: "AVFoundation (macOS)", 
        cv2.CAP_GSTREAMER: "GStreamer",
        cv2.CAP_FFMPEG: "FFmpeg",
        cv2.CAP_ANY: "Any available"
    }
    
    print("Testing camera backends:")
    print("-" * 40)
    
    working_backends = []
    
    for backend_id, backend_name in backends.items():
        try:
            cap = cv2.VideoCapture(0, backend_id)
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"✓ {backend_name}: Working ({width}x{height})")
                    working_backends.append((backend_id, backend_name))
                else:
                    print(f"⚠ {backend_name}: Opens but no frame")
            else:
                print(f"✗ {backend_name}: Cannot open")
            cap.release()
        except Exception as e:
            print(f"✗ {backend_name}: Error - {e}")
    
    return working_backends

def scan_camera_indices():
    """Scan for available camera indices"""
    print("\nScanning camera indices:")
    print("-" * 40)
    
    available_cameras = []
    
    for i in range(10):  # Check indices 0-9
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"✓ Camera {i}: Available ({width}x{height})")
                    available_cameras.append(i)
                else:
                    print(f"⚠ Camera {i}: Opens but no frame")
            cap.release()
        except Exception:
            # Don't print errors for non-existent cameras
            pass
    
    if not available_cameras:
        print("✗ No cameras found")
    
    return available_cameras

def test_camera_live(camera_index=0):
    """Test live camera feed"""
    print(f"\nTesting live feed from camera {camera_index}")
    print("Press 'q' to quit, 's' to save a test frame")
    print("-" * 40)
    
    try:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"✗ Cannot open camera {camera_index}")
            return False
        
        # Set some properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera properties: {width}x{height} @ {fps} FPS")
        
        cv2.namedWindow('Camera Test', cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Add overlay text
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'test_frame_{frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Saved frame as {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"✓ Camera test completed. Total frames: {frame_count}")
        return True
        
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def main():
    """Main function"""
    print("Camera Detection and Testing Tool")
    print("=" * 40)
    
    # Test backends
    working_backends = test_camera_backends()
    
    # Scan indices
    available_cameras = scan_camera_indices()
    
    # Summary
    print("\nSummary:")
    print("-" * 40)
    print(f"Working backends: {len(working_backends)}")
    print(f"Available cameras: {available_cameras}")
    
    if available_cameras:
        print(f"\nRecommendation: Use camera index {available_cameras[0]}")
        
        # Ask user if they want to test live feed
        try:
            test_live = input("\nTest live camera feed? (y/n): ").lower().strip()
            if test_live in ['y', 'yes']:
                test_camera_live(available_cameras[0])
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        print("\nNo cameras detected. Possible solutions:")
        print("1. Check if camera is connected and not used by other apps")
        print("2. Try running with sudo (Linux)")
        print("3. Check camera permissions")
        print("4. Install camera drivers")

if __name__ == "__main__":
    main()