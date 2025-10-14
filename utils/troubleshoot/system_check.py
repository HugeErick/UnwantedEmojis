#!/usr/bin/env python3
"""
System check for video devices and camera troubleshooting
"""

import subprocess
import os
import glob

def check_video_devices():
    """Check for video devices on Linux"""
    print("Checking for video devices...")
    print("-" * 40)
    
    # Check /dev/video* devices
    video_devices = glob.glob("/dev/video*")
    if video_devices:
        print(f"Found video devices: {video_devices}")
        
        # Check permissions
        for device in video_devices:
            try:
                stat_info = os.stat(device)
                print(f"{device}: permissions {oct(stat_info.st_mode)[-3:]}")
            except Exception as e:
                print(f"{device}: error checking permissions - {e}")
    else:
        print("No /dev/video* devices found")
    
    return video_devices

def check_usb_cameras():
    """Check for USB cameras"""
    print("\nChecking for USB cameras...")
    print("-" * 40)
    
    try:
        result = subprocess.run(["lsusb"], capture_output=True, text=True)
        usb_devices = result.stdout
        
        # Look for common camera keywords
        camera_keywords = ["camera", "webcam", "video", "imaging", "uvc"]
        camera_lines = []
        
        for line in usb_devices.split('\n'):
            if any(keyword in line.lower() for keyword in camera_keywords):
                camera_lines.append(line.strip())
        
        if camera_lines:
            print("Potential camera devices:")
            for line in camera_lines:
                print(f"  {line}")
        else:
            print("No obvious camera devices in USB list")
            print("\nFull USB device list:")
            print(usb_devices)
            
    except FileNotFoundError:
        print("lsusb command not found")
    except Exception as e:
        print(f"Error running lsusb: {e}")

def check_v4l_utils():
    """Check Video4Linux utilities"""
    print("\nChecking Video4Linux...")
    print("-" * 40)
    
    try:
        # Check if v4l2-ctl is available
        result = subprocess.run(["v4l2-ctl", "--list-devices"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Video4Linux devices:")
            print(result.stdout)
        else:
            print("v4l2-ctl found but no devices listed")
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("v4l2-ctl not found")
        print("Install with: sudo apt install v4l-utils")
    except Exception as e:
        print(f"Error running v4l2-ctl: {e}")

def check_user_groups():
    """Check if user is in video group"""
    print("\nChecking user groups...")
    print("-" * 40)
    
    try:
        result = subprocess.run(["groups"], capture_output=True, text=True)
        groups = result.stdout.strip()
        print(f"User groups: {groups}")
        
        if "video" in groups:
            print("✓ User is in 'video' group")
        else:
            print("⚠ User is NOT in 'video' group")
            print("Add to video group with: sudo usermod -a -G video $USER")
            print("Then logout and login again")
            
    except Exception as e:
        print(f"Error checking groups: {e}")

def suggest_solutions():
    """Suggest solutions for camera issues"""
    print("\nTroubleshooting suggestions:")
    print("-" * 40)
    print("1. Install camera drivers:")
    print("   sudo apt update")
    print("   sudo apt install cheese  # Test camera app") 
    print("   sudo apt install v4l-utils  # Video4Linux utilities")
    print()
    print("2. Add user to video group:")
    print("   sudo usermod -a -G video $USER")
    print("   # Then logout and login")
    print()
    print("3. Check if camera is being used:")
    print("   sudo lsof /dev/video*")
    print()
    print("4. Test with cheese (GUI camera app):")
    print("   cheese")
    print()
    print("5. For virtual machines:")
    print("   - Enable USB passthrough")
    print("   - Install guest additions")
    print("   - Check VM camera settings")
    print()
    print("6. For laptops with integrated cameras:")
    print("   - Check BIOS settings")
    print("   - Look for camera privacy switches")
    print("   - Check if camera LED is blocked")

def main():
    """Main function"""
    print("System Camera Check")
    print("=" * 40)
    
    check_video_devices()
    check_usb_cameras() 
    check_v4l_utils()
    check_user_groups()
    suggest_solutions()
    
    print("\n" + "=" * 40)
    print("After following suggestions, test again with:")
    print("python camera_test.py")

if __name__ == "__main__":
    main()