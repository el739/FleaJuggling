import os
import time
from datetime import datetime
import pyautogui

def create_screenshots_directory():
    """Create screenshots directory if it doesn't exist"""
    if not os.path.exists('./screenshots'):
        os.makedirs('./screenshots')

def take_screenshot():
    """Take a screenshot and save it with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds precision
    filename = f"screenshot_{timestamp}.png"
    filepath = os.path.join('./screenshots', filename)

    screenshot = pyautogui.screenshot()
    screenshot.save(filepath)
    print(f"Screenshot saved: {filename}")

def main():
    """Main function to run the screenshot capture loop"""
    create_screenshots_directory()
    print("Starting screenshot capture (Press Ctrl+C to stop)...")

    try:
        while True:
            take_screenshot()
            time.sleep(1)  # Wait 1 second
    except KeyboardInterrupt:
        print("\nScreenshot capture stopped.")

if __name__ == "__main__":
    main()