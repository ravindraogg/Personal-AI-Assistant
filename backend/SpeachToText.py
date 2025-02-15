import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import dotenv_values

def setup_driver():
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Create Data directory if it doesn't exist
    data_dir = os.path.join(current_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save HTML file
    html_path = os.path.join(data_dir, "Voice.html")
    
    # Configure Chrome options
    chrome_options = Options()
    # Remove --headless for debugging
    # chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--use-fake-ui-for-media-stream")
    chrome_options.add_argument("--use-fake-device-for-media-stream")
    chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")
    chrome_options.add_argument("--allow-file-access-from-files")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Check ChromeDriver
    chrome_driver_path = os.path.join(current_dir, "chromedriver.exe")
    if not os.path.exists(chrome_driver_path):
        raise FileNotFoundError(f"ChromeDriver not found at: {chrome_driver_path}")
    
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver, f"file:///{html_path}"

def speech_recognition():
    driver = None
    try:
        print("\nInitializing speech recognition...")
        driver, html_path = setup_driver()
        
        print(f"Loading page: {html_path}")
        driver.get(html_path)
        
        # Wait for elements and check initial status
        wait = WebDriverWait(driver, 10)
        debug_element = wait.until(EC.presence_of_element_located((By.ID, "debug")))
        print(f"Initial debug status: {debug_element.text}")
        
        start_button = wait.until(EC.element_to_be_clickable((By.ID, "start")))
        start_button.click()
        print("Recognition started")
        
        max_wait_time = 30
        start_time = time.time()
        last_debug = ""
        
        while time.time() - start_time < max_wait_time:
            try:
                current_debug = driver.find_element(By.ID, "debug").text
                if current_debug != last_debug:
                    print(f"Debug: {current_debug}")
                    last_debug = current_debug
                
                output_text = driver.find_element(By.ID, "output").text
                if output_text:
                    print(f"Speech detected: {output_text}")
                    return output_text.capitalize()
                
                time.sleep(0.1)
            except Exception as e:
                print(f"Loop error: {e}")
                break
        
        print("Timeout reached - no speech detected")
        return None
        
    except Exception as e:
        print(f"Recognition error: {e}")
        return None
        
    finally:
        if driver:
            try:
                driver.quit()
                print("Browser closed")
            except:
                print("Error closing browser")

if __name__ == "__main__":
    print("Speech Recognition System Starting...")
    
    while True:
        try:
            result = speech_recognition()
            if result:
                print(f"Final result: {result}")
            else:
                print("No result - retrying in 2 seconds...")
            time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopping speech recognition...")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(2)
