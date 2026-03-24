import logging
import time
import os

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Basic safety setup for PyAutoGUI
if PYAUTOGUI_AVAILABLE:
    try:
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
    except Exception as e:
        logger.warning(f"Could not initialize PyAutoGUI settings: {e}")
        PYAUTOGUI_AVAILABLE = False

class BCIController:
    def __init__(self, step_size=30):
        self.step_size = step_size
        self.active = False
        
    def execute_command(self, command):
        """Translate predicted BCI command to cursor action."""
        if not self.active:
            return
            
        if not PYAUTOGUI_AVAILABLE:
            logger.warning("PyAutoGUI not available. Simulation mode only.")
            return
            
        try:
            if command == 'LEFT':
                pyautogui.moveRel(-self.step_size, 0, duration=0.1)
            elif command == 'RIGHT':
                pyautogui.moveRel(self.step_size, 0, duration=0.1)
            elif command == 'UP':
                pyautogui.moveRel(0, -self.step_size, duration=0.1)
            elif command == 'DOWN':
                pyautogui.moveRel(0, self.step_size, duration=0.1)
            elif command == 'CLICK':
                pyautogui.click()
            elif command == 'IDLE':
                pass # Do nothing
        except Exception as e:
            logger.error(f"Cursor control error: {e}")

    def set_active(self, status: bool):
        self.active = status
        logger.info(f"BCI Controller active: {status}")

if __name__ == "__main__":
    # Test script (safety check: hover in corner to abort)
    controller = BCIController()
    controller.set_active(True)
    print("Test: Moving RIGHT 3 times...")
    for _ in range(3):
        controller.execute_command('RIGHT')
        time.sleep(0.5)
