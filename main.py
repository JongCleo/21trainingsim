from dotenv import load_dotenv

from src.vision_paper_detector_moondream import main as moondream_vision_main

load_dotenv()

if __name__ == "__main__":
    # Use Moondream detector
    moondream_vision_main()
