"""
Paper detection module using Moondream vision model.

This module handles:
1. Camera capture (reused from existing detector)
2. Multi-approach paper and text detection using Moondream
3. Performance optimization for real-time gameplay
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

from src.vision.config import VisionConfig
from src.vision.moondream_client import MoondreamClient, MoondreamResponse

load_dotenv()

# Load known ad-libs (reuse existing logic)
with open("data/adlibs.json", "r") as f:
    ADLIBS_DATA = json.load(f)

KNOWN_ADLIBS = set()
for item in ADLIBS_DATA:
    text = item.get("text", "").lower()
    if text and text not in ["hahahahahaha", "pussy"]:
        KNOWN_ADLIBS.add(text)

logger.info(f"Loaded known ad-libs: {KNOWN_ADLIBS}")


@dataclass
class PaperDetection:
    """Result of paper detection containing the detected text and confidence."""

    text: Optional[str] = None
    confidence: float = 0.0


class VisionPaperDetector:
    """Paper detector using Moondream vision model."""

    def __init__(self, config: Optional[VisionConfig] = None):
        """Initialize the paper detector with configuration.

        Args:
            config: Vision configuration object (defaults to environment-based config)
        """
        self.config = config or VisionConfig.from_env()

        if not self.config.validate():
            raise ValueError("Invalid configuration provided")

        self.camera_id = self.config.camera_id
        self.cap = None
        self.consecutive_failures = 0
        self.max_failures = 5

        # Initialize Moondream client with cloud priority
        self.moondream = MoondreamClient(
            api_key=self.config.moondream_api_key,
            local_endpoint=self.config.moondream_local_endpoint,
        )

        # Performance tracking
        self.detection_count = 0
        self.total_latency = 0.0

        logger.info(
            f"Initializing Moondream VisionPaperDetector with camera {self.camera_id}"
        )
        logger.info(
            f"Config: max_latency={self.config.max_detection_latency_ms}ms, "
            f"image_resize={self.config.image_resize_max_dimension}px"
        )

    def start(self) -> bool:
        """Start camera and initialize Moondream model.

        Returns:
            bool: True if both camera and model started successfully, False otherwise
        """
        # Initialize Moondream first
        if not self.moondream.initialize():
            logger.error("Failed to initialize Moondream model")
            return False

        logger.info(f"Using Moondream {self.moondream.endpoint_type} endpoint")

        # Start camera (reuse existing logic)
        return self._start_camera()

    def stop(self):
        """Stop the camera capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")

    def _start_camera(self) -> bool:
        """Start the camera capture (adapted from existing detector).

        Returns:
            bool: True if camera started successfully, False otherwise
        """
        try:
            if self.cap is not None:
                self.stop()  # Ensure we clean up any existing capture

            # Try different camera backends (from existing detector)
            backends = [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION]
            for backend in backends:
                self.cap = cv2.VideoCapture(self.camera_id + backend)

                if not self.cap.isOpened():
                    logger.warning(f"Failed to open camera with backend {backend}")
                    continue

                # Set camera properties from config
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.config.frame_rate)

                # Try reading a frame with a timeout
                start_time = time.time()
                while time.time() - start_time < 2.0:  # 2 second timeout
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logger.success(
                            f"Successfully started camera with backend {backend}"
                        )
                        self.consecutive_failures = 0
                        return True
                    time.sleep(0.1)

                # If we get here, reading failed
                self.cap.release()
                self.cap = None

            logger.error("Failed to start camera with any backend")
            return False

        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False

    def _restart_camera(self) -> bool:
        """Attempt to restart the camera after failures."""
        logger.warning(
            f"Attempting to restart camera after {self.consecutive_failures} failures"
        )
        return self._start_camera()

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera.

        Returns:
            Optional[np.ndarray]: Frame if successful, None otherwise
        """
        if self.cap is None:
            logger.warning("Camera not started")
            return None

        ret, frame = self.cap.read()
        if not ret:
            self.consecutive_failures += 1
            logger.warning(
                f"Failed to capture frame (failure #{self.consecutive_failures})"
            )

            if self.consecutive_failures >= self.max_failures:
                if not self._restart_camera():
                    logger.error("Failed to restart camera after multiple failures")
            return None

        self.consecutive_failures = 0  # Reset counter on successful capture
        return frame

    def detect_paper(self, frame: np.ndarray) -> Optional[PaperDetection]:
        """Detect paper and extract text using Moondream.

        Args:
            frame: Input image frame

        Returns:
            Optional[PaperDetection]: Detection result if successful, None otherwise
        """
        if not self.moondream.is_initialized:
            logger.error("Moondream model not initialized")
            return None

        try:
            start_time = time.time()

            # Convert frame to PIL Image for Moondream
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Optimize image size for performance
            pil_image = self._optimize_image_for_detection(pil_image)

            # Save debug image if enabled
            if self.config.save_debug_images:
                cv2.imwrite("debug_moondream_input.jpg", frame)

            # Try multiple detection approaches in order of speed/accuracy
            result = (
                self._try_direct_query(pil_image)
                or self._try_paper_detection_query(pil_image)
                or self._try_fallback_detection(pil_image)
            )

            if result and result.success:
                latency = (time.time() - start_time) * 1000
                self._update_performance_stats(latency)
                logger.info(
                    f"Detection completed in {latency:.1f}ms using {result.method_used}"
                )

                # Validate result against known ad-libs
                if self._validate_detection(result.text):
                    return PaperDetection(
                        text=result.text, confidence=result.confidence
                    )

            return None

        except Exception as e:
            logger.error(f"Error in paper detection: {e}")
            return None

    def _optimize_image_for_detection(self, image: Image.Image) -> Image.Image:
        """Optimize image for faster Moondream processing."""
        # Resize if too large (maintain aspect ratio)
        max_dimension = self.config.image_resize_max_dimension
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to {new_size} for faster processing")

        return image

    def _try_direct_query(self, image: Image.Image) -> Optional[MoondreamResponse]:
        """Fast approach: Direct query for text on paper."""
        try:
            start_time = time.time()

            # Ask directly about text on paper
            response = self.moondream.query(
                image,
                "What text is written on the paper or sign in this image? Only respond with the exact text you see, or 'none' if no text is visible.",
            )

            latency = (time.time() - start_time) * 1000

            if response and "answer" in response:
                text = response["answer"].lower().strip()

                # Clean up common variations
                text = self._normalize_text(text)

                if text in KNOWN_ADLIBS:
                    return MoondreamResponse(
                        success=True,
                        text=text,
                        confidence=0.8,  # High confidence for direct recognition
                        latency_ms=latency,
                        method_used="direct_query",
                    )

        except Exception as e:
            logger.debug(f"Direct query failed: {e}")

        return None

    def _try_paper_detection_query(
        self, image: Image.Image
    ) -> Optional[MoondreamResponse]:
        """Medium approach: Detect paper first, then query for text."""
        try:
            start_time = time.time()

            # First detect if there's a paper
            paper_check = self.moondream.query(
                image,
                "Is there a piece of paper or sign with text visible in this image? Answer yes or no.",
            )

            if paper_check and "answer" in paper_check:
                if "yes" in paper_check["answer"].lower():
                    # If paper detected, ask for text
                    text_response = self.moondream.query(
                        image,
                        "What specific text or number is written on the paper? Respond with just the text.",
                    )

                    if text_response and "answer" in text_response:
                        text = text_response["answer"].lower().strip()
                        text = self._normalize_text(text)

                        if text in KNOWN_ADLIBS:
                            latency = (time.time() - start_time) * 1000
                            return MoondreamResponse(
                                success=True,
                                text=text,
                                confidence=0.7,
                                latency_ms=latency,
                                method_used="paper_detection_query",
                            )

        except Exception as e:
            logger.debug(f"Paper detection query failed: {e}")

        return None

    def _try_fallback_detection(
        self, image: Image.Image
    ) -> Optional[MoondreamResponse]:
        """Fallback approach using object detection if queries fail."""
        try:
            start_time = time.time()

            # Use object detection to find papers/signs
            detection_response = self.moondream.detect(image, "paper")

            if detection_response and "objects" in detection_response:
                objects = detection_response["objects"]
                if objects:
                    # Found paper(s), now try to read text
                    text_response = self.moondream.query(
                        image,
                        "Focus on the detected paper or sign. What text is written on it? Just the text, nothing else.",
                    )

                    if text_response and "answer" in text_response:
                        text = self._normalize_text(text_response["answer"])

                        if text in KNOWN_ADLIBS:
                            latency = (time.time() - start_time) * 1000
                            return MoondreamResponse(
                                success=True,
                                text=text,
                                confidence=0.6,  # Lower confidence for fallback
                                latency_ms=latency,
                                method_used="fallback_detection",
                            )

        except Exception as e:
            logger.debug(f"Fallback detection failed: {e}")

        return None

    def _normalize_text(self, text: str) -> str:
        """Normalize detected text to match known ad-libs."""
        if not text:
            return ""

        text = text.lower().strip()

        # Handle common variations
        text = text.replace("twenty-one", "21")
        text = text.replace("twenty one", "21")
        text = text.replace("ongod", "on god")
        text = text.replace("straightup", "straight up")

        # Remove common OCR artifacts
        text = text.replace(".", "").replace(",", "").replace("!", "")
        text = text.replace("none", "")

        return text.strip()

    def _validate_detection(self, text: Optional[str]) -> bool:
        """Validate that detected text matches expected ad-libs."""
        if not text:
            return False
        return text.lower() in KNOWN_ADLIBS

    def _update_performance_stats(self, latency_ms: float):
        """Track performance statistics for monitoring."""
        self.detection_count += 1
        self.total_latency += latency_ms

        if self.detection_count % self.config.performance_log_interval == 0:
            avg_latency = self.total_latency / self.detection_count
            logger.info(
                f"Performance: {self.detection_count} detections, avg {avg_latency:.1f}ms"
            )

            # Reset for next batch
            self.detection_count = 0
            self.total_latency = 0.0


def main():
    """Test the paper detector with live camera feed."""
    config = VisionConfig.from_env()
    logger.info(f"Starting Moondream detector with config: {config}")

    detector = VisionPaperDetector(config)

    if not detector.start():
        logger.error("Failed to start detector")
        return

    last_result = None  # Store last detection result
    last_latency = None  # Store last API latency
    detection_start_time = None  # Track when detection started
    frame_count = 0  # Track number of successful frames
    last_frame_time = time.time()  # Track frame timing

    try:
        while True:
            frame = detector.get_frame()
            current_time = time.time()

            if frame is not None:
                frame_count += 1
                if frame_count % 30 == 0:  # Log FPS every 30 frames
                    fps = 30 / (current_time - last_frame_time)
                    logger.info(f"Camera FPS: {fps:.1f}")
                    last_frame_time = current_time

                # Create display frame
                display_frame = frame.copy()

                # Check for key press
                key = cv2.waitKey(1) & 0xFF

                # Detect paper on spacebar press
                if key == ord(" "):
                    logger.info("Spacebar pressed - detecting paper")
                    detection_start_time = time.time()
                    last_result = detector.detect_paper(frame)
                    if last_result is not None:
                        last_latency = time.time() - detection_start_time
                        logger.info(
                            f"Detection completed in {last_latency:.2f} seconds"
                        )
                # Quit on 'q' press
                elif key == ord("q"):
                    break

                # Draw detection results if we have any
                if last_result is not None:
                    # Display detection text and latency
                    text = f"Detected: '{last_result.text}' (conf: {last_result.confidence:.2f})"
                    if last_latency is not None:
                        text += f" - Latency: {last_latency * 1000:.0f}ms"

                    cv2.putText(
                        display_frame,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                # Show "Processing..." during detection
                elif detection_start_time is not None:
                    current_latency = time.time() - detection_start_time
                    cv2.putText(
                        display_frame,
                        f"Processing... ({current_latency * 1000:.0f}ms)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),  # Yellow color
                        2,
                    )

                # Add instruction text
                cv2.putText(
                    display_frame,
                    f"Press SPACE to detect paper, Q to quit | {detector.moondream.endpoint_type.upper()}",
                    (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Show frame
                cv2.imshow("Moondream Paper Detector", display_frame)
            else:
                # If frame capture failed, show error window
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    error_frame,
                    "Camera Error - Attempting to reconnect...",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Moondream Paper Detector", error_frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Small delay before retry
                time.sleep(0.5)

    finally:
        detector.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
