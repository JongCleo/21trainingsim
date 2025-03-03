"""
Paper detection module using traditional computer vision.

This module handles:
1. Camera capture
2. Paper detection using contour detection and filtering
3. Text extraction and recognition using simple OpenCV techniques
"""

import time
import json
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np
import pytesseract
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Load the known ad-libs
with open("data/adlibs.json", "r") as f:
    ADLIBS_DATA = json.load(f)

# Extract unique ad-libs for detection
KNOWN_ADLIBS = set()
for item in ADLIBS_DATA:
    text = item.get("text", "").lower()
    if text and text not in ["hahahahahaha", "pussy"]:  # Filter out non-target ad-libs
        KNOWN_ADLIBS.add(text)

logger.info(f"Loaded known ad-libs: {KNOWN_ADLIBS}")


@dataclass
class PaperDetection:
    """Result of paper detection containing the detected text and confidence."""

    text: Optional[str] = None  # Detected text
    confidence: float = 0.0  # Confidence score of the detection


class VisionPaperDetector:
    def __init__(self, camera_id: int = 0):
        """Initialize the paper detector with specified camera.

        Args:
            camera_id: Index of the camera to use (default: 0 for primary webcam)
        """
        self.camera_id = camera_id
        self.cap = None
        self.consecutive_failures = 0
        self.max_failures = 5  # Maximum number of consecutive failures before restarting
        # Set minimum paper area (can be adjusted based on testing)
        self.min_paper_area = 10000
        # Set maximum paper area (to filter out too large regions)
        self.max_paper_area = 250000
        logger.info(f"Initializing VisionPaperDetector with camera {camera_id}")

    def start(self) -> bool:
        """Start the camera capture.

        Returns:
            bool: True if camera started successfully, False otherwise
        """
        try:
            if self.cap is not None:
                self.stop()  # Ensure we clean up any existing capture

            self.cap = cv2.VideoCapture(self.camera_id)

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False

            # Verify we can actually read from the camera
            ret, _ = self.cap.read()
            if not ret:
                logger.error("Camera opened but failed to read first frame")
                return False

            logger.success(f"Successfully started camera {self.camera_id}")
            self.consecutive_failures = 0
            return True

        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False

    def stop(self):
        """Stop the camera capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")

    def _restart_camera(self) -> bool:
        """Attempt to restart the camera after failures."""
        logger.warning(
            f"Attempting to restart camera after {self.consecutive_failures} failures"
        )
        return self.start()

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
    
    def _find_paper_contours(self, frame: np.ndarray) -> List[np.ndarray]:
        """Find potential paper contours in the frame.
        
        Args:
            frame: Input camera frame
            
        Returns:
            List of potential paper contours
        """
        # Make a copy for debug visualization
        debug_img = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple approaches to find contours
        
        # Approach 1: Try edge detection
        edges = cv2.Canny(blurred, 30, 200)
        
        # Approach 2: Try adaptive thresholding 
        thresh1 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Approach 3: Try simple thresholding
        _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Save images for debugging
        cv2.imwrite("debug_edges.jpg", edges)
        cv2.imwrite("debug_thresh1.jpg", thresh1)
        cv2.imwrite("debug_thresh2.jpg", thresh2)
        
        # Combine approaches
        combined = cv2.bitwise_or(edges, thresh1)
        
        # Find contours with different approaches
        contours1, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours2, _ = cv2.findContours(
            thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours3, _ = cv2.findContours(
            thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Collect all contours
        all_contours = contours1 + contours2 + contours3
        
        # Lower min area threshold to be more permissive
        min_area = self.min_paper_area / 2  # More permissive
        max_area = self.max_paper_area * 1.5  # More permissive
        
        logger.debug(f"Found {len(all_contours)} total contours")
        
        # Filter contours by area and shape to find paper-like rectangles
        paper_contours = []
        for contour in all_contours:
            area = cv2.contourArea(contour)
            
            # First check area
            if min_area <= area <= max_area:
                # Approximate the contour to a polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)  # More permissive
                
                # If the polygon has 4 vertices (or close to it), it's likely a paper
                # More permissive: allow 3-6 vertices
                if 3 <= len(approx) <= 6:
                    # Draw contour and corners on debug image
                    cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
                    for point in approx:
                        cv2.circle(debug_img, tuple(point[0]), 5, (0, 0, 255), -1)
                    
                    # Get bounding rect for aspect ratio
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    # Paper-like aspect ratio (between 0.5 and 2.0)
                    if 0.5 <= aspect_ratio <= 2.0:
                        paper_contours.append(approx)
                        
                        # Add text with info
                        text = f"A:{area:.0f} AR:{aspect_ratio:.1f} V:{len(approx)}"
                        cv2.putText(debug_img, text, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Save the debug image
        cv2.imwrite("debug_contours.jpg", debug_img)
        
        logger.debug(f"Found {len(paper_contours)} paper-like contours")
        
        # If we didn't find any, try to be even more permissive
        if not paper_contours:
            logger.debug("No paper contours with standard criteria, trying more permissive filters")
            paper_contours = self._find_simple_regions(frame)
        
        return paper_contours
        
    def _find_simple_regions(self, frame: np.ndarray) -> List[np.ndarray]:
        """Find regions using simpler criteria when standard approaches fail."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple binary threshold
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # Find all contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Take largest few contours
        candidates = contours[:10] if len(contours) >= 10 else contours
        
        # Filter to keep only reasonable sized ones
        result = []
        for contour in candidates:
            area = cv2.contourArea(contour)
            if area > 5000:  # Very permissive minimum size
                # Convert to rectangular approximation
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                result.append(box)
                
        logger.debug(f"Simple region finder found {len(result)} candidates")
        return result
    
    def _extract_paper_roi(self, frame: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """Extract and transform the paper region from the frame.
        
        Args:
            frame: Input camera frame
            contour: Paper contour with 4 points
            
        Returns:
            Transformed and cropped paper region
        """
        # Calculate width and height of the paper
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        
        width = int(rect[1][0])
        height = int(rect[1][1])
        
        # Ensure width and height are non-zero and reasonable
        if width < 10 or height < 10:
            width, height = 300, 200  # Default size
            
        # Get perspective transform
        src_pts = box.astype("float32")
        dst_pts = np.array([
            [0, 0],
            [width-1, 0],
            [width-1, height-1],
            [0, height-1]
        ], dtype="float32")
        
        # Sort source points to match destination order
        src_pts = self._order_points(src_pts)
        
        # Get perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(frame, M, (width, height))
        
        return warped
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in top-left, top-right, bottom-right, bottom-left order."""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum of coordinates
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum
        
        # Difference between coordinates
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference
        
        return rect
    
    def _preprocess_for_ocr(self, paper_roi: np.ndarray) -> np.ndarray:
        """Preprocess paper ROI for better OCR results.
        
        Args:
            paper_roi: Extracted paper region
            
        Returns:
            Preprocessed image ready for OCR
        """
        # Convert to grayscale
        gray = cv2.cvtColor(paper_roi, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast
        gray = cv2.equalizeHist(gray)
        
        # Save the grayscale image
        cv2.imwrite("debug_gray_roi.jpg", gray)
        
        # Try different thresholding approaches for black text on white paper
        
        # 1. Simple binary thresholding - good for high contrast text
        _, thresh1 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        # 2. Otsu's thresholding - automatically determines optimal threshold
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # 3. Adaptive thresholding - good for varying lighting
        thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        
        # 4. Inverse thresholding for white text on black paper
        _, thresh4 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Save all threshold variants for debugging
        cv2.imwrite("debug_thresh1.jpg", thresh1)
        cv2.imwrite("debug_thresh2.jpg", thresh2)
        cv2.imwrite("debug_thresh3.jpg", thresh3)
        cv2.imwrite("debug_thresh4.jpg", thresh4)
        
        # Combine results
        combined = cv2.bitwise_and(thresh1, thresh2)
        
        # Dilate to connect text components
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(combined, kernel, iterations=1)
        cv2.imwrite("debug_dilated.jpg", dilated)
        
        # Invert for black text on white background (what tesseract expects)
        if np.mean(dilated) > 127:
            processed = cv2.bitwise_not(dilated)
        else:
            processed = dilated
        
        cv2.imwrite("debug_final_processed.jpg", processed)
        
        return processed
    
    def _extract_text(self, preprocessed_roi: np.ndarray) -> Tuple[str, float]:
        """Extract text from preprocessed paper ROI.
        
        Args:
            preprocessed_roi: Preprocessed paper image
            
        Returns:
            Tuple of (detected_text, confidence)
        """
        try:
            # Try multiple OCR configurations
            results = []
            
            # 1. Standard OCR for single line
            text1 = pytesseract.image_to_string(
                preprocessed_roi, 
                config='--psm 7 --oem 1'  # Treat image as single line of text
            ).strip().lower()
            results.append(text1)
            
            # 2. Alternative OCR for single word
            text2 = pytesseract.image_to_string(
                preprocessed_roi, 
                config='--psm 8 --oem 1'  # Treat image as single word
            ).strip().lower()
            results.append(text2)
            
            # 3. Use the original image directly
            text3 = pytesseract.image_to_string(
                preprocessed_roi, 
                config='--psm 10'  # Treat as single character
            ).strip().lower()
            results.append(text3)
            
            # 4. Try with the inverted image
            inverted = cv2.bitwise_not(preprocessed_roi)
            text4 = pytesseract.image_to_string(
                inverted, 
                config='--psm 7'
            ).strip().lower()
            results.append(text4)
            
            # Log all results for debugging
            logger.debug(f"OCR results: {results}")
            
            # Select the most common result or the longest
            non_empty_results = [r for r in results if r]
            if non_empty_results:
                # If we have results, use the longest one for now
                text = max(non_empty_results, key=len)
            else:
                return "", 0.0
            
            logger.debug(f"Raw OCR text: {text}")
            
            # Normalize text for better matching
            normalized_text = ''.join(c.lower() for c in text if c.isalnum() or c.isspace())
            
            # Direct match attempts
            
            # First try exact matching (case insensitive)
            for adlib in KNOWN_ADLIBS:
                normalized_adlib = ''.join(c.lower() for c in adlib if c.isalnum() or c.isspace())
                if normalized_text == normalized_adlib:
                    return adlib, 0.95  # Very high confidence for exact match
            
            # Then try substring matching
            for adlib in KNOWN_ADLIBS:
                normalized_adlib = ''.join(c.lower() for c in adlib if c.isalnum() or c.isspace())
                # Check if adlib is in text or text is in adlib
                if normalized_adlib in normalized_text or normalized_text in normalized_adlib:
                    # Length comparison for partial matches
                    similarity = min(len(normalized_text), len(normalized_adlib)) / max(len(normalized_text), len(normalized_adlib))
                    # More confidence if lengths are similar
                    confidence = 0.7 + (similarity * 0.2)
                    return adlib, confidence
            
            # Special pattern matching for common OCR mistakes
            # For "21"
            if re.search(r'(2\s*1|2\s*i|2\s*l|21|twenty\s*one|twent)', normalized_text):
                return "21", 0.9
            
            # For "on God"
            if re.search(r'(on\s*g|o\s*n\s*g|g\s*o\s*d|on\s*god)', normalized_text):
                return "on god", 0.9
            
            # For "straight up"
            if re.search(r'(str|st\s*r|stra|up|aight)', normalized_text):
                return "straight up", 0.9
                
            # Flexible character-based matching as fallback
            best_match = None
            best_score = 0
            
            for adlib in KNOWN_ADLIBS:
                normalized_adlib = ''.join(c.lower() for c in adlib if c.isalnum() or c.isspace())
                
                # Use more sophisticated matching
                # 1. Calculate character overlap ratio
                common_chars = set(normalized_text) & set(normalized_adlib)
                char_score = len(common_chars) / max(len(normalized_adlib), 1)
                
                # 2. Look for character sequences
                seq_score = 0
                for i in range(1, min(len(normalized_text), len(normalized_adlib))):
                    if normalized_text[:i] in normalized_adlib:
                        seq_score = max(seq_score, i / len(normalized_adlib))
                
                # Combined score
                score = (char_score * 0.7) + (seq_score * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_match = adlib
            
            # More permissive threshold
            if best_score > 0.2:  # Lower threshold for fallback matching
                confidence = 0.4 + (best_score * 0.5)  # Scale confidence with score
                return best_match, min(confidence, 0.7)  # Cap at 0.7
            
            # No good match found
            return text, 0.2
            
        except Exception as e:
            logger.error(f"Error in OCR: {e}")
            return "", 0.0
        
    def detect_paper(self, frame: np.ndarray) -> Optional[PaperDetection]:
        """Detect paper and extract text using traditional computer vision.

        Args:
            frame: Input image frame

        Returns:
            Optional[PaperDetection]: Detection result if successful, None otherwise
        """
        try:
            # Find paper-like contours
            paper_contours = self._find_paper_contours(frame)
            
            if not paper_contours:
                logger.debug("No paper contours found")
                return None
            
            # Process largest contour as the paper
            paper_contours.sort(key=cv2.contourArea, reverse=True)
            best_contour = paper_contours[0]
            
            # For visualization, draw the contour on a debug frame (could be saved or displayed)
            debug_frame = frame.copy()
            cv2.drawContours(debug_frame, [best_contour], -1, (0, 255, 0), 2)
            
            # Extract and transform paper region
            paper_roi = self._extract_paper_roi(frame, best_contour)
            
            # Preprocess for OCR
            preprocessed_roi = self._preprocess_for_ocr(paper_roi)
            
            # Extract text
            text, confidence = self._extract_text(preprocessed_roi)
            
            logger.info(f"Detected text: '{text}' with confidence {confidence:.2f}")
            
            # More permissive confidence threshold
            if confidence > 0.3:  # Lower threshold to catch more matches
                return PaperDetection(text=text, confidence=confidence)
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error in paper detection: {e}")
            return None


def main():
    """Test the paper detector with live camera feed."""
    detector = VisionPaperDetector(camera_id=0)

    if not detector.start():
        logger.error("Failed to start camera")
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
                    text = f"Detected (conf: {last_result.confidence:.2f})"
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

                    # Display extracted text
                    if last_result.text:
                        # Truncate text to fit on screen
                        display_text = (
                            last_result.text[:50] + "..."
                            if len(last_result.text) > 50
                            else last_result.text
                        )
                        cv2.putText(
                            display_frame,
                            display_text,
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
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
                    "Press SPACE to detect paper, Q to quit",
                    (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Show frame
                cv2.imshow("Paper Detector", display_frame)
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
                cv2.imshow("Paper Detector", error_frame)

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
