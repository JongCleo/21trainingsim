"""
Paper detection module using traditional computer vision.

This module handles:
1. Camera capture
2. Paper detection using contour detection and filtering
3. Text detection using template matching
"""

import json
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
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
        self.max_failures = (
            5  # Maximum number of consecutive failures before restarting
        )
        self.min_paper_area = 10000  # Set minimum paper area
        self.max_paper_area = 250000  # Set maximum paper area
        logger.info(f"Initializing VisionPaperDetector with camera {camera_id}")

    def start(self) -> bool:
        """Start the camera capture.

        Returns:
            bool: True if camera started successfully, False otherwise
        """
        try:
            if self.cap is not None:
                self.stop()  # Ensure we clean up any existing capture

            # Try different camera backends
            backends = [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION]
            for backend in backends:
                self.cap = cv2.VideoCapture(self.camera_id + backend)

                if not self.cap.isOpened():
                    logger.warning(f"Failed to open camera with backend {backend}")
                    continue

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

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
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply Canny edge detection with automatic thresholding
        sigma = 0.33
        median_val = float(np.median(denoised.astype(np.float32)))
        lower = int(max(0, (1.0 - sigma) * median_val))
        upper = int(min(255, (1.0 + sigma) * median_val))
        edges = cv2.Canny(denoised, lower, upper)

        # Dilate edges to connect components
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours from edges
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Save debug images
        cv2.imwrite("debug_edges.jpg", edges)
        cv2.imwrite("debug_dilated.jpg", dilated)

        logger.debug(f"Found {len(contours)} total contours")
        
        # Filter contours by area and shape to find paper-like rectangles
        paper_contours = []
        min_area = 5000  # More permissive minimum area
        max_area = self.max_paper_area * 2

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:10]:  # Only look at the 10 largest contours
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                # Approximate the contour to a polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # If the polygon has 4-8 vertices, it's likely a paper
                if 4 <= len(approx) <= 8:
                    # Get bounding rect for aspect ratio
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    # Calculate the center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w // 2, y + h // 2

                    # Calculate distance from center of frame
                    frame_center_x = frame.shape[1] // 2
                    frame_center_y = frame.shape[0] // 2
                    dist_from_center = np.sqrt(
                        (cx - frame_center_x) ** 2 + (cy - frame_center_y) ** 2
                    )

                    # Score the contour based on multiple factors
                    score = 0.0

                    # Aspect ratio score (prefer paper-like ratios)
                    if 0.5 <= aspect_ratio <= 2.0:
                        score += 0.3

                    # Size score (prefer medium-sized objects)
                    size_ratio = area / (frame.shape[0] * frame.shape[1])
                    if 0.05 <= size_ratio <= 0.3:
                        score += 0.3

                    # Distance score (prefer objects closer to center)
                    max_dist = np.sqrt(
                        (frame.shape[1] // 2) ** 2 + (frame.shape[0] // 2) ** 2
                    )
                    dist_score = 1.0 - (dist_from_center / max_dist)
                    score += 0.4 * dist_score

                    # If score is good enough, keep this contour
                    if score > 0.5:
                        # Draw contour and corners on debug image
                        cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 2)
                        for point in approx:
                            point_arr = point.reshape(-1)
                            pt = (int(point_arr[0]), int(point_arr[1]))
                            cv2.circle(debug_img, pt, 5, (0, 0, 255), -1)

                        paper_contours.append(approx)
                        
                        # Add text with info
                        text = f"A:{area:.0f} AR:{aspect_ratio:.1f} S:{score:.2f}"
                        cv2.putText(
                            debug_img,
                            text,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                        )
        
        # Save the debug image
        cv2.imwrite("debug_contours.jpg", debug_img)
        
        logger.debug(f"Found {len(paper_contours)} paper-like contours")
        
        return paper_contours
        
    def _find_simple_regions(self, frame: np.ndarray) -> List[np.ndarray]:
        """Find regions using simpler criteria when standard approaches fail."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple binary threshold
        _, thresh = cv2.threshold(
            gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        
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
        dst_pts = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )
        
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
    
    def _preprocess_for_ocr(self, paper_roi: np.ndarray) -> List[np.ndarray]:
        """Preprocess paper ROI for better text detection.
        
        Args:
            paper_roi: Extracted paper region
            
        Returns:
            List of preprocessed paper images
        """
        # Convert to grayscale
        gray = cv2.cvtColor(paper_roi, cv2.COLOR_BGR2GRAY)
        
        # Save the grayscale image
        cv2.imwrite("debug_gray_roi.jpg", gray)
        
        # Create multiple preprocessing variants
        preprocessed_images = []

        # 1. Simple binary thresholding with high contrast
        _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite("debug_thresh1.jpg", thresh1)
        preprocessed_images.append(thresh1)
        
        # 2. Otsu's thresholding - automatically determines optimal threshold
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite("debug_thresh2.jpg", thresh2)
        preprocessed_images.append(thresh2)

        # 3. Inverse thresholding for white text on black paper
        _, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite("debug_thresh4.jpg", thresh4)
        preprocessed_images.append(thresh4)

        # 4. Noise removal and edge enhancement
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh6 = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        cv2.imwrite("debug_denoised_thresh.jpg", thresh6)
        preprocessed_images.append(thresh6)

        # 5. Morphological operations to enhance text
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh2, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        cv2.imwrite("debug_morph.jpg", eroded)
        preprocessed_images.append(eroded)

        return preprocessed_images

    def _extract_text(self, preprocessed_images: List[np.ndarray]) -> Tuple[str, float]:
        """Extract text using template matching.
        
        Args:
            preprocessed_images: List of preprocessed paper images
            
        Returns:
            Tuple of (detected_text, confidence)
        """
        try:
            # Try template matching for the specific ad-libs
            template_result = self._try_template_matching(preprocessed_images)
            if template_result:
                return template_result
            return "", 0.0

        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return "", 0.0

    def _try_template_matching(
        self, preprocessed_images: List[np.ndarray]
    ) -> Optional[Tuple[str, float]]:
        """Try to match the image against templates of known ad-libs.

        Args:
            preprocessed_images: List of preprocessed paper images

        Returns:
            Tuple of (text, confidence) if match found, None otherwise
        """
        try:
            best_match = None
            best_score = 0

            # Try each preprocessed image
            for img in preprocessed_images:
                # Resize to a standard size for pattern matching
                resized = cv2.resize(img, (200, 100), interpolation=cv2.INTER_AREA)

                # Save debug image
                cv2.imwrite("debug_template_matching.jpg", resized)

                # Extract features for each ad-lib

                # For "21" - Look for two distinct characters with specific shapes
                score_21 = self._detect_21(resized)
                if score_21 > best_score:
                    best_score = score_21
                    best_match = "21"

                # For "on god" - Look for multiple characters with specific pattern
                score_on_god = self._detect_on_god(resized)
                if score_on_god > best_score:
                    best_score = score_on_god
                    best_match = "on god"

                # For "straight up" - Look for longer text with more characters
                score_straight_up = self._detect_straight_up(resized)
                if score_straight_up > best_score:
                    best_score = score_straight_up
                    best_match = "straight up"

            # Return the best match if score is high enough
            if best_score > 0.5 and best_match is not None:
                logger.info(
                    f"Template matching found '{best_match}' with score {best_score:.2f}"
                )
                return best_match, best_score * 0.9

            return None

        except Exception as e:
            logger.error(f"Error in template matching: {e}")
            return None

    def _detect_21(self, img: np.ndarray) -> float:
        """Detect the "21" ad-lib based on visual characteristics.

        Args:
            img: Preprocessed image

        Returns:
            Confidence score (0-1)
        """
        try:
            # Convert to binary if not already
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            else:
                binary = img

            # Save debug image
            cv2.imwrite("debug_21_input.jpg", binary)

            # Split the image into left and right halves (for "2" and "1")
            height, width = binary.shape
            left_half = binary[:, : width // 2]
            right_half = binary[:, width // 2 :]

            # Count white pixels in each half
            left_white = np.sum(left_half == 255)
            right_white = np.sum(right_half == 255)

            # Calculate total white pixels
            total_white = left_white + right_white

            # Check if there are enough white pixels overall
            if total_white < 50:
                return 0.0

            # Check if left half has more pixels (for "2")
            left_ratio = left_white / total_white

            # Check if right half has a vertical line pattern (for "1")
            # Sum each column in the right half
            right_cols = np.sum(right_half == 255, axis=0)

            # Find the strongest vertical line
            max_vertical = np.max(right_cols)
            vertical_score = max_vertical / height

            # Calculate score based on these characteristics
            score = 0.0

            # Strong vertical line in right half is crucial for "1"
            if vertical_score > 0.6:  # At least 60% of height should be white
                score += 0.4
                # Check if it's near the center of the right half
                max_col_idx = np.argmax(right_cols)
                center_dist = abs(max_col_idx - (right_half.shape[1] // 2))
                if center_dist < right_half.shape[1] // 4:  # Within central 50%
                    score += 0.2

            # Left half should have "2" characteristics
            if 0.6 < left_ratio < 0.8:  # "2" typically has more pixels than "1"
                score += 0.2

                # Check for the distinctive curve of "2"
                # Split left half into thirds vertically
                top = left_half[: height // 3, :]
                middle = left_half[height // 3 : 2 * height // 3, :]
                bottom = left_half[2 * height // 3 :, :]

                # "2" typically has more pixels in top and bottom
                top_density = np.sum(top == 255) / top.size
                middle_density = np.sum(middle == 255) / middle.size
                bottom_density = np.sum(bottom == 255) / bottom.size

                if top_density > middle_density and bottom_density > middle_density:
                    score += 0.2

            # Penalize if there are too many separate components (like "on god")
            num_components, _ = cv2.connectedComponents(binary)
            if num_components <= 3:  # "21" should have 2-3 components
                score += 0.2
            else:
                score -= 0.3  # Heavy penalty for too many components

            # Save debug visualization
            debug_vis = np.zeros((height, width, 3), dtype=np.uint8)
            debug_vis[:, : width // 2] = cv2.cvtColor(left_half, cv2.COLOR_GRAY2BGR)
            debug_vis[:, width // 2 :] = cv2.cvtColor(right_half, cv2.COLOR_GRAY2BGR)
            # Draw vertical line detection
            max_col = np.argmax(right_cols) + width // 2
            cv2.line(debug_vis, (max_col, 0), (max_col, height), (0, 255, 0), 1)
            cv2.imwrite("debug_21_analysis.jpg", debug_vis)

            logger.debug(
                f"'21' detection - Vertical score: {vertical_score:.2f}, Left ratio: {left_ratio:.2f}, Components: {num_components}, Score: {score:.2f}"
            )

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Error in _detect_21: {e}")
            return 0.0

    def _detect_on_god(self, img: np.ndarray) -> float:
        """Detect the "on god" ad-lib based on visual characteristics.

        Args:
            img: Preprocessed image

        Returns:
            Confidence score (0-1)
        """
        try:
            # Convert to binary if not already
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            else:
                binary = img

            # Save debug image
            cv2.imwrite("debug_on_god_input.jpg", binary)

            # Count connected components - "on god" should have more components than "21"
            num_components, labels = cv2.connectedComponents(binary)

            # "on god" typically has 5-7 components (o, n, g, o, d)
            if num_components < 4 or num_components > 8:
                return 0.0  # Early exit if component count is wrong

            # Split image into left ("on") and right ("god") parts
            height, width = binary.shape
            left_part = binary[:, : width // 2]
            right_part = binary[:, width // 2 :]

            # Look for "on" characteristics in left part
            score = 0.0

            # "on" should have 2-3 components
            left_components, _ = cv2.connectedComponents(left_part)
            if 2 <= left_components <= 3:
                score += 0.2

            # Look for circular shapes (o's) using contour analysis
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            num_circles = 0

            for contour in contours:
                # Filter small contours
                if cv2.contourArea(contour) < 50:
                    continue

                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = (
                    4 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter)
                )

                if circularity > 0.6:  # Threshold for circular shapes
                    num_circles += 1

            # "on god" should have 2 circular shapes (the o's)
            if num_circles == 2:
                score += 0.3
            elif num_circles == 1:
                score += 0.1

            # Check for 'd' characteristics in right part
            right_bottom = right_part[height // 2 :, :]
            right_top = right_part[: height // 2, :]

            # 'd' typically has more pixels in bottom half
            if np.sum(right_bottom == 255) > np.sum(right_top == 255):
                score += 0.2

            # Check spacing between components
            # Project onto horizontal axis
            h_projection = np.sum(binary == 255, axis=0)
            h_projection = (
                h_projection / np.max(h_projection)
                if np.max(h_projection) > 0
                else h_projection
            )

            # Find gaps between letters
            gaps = []
            in_gap = False
            for i, val in enumerate(h_projection):
                if val < 0.1 and not in_gap:  # Start of gap
                    in_gap = True
                    gaps.append(i)
                elif val >= 0.1 and in_gap:  # End of gap
                    in_gap = False

            # "on god" should have consistent spacing between words
            if 3 <= len(gaps) <= 5:
                score += 0.2

            # Heavy penalty if we see a strong vertical line (characteristic of "1")
            v_projection = np.sum(binary == 255, axis=0)
            if np.max(v_projection) > height * 0.7:  # Strong vertical line detected
                score -= 0.5

            # Save debug visualization
            debug_vis = np.zeros((height, width, 3), dtype=np.uint8)
            # Color each component differently
            for label in range(1, num_components):
                component = (labels == label).astype(np.uint8) * 255
                color = tuple(np.random.randint(0, 255, 3).tolist())
                debug_vis[labels == label] = color
            cv2.imwrite("debug_on_god_analysis.jpg", debug_vis)

            logger.debug(
                f"'on god' detection - Components: {num_components}, Circles: {num_circles}, Gaps: {len(gaps)}, Score: {score:.2f}"
            )

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Error in _detect_on_god: {e}")
            return 0.0

    def _detect_straight_up(self, img: np.ndarray) -> float:
        """Detect the "straight up" ad-lib based on visual characteristics.

        Args:
            img: Preprocessed image

        Returns:
            Confidence score (0-1)
        """
        try:
            # Convert to binary if not already
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            else:
                binary = img

            # Calculate horizontal projection profile
            h_projection = np.sum(binary == 255, axis=0)

            # Normalize
            if np.max(h_projection) > 0:
                h_projection = h_projection / np.max(h_projection)

            # "straight up" is longer with more characters and gaps
            # Find potential character boundaries by looking for gaps
            gaps = []
            in_gap = False
            for i, val in enumerate(h_projection):
                if val < 0.2 and not in_gap:
                    in_gap = True
                    gaps.append(i)
                elif val >= 0.2 and in_gap:
                    in_gap = False

            # "straight up" should have more gaps than "on god" or "21"
            num_gaps = len(gaps)

            # Calculate score based on number of gaps
            score = 0.0

            # "straight up" has more characters, so more gaps
            if num_gaps >= 7:
                score += 0.7
            elif num_gaps >= 5:
                score += 0.4

            # "straight up" should have more white pixels overall
            white_ratio = np.sum(binary == 255) / binary.size

            # Typical white ratio for text
            if 0.1 <= white_ratio <= 0.3:
                score += 0.3

            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error in _detect_straight_up: {e}")
            return 0.0
        
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
            
            # For visualization, draw the contour on a debug frame
            debug_frame = frame.copy()
            cv2.drawContours(debug_frame, [best_contour], -1, (0, 255, 0), 2)
            cv2.imwrite("debug_detected_contour.jpg", debug_frame)
            
            # Extract and transform paper region
            paper_roi = self._extract_paper_roi(frame, best_contour)
            cv2.imwrite("debug_paper_roi.jpg", paper_roi)
            
            # Preprocess for OCR - returns multiple preprocessing variants
            preprocessed_images = self._preprocess_for_ocr(paper_roi)
            
            # Extract text from all preprocessing variants
            text, confidence = self._extract_text(preprocessed_images)
            
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
