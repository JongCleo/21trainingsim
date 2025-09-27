"""Configuration management for vision detection."""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class VisionConfig:
    """Configuration for vision detection system."""

    # Moondream settings
    moondream_local_endpoint: str = "http://localhost:2020/v1"
    moondream_api_key: Optional[str] = None

    # Performance settings
    max_detection_latency_ms: float = 200.0
    image_resize_max_dimension: int = 800

    # Camera settings
    camera_id: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    frame_rate: int = 30

    # Debug settings
    save_debug_images: bool = True
    performance_log_interval: int = 10  # Log performance every N detections

    @classmethod
    def from_env(cls) -> "VisionConfig":
        """Create config from environment variables."""
        return cls(
            moondream_api_key=os.getenv("MOONDREAM_API_KEY"),
            camera_id=int(os.getenv("CAMERA_ID", "0")),
            frame_width=int(os.getenv("CAMERA_WIDTH", "1280")),
            frame_height=int(os.getenv("CAMERA_HEIGHT", "720")),
            frame_rate=int(os.getenv("CAMERA_FPS", "30")),
            max_detection_latency_ms=float(os.getenv("MAX_DETECTION_LATENCY_MS", "200.0")),
            image_resize_max_dimension=int(os.getenv("IMAGE_RESIZE_MAX_DIM", "800")),
            save_debug_images=os.getenv("SAVE_DEBUG_IMAGES", "true").lower() == "true",
            performance_log_interval=int(os.getenv("PERFORMANCE_LOG_INTERVAL", "10")),
        )

    def validate(self) -> bool:
        """Validate configuration settings."""
        if self.camera_id < 0:
            return False
        if self.max_detection_latency_ms <= 0:
            return False
        if self.image_resize_max_dimension <= 0:
            return False
        if self.frame_width <= 0 or self.frame_height <= 0:
            return False
        if self.frame_rate <= 0:
            return False
        return True