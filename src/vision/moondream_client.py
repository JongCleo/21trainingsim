"""
Moondream client wrapper for 21 Savage paper detection.
Handles model initialization, fallback logic, and response parsing.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import moondream as md
import numpy as np
from loguru import logger
from PIL import Image


@dataclass
class MoondreamResponse:
    """Response from Moondream model."""

    success: bool
    text: Optional[str] = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    method_used: str = "unknown"


class MoondreamClient:
    """Client for Moondream vision model with cloud priority."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        local_endpoint: str = "http://localhost:2020/v1",
    ):
        self.api_key = api_key
        self.local_endpoint = local_endpoint
        self.model = None
        self.use_cloud = True  # Prioritize cloud for user preference

        logger.info(f"Initializing MoondreamClient (cloud priority: {bool(api_key)})")

    def initialize(self) -> bool:
        """Initialize Moondream model with cloud priority."""
        if md is None:
            logger.error("Moondream package not available")
            return False

        # Try cloud endpoint first (user preference)
        if self.api_key and self._try_cloud_model():
            return True

        # Fallback to local if cloud fails
        if self._try_local_model():
            return True

        logger.error("Failed to initialize any Moondream model")
        return False

    def _try_cloud_model(self) -> bool:
        """Try to initialize cloud Moondream model."""
        try:
            logger.info("Attempting to connect to Moondream cloud API")
            self.model = md.vl(api_key=self.api_key)

            # Test with a simple call
            test_response = self._test_model_connection(self.model)
            if test_response:
                logger.success("Cloud Moondream model initialized successfully")
                self.use_cloud = True
                return True
        except Exception as e:
            logger.warning(f"Cloud Moondream model failed: {e}")
        return False

    def _try_local_model(self) -> bool:
        """Try to initialize local Moondream model."""
        try:
            logger.info(
                f"Attempting to connect to local Moondream at {self.local_endpoint}"
            )
            self.model = md.vl(endpoint=self.local_endpoint)

            # Test with a simple call
            test_response = self._test_model_connection(self.model)
            if test_response:
                logger.success("Local Moondream model initialized successfully")
                self.use_cloud = False
                return True
        except Exception as e:
            logger.warning(f"Local Moondream model failed: {e}")
        return False

    def _test_model_connection(self, model) -> bool:
        """Test model connection with a minimal request."""
        try:
            # Create a small test image
            test_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
            pil_img = Image.fromarray(test_img)
            response = model.caption(pil_img)
            return response is not None
        except Exception as e:
            logger.debug(f"Model test failed: {e}")
            return False

    def query(self, image: Image.Image, prompt: str) -> Optional[Dict[str, Any]]:
        """Query the model with an image and prompt."""
        if not self.model:
            logger.error("Model not initialized")
            return None

        try:
            response = self.model.query(image, prompt)
            return response
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None

    def detect(self, image: Image.Image, object_type: str) -> Optional[Dict[str, Any]]:
        """Detect objects in the image."""
        if not self.model:
            logger.error("Model not initialized")
            return None

        try:
            response = self.model.detect(image, object_type)
            return response
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return None

    def caption(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """Generate a caption for the image."""
        if not self.model:
            logger.error("Model not initialized")
            return None

        try:
            response = self.model.caption(image)
            return response
        except Exception as e:
            logger.error(f"Caption failed: {e}")
            return None

    @property
    def is_initialized(self) -> bool:
        """Check if the model is initialized."""
        return self.model is not None

    @property
    def endpoint_type(self) -> str:
        """Get the type of endpoint being used."""
        return "cloud" if self.use_cloud else "local"
