import json
import os
import time
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, Field

load_dotenv()

# Initialize OpenAI client with native OpenAI API key
llm_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Use GPT-4o-mini for vision tasks
VISION_MODEL_NAME = "gpt-4o-mini"


class Response(BaseModel):
    found: bool = Field(False, description="Whether a paper was found")
    text: Optional[Literal["21", "on god", "straight up"]] = Field(
        None, description="The text of the paper"
    )


def _safe_get_content(response: ChatCompletion) -> Optional[Dict[str, Any]]:
    try:
        content = response.choices[0].message.content
        if content:
            return json.loads(content)
        return None
    except Exception as e:
        logger.error(f"Error parsing content: {e}")
        return None


def get_text_from_llm(base64_image: str) -> Optional[Response]:
    """Process an image with GPT-4o-mini to detect ad-libs on paper.

    Args:
        base64_image: Base64 encoded image

    Returns:
        Response object with detection results
    """
    start_time = time.time()
    logger.info("Sending image to GPT-4o-mini for text detection")

    try:
        response = llm_client.chat.completions.create(
            model=VISION_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized text detection system for a 21 Savage rhythm game. You analyze images to identify text on papers that people are holding.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this image and identify if there's a piece of paper with text on it.
                            The only valid texts you should identify are: "21", "on god", or "straight up".
                            If you see one of these texts (or something very close to them), return it.
                            If you're unable to read the text or don't see a paper, or see text that doesn't match these options, return found=False.
                            """,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": Response.__name__,
                    "schema": Response.model_json_schema(),
                },
            },
            max_tokens=300,
        )

        parsed_response = _safe_get_content(response)
        elapsed = time.time() - start_time
        logger.info(f"GPT-4o-mini API call completed in {elapsed:.2f} seconds")

        if parsed_response:
            # Convert to our Response model to validate structure
            try:
                result = Response(**parsed_response)
                logger.info(f"Detected: {result}")
                return result
            except Exception as e:
                logger.error(f"Failed to parse response into valid format: {e}")
                return Response(found=False, text=None)
        else:
            logger.error("Failed to get valid response from API")
            return Response(found=False, text=None)

    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return Response(found=False, text=None)
