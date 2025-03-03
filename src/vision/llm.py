import os
import time
from typing import Literal, Optional

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, Field

load_dotenv()

llm_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_API_BASE_URL"),
)

VISION_MODEL_NAME = "llama-3.2-11b-vision-preview"
FIXER_MODEL_NAME = "llama-3.2-3b-preview"


class Response(BaseModel):
    found: bool = Field(False, description="Whether a paper was found")
    text: Optional[Literal["21", "On God", "straight up"]] = Field(
        None, description="The text of the paper"
    )


def _safe_get_content(response: ChatCompletion) -> Optional[str]:
    try:
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting content: {e}")
        return None


def get_text_from_llm(base64_image: str) -> Optional[str]:
    start_time = time.time()

    raw_response = llm_client.chat.completions.create(
        model=VISION_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Please analyze this image. There should be a human holding a piece of paper, and your job is to identify what it says. If you're unable to read the text or find a paper, just return found=False.
                        Please respond in valid JSON format per the following schema:
                        {Response.model_json_schema()}
                        """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )
    raw_response_content = _safe_get_content(raw_response)
    if raw_response_content is None:
        logger.error("No content found in raw response")
        return None

    fixer_response = llm_client.chat.completions.create(
        model=FIXER_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": f"You will be given a JSON response. Your job is to fix it to conform to the following schema: {Response.model_json_schema()}.",
            },
            {
                "role": "user",
                "content": raw_response_content,
            },
        ],
        response_format={
            "type": "json_object",
        },
    )
    fixer_response_content = _safe_get_content(fixer_response)
    if fixer_response_content is None:
        logger.error("No content found in fixer response")
        return None

    elapsed = time.time() - start_time
    logger.info(f"Initial API call completed in {elapsed:.2f} seconds")
    results = fixer_response_content
    logger.info(f"Results: {results}")
    return results
