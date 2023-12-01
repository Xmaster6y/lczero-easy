"""
Global API Schema
"""

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    error: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "description of the error",
                }
            ],
        }
    }


class SuccessDetail(BaseModel):
    success: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": "description of the success",
                }
            ],
        }
    }
