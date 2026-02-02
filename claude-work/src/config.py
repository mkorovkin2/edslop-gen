"""Configuration management with Pydantic validation."""

import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class Config(BaseSettings):
    """Application configuration with validation."""

    # API Keys (required)
    openai_api_key: str = Field(..., min_length=20, description="OpenAI API key")
    tavily_api_key: str = Field(..., min_length=20, description="Tavily API key")

    # LLM Configuration
    model_name: str = Field(default="gpt-5.2", description="OpenAI model for generation")

    # Script Generation Limits
    script_min_words: int = Field(default=200, ge=100, le=300, description="Minimum script word count")
    script_max_words: int = Field(default=500, ge=300, le=1000, description="Maximum script word count")

    # Image Collection Configuration
    images_min_total: int = Field(default=10, ge=5, le=100, description="Minimum total images to collect")
    images_per_section: int = Field(default=5, ge=2, le=10, description="Image queries per section")

    # Text-to-Speech Configuration
    tts_model: str = Field(default="tts-1-hd", description="OpenAI TTS model")
    tts_voice: str = Field(
        default="alloy",
        description="TTS voice (alloy, echo, fable, onyx, nova, shimmer)"
    )

    # Rate Limiting
    max_concurrent_tavily: int = Field(default=5, ge=1, le=20, description="Max concurrent Tavily calls")
    max_concurrent_openai: int = Field(default=10, ge=1, le=50, description="Max concurrent OpenAI calls")
    max_rate_tavily_per_min: int = Field(default=100, ge=10, le=500, description="Tavily calls per minute")
    max_rate_openai_per_min: int = Field(default=500, ge=10, le=5000, description="OpenAI calls per minute")

    # Output Management
    max_output_age_days: int = Field(default=7, ge=1, le=365, description="Delete outputs older than N days")
    max_output_size_gb: int = Field(default=10, ge=1, le=1000, description="Max total output size in GB")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @field_validator("script_max_words")
    @classmethod
    def validate_word_range(cls, v: int, info) -> int:
        """Ensure max words > min words."""
        script_min = info.data.get("script_min_words")
        if script_min and v <= script_min:
            raise ValueError(f"script_max_words ({v}) must be greater than script_min_words ({script_min})")
        return v

    @field_validator("tts_voice")
    @classmethod
    def validate_tts_voice(cls, v: str) -> str:
        """Validate TTS voice is one of the supported options."""
        valid_voices = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
        if v.lower() not in valid_voices:
            raise ValueError(f"tts_voice must be one of {valid_voices}, got '{v}'")
        return v.lower()

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        """Validate OpenAI API key format."""
        if not v.startswith("sk-"):
            raise ValueError("openai_api_key must start with 'sk-'")
        return v

    @field_validator("tavily_api_key")
    @classmethod
    def validate_tavily_key(cls, v: str) -> str:
        """Validate Tavily API key format."""
        if not v.startswith("tvly-"):
            raise ValueError("tavily_api_key must start with 'tvly-'")
        return v


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Returns:
        Config: Validated configuration object

    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If .env file not found and required vars missing
    """
    # Load .env file if it exists
    load_dotenv()

    # Create and validate config
    try:
        config = Config()
        return config
    except Exception as e:
        raise ValueError(f"Configuration error: {e}\nPlease check your .env file or environment variables.")


# Global config instance (lazy-loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance (singleton pattern).

    Returns:
        Config: Global configuration object
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config
