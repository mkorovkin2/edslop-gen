"""OpenAI API client with retry logic and rate limiting."""

from openai import AsyncOpenAI, OpenAIError
from typing import Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import tiktoken

from .rate_limiter import RateLimitedClient


class OpenAIClient(RateLimitedClient):
    """
    OpenAI API client for text generation and speech synthesis.

    Inherits rate limiting from RateLimitedClient and adds
    retry logic with exponential backoff.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.2",
        max_concurrent: int = 10,
        max_per_minute: int = 500
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use for generation (default: gpt-5.2)
            max_concurrent: Maximum concurrent requests
            max_per_minute: Maximum requests per minute
        """
        super().__init__(max_concurrent, max_per_minute)
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

        # Initialize tiktoken for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # If model not found, use cl100k_base (GPT-4/GPT-3.5)
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(OpenAIError),
        reraise=True
    )
    async def _generate_impl(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> str:
        """
        Internal implementation of text generation with retry.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            system_message: Optional system message

        Returns:
            Generated text

        Raises:
            OpenAIError: If API call fails
            ValueError: If model not found
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            error_msg = str(e).lower()
            # Some models only accept max_tokens; fall back if needed.
            if "unsupported parameter" in error_msg and "max_completion_tokens" in error_msg:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            # Provide clear error for model not found
            if "model" in error_msg and ("not found" in error_msg or "does not exist" in error_msg):
                raise ValueError(
                    f"Model '{self.model}' not available. "
                    f"Please verify MODEL_NAME in your .env file. "
                    f"Error: {e}"
                )
            raise
        except Exception as e:
            # Provide clear error for model not found
            error_msg = str(e).lower()
            if "model" in error_msg and ("not found" in error_msg or "does not exist" in error_msg):
                raise ValueError(
                    f"Model '{self.model}' not available. "
                    f"Please verify MODEL_NAME in your .env file. "
                    f"Error: {e}"
                )
            raise

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate text completion with rate limiting.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            system_message: Optional system message for context

        Returns:
            Generated text

        Example:
            response = await client.generate(
                "Explain quantum computing",
                system_message="You are a technical educator."
            )
        """
        return await self._execute_with_limits(
            self._generate_impl(prompt, max_tokens, temperature, system_message)
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(OpenAIError),
        reraise=True
    )
    async def _generate_speech_impl(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1-hd"
    ) -> bytes:
        """
        Internal implementation of speech generation with retry.

        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model to use

        Returns:
            Audio data as bytes (MP3 format)

        Raises:
            OpenAIError: If API call fails
            ValueError: If text exceeds character limit
        """
        # OpenAI TTS has a 4096 character limit
        if len(text) > 4096:
            raise ValueError(
                f"Text length ({len(text)}) exceeds OpenAI TTS limit of 4096 characters. "
                "Please chunk the text before calling this method."
            )

        response = await self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )

        # Response.content gives us the audio bytes directly
        return response.content

    async def generate_speech(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1-hd"
    ) -> bytes:
        """
        Generate speech from text with rate limiting.

        Args:
            text: Text to convert to speech (max 4096 characters)
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model to use (tts-1 or tts-1-hd)

        Returns:
            Audio data as bytes (MP3 format)

        Example:
            audio = await client.generate_speech(
                "Hello world",
                voice="nova"
            )
            with open("output.mp3", "wb") as f:
                f.write(audio)
        """
        return await self._execute_with_limits(
            self._generate_speech_impl(text, voice, model)
        )

    async def generate_speech_chunks(
        self,
        text_chunks: list[str],
        voice: str = "alloy",
        model: str = "tts-1-hd"
    ) -> list[bytes]:
        """
        Generate speech for multiple text chunks concurrently.

        Args:
            text_chunks: List of text chunks (each max 4096 chars)
            voice: Voice to use
            model: TTS model to use

        Returns:
            List of audio data bytes, one per chunk

        Example:
            chunks = ["First part.", "Second part."]
            audio_chunks = await client.generate_speech_chunks(chunks)
        """
        import asyncio
        tasks = [self.generate_speech(chunk, voice, model) for chunk in text_chunks]
        return await asyncio.gather(*tasks, return_exceptions=False)
