"""Rate limiting utilities for API clients."""

import asyncio
from typing import Any, Coroutine, TypeVar
from aiolimiter import AsyncLimiter


T = TypeVar('T')


class RateLimitedClient:
    """
    Base class for API clients with rate limiting.

    Provides concurrent request limiting (via semaphore) and
    per-minute rate limiting (via AsyncLimiter).
    """

    def __init__(self, max_concurrent: int = 10, max_per_minute: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_concurrent: Maximum number of concurrent requests
            max_per_minute: Maximum requests per minute
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = AsyncLimiter(max_per_minute, 60)  # max_rate per 60 seconds

        self._stats = {
            "total_requests": 0,
            "concurrent_peak": 0,
            "current_concurrent": 0
        }

    async def _execute_with_limits(self, coro: Coroutine[Any, Any, T]) -> T:
        """
        Execute a coroutine with rate limiting applied.

        Args:
            coro: The coroutine to execute

        Returns:
            The result of the coroutine

        Raises:
            Any exception raised by the coroutine
        """
        async with self.semaphore:
            # Track concurrent requests
            self._stats["current_concurrent"] += 1
            self._stats["concurrent_peak"] = max(
                self._stats["concurrent_peak"],
                self._stats["current_concurrent"]
            )

            try:
                async with self.rate_limiter:
                    self._stats["total_requests"] += 1
                    result = await coro
                    return result
            finally:
                self._stats["current_concurrent"] -= 1

    def get_stats(self) -> dict:
        """
        Get current rate limiting statistics.

        Returns:
            dict: Statistics including total requests and peak concurrency
        """
        return self._stats.copy()

    def reset_stats(self):
        """Reset statistics counters."""
        self._stats = {
            "total_requests": 0,
            "concurrent_peak": 0,
            "current_concurrent": 0
        }
