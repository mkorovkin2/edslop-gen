"""Tavily API client with retry logic and rate limiting."""

import logging
import httpx
from typing import List, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from .rate_limiter import RateLimitedClient

logger = logging.getLogger(__name__)


class TavilyClient(RateLimitedClient):
    """
    Tavily API client for web search and image search.

    Inherits rate limiting from RateLimitedClient and adds
    retry logic with exponential backoff.
    """

    def __init__(self, api_key: str, max_concurrent: int = 5, max_per_minute: int = 100):
        """
        Initialize Tavily client.

        Args:
            api_key: Tavily API key
            max_concurrent: Maximum concurrent requests
            max_per_minute: Maximum requests per minute
        """
        super().__init__(max_concurrent, max_per_minute)
        self.api_key = api_key
        self.base_url = "https://api.tavily.com"
        logger.info(
            "TavilyClient initialized (max_concurrent=%d, max_per_minute=%d)",
            max_concurrent,
            max_per_minute
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        reraise=True
    )
    async def _search_impl(self, query: str, search_depth: str = "advanced") -> List[Dict[str, Any]]:
        """
        Internal implementation of web search with retry.

        Args:
            query: Search query string
            search_depth: "basic" or "advanced"

        Returns:
            List of search results

        Raises:
            httpx.HTTPStatusError: If API returns error status
            httpx.TimeoutException: If request times out
        """
        logger.debug("Tavily search: query=%r depth=%s", query, search_depth)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "api_key": self.api_key,
                    "search_depth": search_depth,
                    "max_results": 5
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            logger.debug("Tavily search: %d results for query=%r", len(data.get('results', [])), query)
            return data.get('results', [])

    async def search(self, query: str, search_depth: str = "advanced") -> List[Dict[str, Any]]:
        """
        Perform web search with rate limiting.

        Args:
            query: Search query string
            search_depth: "basic" or "advanced"

        Returns:
            List of search results with keys: title, url, content, score

        Example:
            results = await client.search("quantum computing")
            for result in results:
                print(f"{result['title']}: {result['url']}")
        """
        return await self._execute_with_limits(
            self._search_impl(query, search_depth)
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        reraise=True
    )
    async def _search_images_impl(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Internal implementation of image search with retry.

        Args:
            query: Search query string
            max_results: Maximum number of images to return

        Returns:
            List of image results

        Raises:
            httpx.HTTPStatusError: If API returns error status
            httpx.TimeoutException: If request times out
        """
        logger.debug("Tavily image search: query=%r max_results=%d", query, max_results)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "api_key": self.api_key,
                    "search_depth": "basic",  # Basic for images
                    "include_images": True,
                    "max_results": max_results
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            logger.debug("Tavily image search: %d images for query=%r", len(data.get('images', [])), query)
            return data.get('images', [])

    async def search_images(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for images with rate limiting.

        Args:
            query: Search query string
            max_results: Maximum number of images to return (default 10)

        Returns:
            List of image results with keys: url, description

        Example:
            images = await client.search_images("quantum computer diagram")
            for img in images:
                print(f"{img['description']}: {img['url']}")
        """
        return await self._execute_with_limits(
            self._search_images_impl(query, max_results)
        )

    async def batch_search(self, queries: List[str], search_depth: str = "advanced") -> List[List[Dict[str, Any]]]:
        """
        Perform multiple searches concurrently (with rate limiting).

        Args:
            queries: List of search queries
            search_depth: "basic" or "advanced"

        Returns:
            List of result lists, one per query

        Example:
            queries = ["quantum computing", "quantum entanglement"]
            results = await client.batch_search(queries)
        """
        import asyncio
        tasks = [self.search(q, search_depth) for q in queries]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def batch_search_images(self, queries: List[str], max_results: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Perform multiple image searches concurrently (with rate limiting).

        Args:
            queries: List of search queries
            max_results: Maximum images per query

        Returns:
            List of image result lists, one per query

        Example:
            queries = ["quantum circuit", "qubit diagram"]
            all_images = await client.batch_search_images(queries)
        """
        import asyncio
        tasks = [self.search_images(q, max_results) for q in queries]
        return await asyncio.gather(*tasks, return_exceptions=False)
