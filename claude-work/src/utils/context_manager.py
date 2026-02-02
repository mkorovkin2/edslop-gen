"""Context management for LLM token limits."""

from typing import List, Dict, Any
import tiktoken


class ContextManager:
    """Manages LLM context to stay within token limits."""

    def __init__(self, model: str = "gpt-5.2", max_tokens: int = 8000):
        """
        Initialize context manager.

        Args:
            model: Model name for token encoding
            max_tokens: Maximum tokens allowed
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def truncate_context(
        self,
        context: str,
        reserve_tokens: int = 2000
    ) -> str:
        """
        Truncate context to fit within token limits.

        Keeps beginning and end, removes middle content.

        Args:
            context: Text to truncate
            reserve_tokens: Tokens to reserve for response

        Returns:
            Truncated context
        """
        tokens = self.encoding.encode(context)
        max_context_tokens = self.max_tokens - reserve_tokens

        if len(tokens) <= max_context_tokens:
            return context

        # Keep beginning and end, truncate middle
        keep_tokens = max_context_tokens // 2
        truncated = tokens[:keep_tokens] + tokens[-keep_tokens:]

        return self.encoding.decode(truncated)

    def extract_relevant_context(
        self,
        research_data: List[Dict[str, Any]],
        current_section: str,
        max_tokens: int = 1500
    ) -> str:
        """
        Extract most relevant research for current section.

        Simple implementation: take first N tokens of research.
        For production, could use LLM-based extraction.

        Args:
            research_data: List of research results
            current_section: Current section text
            max_tokens: Maximum tokens to extract

        Returns:
            Relevant context string
        """
        # Combine all research into single string
        all_research = "\n\n".join([
            f"{item.get('title', '')}: {item.get('content', '')}"
            for item in research_data
        ])

        # Truncate to max_tokens
        tokens = self.encoding.encode(all_research)
        if len(tokens) <= max_tokens:
            return all_research

        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
