"""Context management for LLM token limits."""

from typing import List, Dict, Any
import tiktoken
import logging
import re

logger = logging.getLogger(__name__)


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

    def _sentence_split(self, text: str) -> List[str]:
        """Split text into sentences with a simple regex."""
        # Normalize whitespace to improve splitting.
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return []
        # Split on sentence-ending punctuation.
        parts = re.split(r"(?<=[.!?])\s+", normalized)
        return [p.strip() for p in parts if p.strip()]

    def _keyword_set(self, text: str) -> set[str]:
        """Extract a rough keyword set from text."""
        stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "to", "of",
            "in", "on", "for", "with", "without", "by", "at", "from", "as", "is",
            "are", "was", "were", "be", "been", "being", "it", "this", "that", "these",
            "those", "we", "you", "they", "he", "she", "i", "our", "your", "their",
            "its", "can", "could", "should", "would", "will", "may", "might", "must"
        }
        words = re.findall(r"[a-zA-Z0-9_-]+", text.lower())
        return {w for w in words if w not in stopwords and len(w) > 2}

    def _compress_text(
        self,
        text: str,
        max_tokens: int,
        focus_text: str = ""
    ) -> str:
        """
        Compress text by selecting the most relevant sentences.
        This is extractive (no generation) and keeps sentence order.
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        sentences = self._sentence_split(text)
        if not sentences:
            return text

        focus_keywords = self._keyword_set(focus_text)
        scored = []
        for idx, sentence in enumerate(sentences):
            words = self._keyword_set(sentence)
            overlap = len(words & focus_keywords) if focus_keywords else 0
            length = len(sentence)
            # Favor relevance first, then moderate length.
            score = (overlap * 2) + (min(length, 160) / 160)
            scored.append((score, idx, sentence))

        # Select sentences by score until token budget is met.
        scored.sort(key=lambda x: (-x[0], x[1]))
        selected = []
        selected_idx = set()
        running_tokens = 0
        for _, idx, sentence in scored:
            sentence_tokens = len(self.encoding.encode(sentence))
            if running_tokens + sentence_tokens > max_tokens:
                continue
            selected.append((idx, sentence))
            selected_idx.add(idx)
            running_tokens += sentence_tokens
            if running_tokens >= max_tokens:
                break

        if not selected:
            # Fall back to keeping the most relevant sentence only.
            best = scored[0][2]
            return best

        # Preserve original order
        selected.sort(key=lambda x: x[0])
        return " ".join([s for _, s in selected])

    def truncate_context(
        self,
        context: str,
        reserve_tokens: int = 2000,
        focus_text: str = ""
    ) -> str:
        """
        Compress context to fit within token limits.

        Args:
            context: Text to compress
            reserve_tokens: Tokens to reserve for response
            focus_text: Optional focus text to bias compression

        Returns:
            Compressed context
        """
        max_context_tokens = self.max_tokens - reserve_tokens
        tokens = self.encoding.encode(context)
        if len(tokens) <= max_context_tokens:
            return context
        logger.warning(
            "Context exceeds token budget (tokens=%d, max=%d). Compressing.",
            len(tokens),
            max_context_tokens
        )
        return self._compress_text(context, max_context_tokens, focus_text=focus_text)

    def extract_relevant_context(
        self,
        research_data: List[Dict[str, Any]],
        current_section: str,
        max_tokens: int = 1500
    ) -> str:
        """
        Extract most relevant research for current section.

        Simple implementation: return all research, compressing if needed.

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

        tokens = self.encoding.encode(all_research)
        if len(tokens) <= max_tokens:
            return all_research

        logger.warning(
            "Relevant context exceeds max_tokens (tokens=%d, max=%d). Compressing.",
            len(tokens),
            max_tokens
        )
        return self._compress_text(all_research, max_tokens, focus_text=current_section)
