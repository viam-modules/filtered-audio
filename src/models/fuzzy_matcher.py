"""Fuzzy wake word matching using Levenshtein distance via rapidfuzz.

This module provides fuzzy matching for wake word detection. It uses
Levenshtein distance (edit distance) on word-level windows to match
trigger phrases even when they are transcribed with slight variations
(e.g., "hey robot" recognized as "the robot").
"""

import re
from typing import Optional

from rapidfuzz.distance import Levenshtein


class FuzzyWakeWordMatcher:
    """Fuzzy wake word matcher using Levenshtein distance.

    Uses word-boundary matching to prevent partial-word false positives.

    Example:
        >>> matcher = FuzzyWakeWordMatcher(threshold=2)
        >>> matcher.match("the robot say something", "hey robot")
        True  # "the robot" matches "hey robot" with distance 2
    """

    def __init__(self, threshold):
        """Initialize the fuzzy matcher.

        Args:
            threshold: Maximum Levenshtein distance for a match (0-5).
                       0 = exact match only
                       2 = default, handles common transcription errors
                       5 = very lenient
        """
        self.threshold = max(0, threshold)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        - Lowercase
        - Remove punctuation (except apostrophes for contractions)
        - Collapse whitespace
        """
        text = text.lower()
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def match(self, transcript: str, wake_word: str) -> bool:
        """Check if wake word matches the start of transcript.

        Args:
            transcript: The recognized speech text
            wake_word: The wake word to match

        Returns:
            True if wake word matches within threshold distance
        """
        normalized_text = self._normalize_text(transcript)
        normalized_wake = self._normalize_text(wake_word)

        text_words = normalized_text.split()
        wake_words = normalized_wake.split()

        if not text_words or not wake_words:
            return False

        # Try windows of varying sizes around the wake word length
        # This handles word insertions/deletions
        for num_words in range(max(1, len(wake_words) - 1), len(wake_words) + 2):
            if num_words > len(text_words):
                continue

            # Only check windows at the start (wake word should be first)
            window_text = " ".join(text_words[:num_words])
            distance = Levenshtein.distance(normalized_wake, window_text)

            if distance <= self.threshold:
                return True

        return False

    def match_with_details(
        self, transcript: str, wake_word: str
    ) -> Optional[dict]:
        """Check if wake word matches and return match details.

        Args:
            transcript: The recognized speech text
            wake_word: The wake word to match

        Returns:
            Dict with match details if matched, None otherwise.
            Keys: matched_text, distance, remaining_text
        """
        normalized_text = self._normalize_text(transcript)
        normalized_wake = self._normalize_text(wake_word)

        text_words = normalized_text.split()
        wake_words = normalized_wake.split()

        if not text_words or not wake_words:
            return None

        best_match = None
        best_distance = self.threshold + 1

        for num_words in range(max(1, len(wake_words) - 1), len(wake_words) + 2):
            if num_words > len(text_words):
                continue

            window_text = " ".join(text_words[:num_words])
            distance = Levenshtein.distance(normalized_wake, window_text)

            if distance < best_distance:
                best_distance = distance
                best_match = {
                    "matched_text": window_text,
                    "distance": distance,
                    "remaining_text": " ".join(text_words[num_words:]),
                }

        if best_match and best_distance <= self.threshold:
            return best_match

        return None
