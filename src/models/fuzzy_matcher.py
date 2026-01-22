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
    """

    def __init__(self, threshold: int):
        """Initialize the fuzzy matcher.

        Args:
            threshold: Maximum Levenshtein distance for a match (0-5).
                       0 = exact match only
                       2 = handles common transcription errors
                       5 = very lenient
        """
        self.threshold = threshold

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

    def match(self, transcript: str, wake_word: str) -> Optional[dict]:
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

        # Try matching the wake phrase against a window of words
        # that is roughly the same length as the wake phrase.
        # We allow:
        #   - one word fewer
        #   - the exact number of words
        #   - one extra word
        for num_words in range(max(1, len(wake_words) - 1), len(wake_words) + 2):
            if num_words > len(text_words):
                continue

            window_text = " ".join(text_words[:num_words])
            distance = Levenshtein.distance(normalized_wake, window_text)

            # If we found a smaller distance, store it.
            if distance < best_distance:
                best_distance = distance
                best_match = {
                    "matched_text": window_text,
                    "distance": distance,
                    "remaining_text": " ".join(text_words[num_words:]),
                }
            if distance == 0:
                break

        if best_match and best_distance <= self.threshold:
            return best_match

        return None
