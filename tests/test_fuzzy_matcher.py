import pytest
from src.models.fuzzy_matcher import FuzzyWakeWordMatcher


class TestFuzzyWakeWordMatcher:
    """Tests for FuzzyWakeWordMatcher class."""

    def test_exact_match(self):
        """Test exact match returns True with distance 0."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        assert matcher.match("hey robot turn on lights", "hey robot") is True

    def test_fuzzy_match_the_robot(self):
        """Test 'the robot' matches 'hey robot' (common transcription error)."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        assert matcher.match("the robot say something", "hey robot") is True

    def test_fuzzy_match_a_robot(self):
        """Test 'a robot' matches 'hey robot' (distance 3)."""
        matcher = FuzzyWakeWordMatcher(threshold=3)
        assert matcher.match("a robot turn on lights", "hey robot") is True
        # With threshold 2, should not match
        matcher_strict = FuzzyWakeWordMatcher(threshold=2)
        assert matcher_strict.match("a robot turn on lights", "hey robot") is False

    def test_fuzzy_match_hey_robert(self):
        """Test 'hey Robert' matches 'hey robot' (distance 1)."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        assert matcher.match("hey robert what time is it", "hey robot") is True

    def test_no_match_too_different(self):
        """Test 'they robotic' does not match 'hey robot' (too far)."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        assert matcher.match("they robotic assistant", "hey robot") is False

    def test_no_match_wake_word_not_at_start(self):
        """Test wake word in middle of transcript doesn't match."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        assert matcher.match("please hey robot do something", "hey robot") is False

    def test_single_word_wake_word(self):
        """Test single word wake word matching."""
        matcher = FuzzyWakeWordMatcher(threshold=1)
        assert matcher.match("robot turn on", "robot") is True
        assert matcher.match("robut turn on", "robot") is True  # typo
        assert matcher.match("robert turn on", "robot") is False  # distance 2

    def test_threshold_0_exact_only(self):
        """Test threshold 0 only allows exact matches."""
        matcher = FuzzyWakeWordMatcher(threshold=0)
        assert matcher.match("hey robot do something", "hey robot") is True
        assert matcher.match("the robot do something", "hey robot") is False

    def test_empty_transcript(self):
        """Test empty transcript returns False."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        assert matcher.match("", "hey robot") is False

    def test_empty_wake_word(self):
        """Test empty wake word returns False."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        assert matcher.match("hey robot", "") is False

    def test_case_insensitive(self):
        """Test matching is case insensitive."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        assert matcher.match("HEY ROBOT turn on", "hey robot") is True
        assert matcher.match("Hey Robot turn on", "HEY ROBOT") is True

    def test_match_with_details_returns_info(self):
        """Test match_with_details returns match information."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        result = matcher.match_with_details("the robot say something", "hey robot")

        assert result is not None
        assert result["matched_text"] == "the robot"
        assert result["distance"] == 2
        assert result["remaining_text"] == "say something"

    def test_match_with_details_exact_match(self):
        """Test match_with_details with exact match."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        result = matcher.match_with_details("hey robot turn on lights", "hey robot")

        assert result is not None
        assert result["matched_text"] == "hey robot"
        assert result["distance"] == 0
        assert result["remaining_text"] == "turn on lights"

    def test_match_with_details_no_match(self):
        """Test match_with_details returns None when no match."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        result = matcher.match_with_details("hello world how are you", "hey robot")

        assert result is None

    def test_match_with_details_empty_remaining(self):
        """Test match_with_details when wake word is entire transcript."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        result = matcher.match_with_details("hey robot", "hey robot")

        assert result is not None
        assert result["matched_text"] == "hey robot"
        assert result["remaining_text"] == ""

    def test_multi_word_wake_word(self):
        """Test multi-word wake words."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        assert matcher.match("ok computer play music", "ok computer") is True
        assert matcher.match("okay computer play music", "ok computer") is True

    def test_word_boundary_prevents_partial_match(self):
        """Test word boundaries prevent matching partial words."""
        matcher = FuzzyWakeWordMatcher(threshold=2)
        # "robotics" should not match "robot" at start
        assert matcher.match("robotics is cool", "robot") is False
