#!/usr/bin/env python3
"""
Wake Word Accuracy Testing CLI Tool

Connects to a running Viam robot with wake word filter, tracks detections
vs manual key presses, and reports accuracy metrics.

Usage:
    python scripts/accuracy_test.py --robot-address localhost:8080 --duration 120
"""

import argparse
import asyncio
import certifi
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# Fix macOS Python SSL certificate verification
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from pynput import keyboard
from viam.app.viam_client import ViamClient
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.components.audio_in import AudioIn


@dataclass
class AccuracyTest:
    """Tracks wake word detections and key presses for accuracy testing."""

    detections: list[float] = field(default_factory=list)
    key_presses: list[float] = field(default_factory=list)
    running: bool = False
    start_time: float = 0.0
    listener: Optional[keyboard.Listener] = None

    def on_key_press(self, key):
        """Handle key press events."""
        if not self.running:
            return
        if key == keyboard.Key.space:
            self.key_presses.append(time.time())
            elapsed = time.time() - self.start_time
            print(f"  [LOGGED] Key press recorded at {elapsed:.1f}s")

    def start_key_listener(self):
        """Start the keyboard listener in a background thread."""
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def stop_key_listener(self):
        """Stop the keyboard listener."""
        if self.listener:
            self.listener.stop()
            self.listener = None

    def calculate_metrics(self, tolerance: float, duration: float) -> dict:
        """
        Calculate accuracy metrics by matching detections to key presses.

        Args:
            tolerance: Time window in seconds for matching detection to key press
            duration: Test duration in seconds (for rate calculations)

        Returns:
            Dictionary containing all metrics
        """
        matched_detections = set()
        matched_presses = set()

        # Match each detection to the nearest key press within tolerance
        for i, det in enumerate(self.detections):
            best_match = None
            best_distance = float("inf")
            for j, press in enumerate(self.key_presses):
                distance = abs(det - press)
                if distance <= tolerance and distance < best_distance:
                    if j not in matched_presses:
                        best_match = j
                        best_distance = distance
            if best_match is not None:
                matched_detections.add(i)
                matched_presses.add(best_match)

        tp = len(matched_detections)
        fp = len(self.detections) - tp
        fn = len(self.key_presses) - len(matched_presses)

        # Calculate rates
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fp_per_min = (fp / duration) * 60 if duration > 0 else 0.0
        fn_rate = (fn / len(self.key_presses) * 100) if self.key_presses else 0.0

        return {
            "attempts": len(self.key_presses),
            "detections": len(self.detections),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "fp_per_minute": fp_per_min,
            "fn_rate_percent": fn_rate,
            "detection_timestamps": self.detections,
            "keypress_timestamps": self.key_presses,
        }


async def fetch_config_attributes(
    api_key: str, api_key_id: str, robot_part_id: str, component_name: str
) -> dict:
    """Fetch component config attributes from the Viam App API."""
    dial_options = DialOptions.with_api_key(api_key=api_key, api_key_id=api_key_id)
    viam_client = await ViamClient.create_from_dial_options(dial_options)
    try:
        app_client = viam_client.app_client
        part = await app_client.get_robot_part(robot_part_id=robot_part_id)

        config = part.robot_config
        if not config:
            raise ValueError("Robot part has no config")

        components = config.get("components", [])
        component = next(
            (c for c in components if c["name"] == component_name), None
        )
        if component is None:
            raise ValueError(
                f"Component '{component_name}' not found in robot part config"
            )

        attrs = component.get("attributes", {})

        name = input("\n  Approach name (e.g. 'vosk default'): ").strip()

        fetched = {
            "name": name,
            "vosk_model": attrs.get("vosk_model", "defualt"),
            "use_grammar": attrs.get("use_grammar", True),
            "silence_duration_ms": attrs.get("silence_duration_ms", 900),
            "min_speech_ms": attrs.get("min_speech_ms", 300),
            "vosk_grammar_confidence": attrs.get("vosk_grammar_confidence", 0.7),
            "vad_aggressiveness": attrs.get("vad_aggressiveness", 3),
        }

        print("\n  Fetched config from robot:")
        for k, v in fetched.items():
            if k != "name":
                print(f"    {k}: {v}")

        return fetched
    finally:
        viam_client.close()


def print_comparison(all_results: list[dict]):
    """Print a side-by-side comparison table of all approach results."""
    print("\n")
    print("=" * 80)
    print("  APPROACH COMPARISON")
    print("=" * 80)

    # Build column headers from approach names
    names = [r["config"]["name"] for r in all_results]
    col_width = max(16, *(len(n) + 2 for n in names))

    # Header row
    header = f"  {'Metric':<26}"
    for name in names:
        header += f"{name:>{col_width}}"
    print(header)
    print("  " + "-" * (26 + col_width * len(names)))

    # Config rows
    config_keys = [
        ("vosk_model", "vosk_model"),
        ("use_grammar", "use_grammar"),
        ("silence_duration_ms", "silence_duration_ms"),
        ("min_speech_ms", "min_speech_ms"),
        ("vosk_grammar_confidence", "vosk_grammar_confidence"),
    ]
    for label, key in config_keys:
        row = f"  {label:<26}"
        for r in all_results:
            val = str(r["config"][key])
            row += f"{val:>{col_width}}"
        print(row)

    print("  " + "-" * (26 + col_width * len(names)))

    # Metric rows — show what's relevant to the mode
    mode = all_results[0]["metrics"].get("mode", "mixed")
    if mode == "fp-only":
        metric_rows = [
            ("Detections (all FP)", "detections", "d"),
            ("FP / minute", "fp_per_minute", ".1f"),
            ("Duration (s)", "duration_seconds", ".0f"),
        ]
    elif mode == "recall-only":
        metric_rows = [
            ("Attempts", "attempts", "d"),
            ("Detected", "true_positives", "d"),
            ("Missed", "false_negatives", "d"),
            ("Recall %", "recall", ".1%"),
            ("Duration (s)", "duration_seconds", ".0f"),
        ]
    else:
        metric_rows = [
            ("Attempts", "attempts", "d"),
            ("Detections", "detections", "d"),
            ("True Positives", "true_positives", "d"),
            ("False Positives", "false_positives", "d"),
            ("False Negatives", "false_negatives", "d"),
            ("Precision %", "precision", ".1%"),
            ("Recall %", "recall", ".1%"),
            ("FP / minute", "fp_per_minute", ".1f"),
            ("FN rate %", "fn_rate_percent", ".1f"),
            ("Duration (s)", "duration_seconds", ".0f"),
        ]
    for label, key, fmt in metric_rows:
        row = f"  {label:<26}"
        for r in all_results:
            val = r["metrics"][key]
            formatted = f"{val:{fmt}}"
            row += f"{formatted:>{col_width}}"
        print(row)

    print()
    print("=" * 80)
    print()


async def run_test(
    robot_address: str,
    api_key: Optional[str],
    api_key_id: Optional[str],
    component_name: str,
    duration: int,
    tolerance: float,
    config: dict,
    mode: str = "mixed",
) -> dict:
    """
    Run the accuracy test.

    Args:
        robot_address: Robot address (e.g., "localhost:8080")
        api_key: Viam API key (optional for local connections)
        api_key_id: Viam API key ID (optional for local connections)
        component_name: Wake word filter component name
        duration: Test duration in seconds
        tolerance: Detection/keypress matching window in seconds
        config: Config attributes for this approach
        mode: Test mode - "fp-only", "recall-only", or "mixed"

    Returns:
        Dictionary containing config and test metrics
    """
    test = AccuracyTest()

    # Connect to robot
    print(f"\nConnecting to robot at {robot_address}...")

    if api_key and api_key_id:
        opts = RobotClient.Options.with_api_key(
            api_key=api_key,
            api_key_id=api_key_id,
        )
    else:
        # Local connection without authentication
        opts = RobotClient.Options(
            refresh_interval=0,
            dial_options=DialOptions(insecure=True),
        )

    robot = await RobotClient.at_address(robot_address, opts)
    print("Connected to robot")

    try:
        # Get the wake word filter component
        wake_filter = AudioIn.from_robot(robot, component_name)
        print(f"Found component: {component_name}")

        # Print instructions
        print("\n" + "=" * 60)
        print(f"  WAKE WORD ACCURACY TEST — {mode.upper()} MODE")
        print("=" * 60)
        print(f"\n  Duration: {duration} seconds")
        print(f"  Component: {component_name}")
        print(f"  Tolerance: ±{tolerance}s")
        print("\n  Instructions:")
        if mode == "fp-only":
            print("  - DO NOT say the wake word")
            print("  - Play background conversation / noise")
            print("  - Any detections are false positives")
        elif mode == "recall-only":
            print("  - Press SPACE right as you START saying the wake word")
            print("  - Then pause, then say a command")
            print("  - No background noise — quiet environment")
            print("  - Measures how many attempts are detected")
        else:
            print("  - Press SPACE right as you START saying the wake word")
            print("  - Then pause, then say a command")
            print("  - Test will automatically end after the duration")
        print("\n" + "-" * 60)
        print("  Starting test in 3 seconds...")
        await asyncio.sleep(3)
        if mode == "fp-only":
            print("  TEST STARTED - Listening for false positives...")
        else:
            print("  TEST STARTED - Press SPACE after each wake word attempt")
        print("-" * 60 + "\n")

        # Start key listener (not needed for fp-only)
        if mode != "fp-only":
            test.start_key_listener()
        test.running = True
        test.start_time = time.time()

        # Stream audio and track detections
        try:
            # get_audio(codec, duration_seconds, previous_timestamp_ns)
            audio_stream = await wake_filter.get_audio("pcm16", duration, 0)

            # Two-segment detection: the filter yields two segments per wake word:
            #   1. Wake word segment (chunks + empty sentinel)
            #   2. Command segment (chunks + empty sentinel)
            # We only count the first (wake word) sentinel as a detection.
            awaiting_command = False
            last_detection_time = 0.0
            command_timeout = 12.0  # slightly longer than filter's COMMAND_TIMEOUT_S

            async for chunk in audio_stream:
                if not test.running:
                    break

                # Check if we've exceeded duration
                elapsed = time.time() - test.start_time
                if elapsed >= duration:
                    test.running = False
                    break

                # Reset awaiting_command if the filter's command timeout has passed
                if awaiting_command and (time.time() - last_detection_time) > command_timeout:
                    print(f"  [TIMEOUT]  Command timeout at {elapsed:.1f}s")
                    awaiting_command = False

                audio_data = chunk.audio.audio_data

                # Empty chunk signals end of a segment
                if len(audio_data) == 0:
                    if not awaiting_command:
                        # First sentinel = wake word detected — play chime
                        test.detections.append(time.time())
                        last_detection_time = time.time()
                        print(f"  [DETECTED] Wake word at {elapsed:.1f}s — say your command now")
                        subprocess.Popen(
                            ["afplay", "/System/Library/Sounds/Glass.aiff"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        awaiting_command = True
                    else:
                        # Second sentinel = command segment ended
                        print(f"  [COMMAND]  Segment ended at {elapsed:.1f}s")
                        awaiting_command = False

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if "EOF" in str(e):
                print("\n  Stream ended (server closed connection)")
            else:
                raise
        finally:
            test.running = False
            test.stop_key_listener()

        # Calculate actual test duration
        actual_duration = time.time() - test.start_time

        # Calculate metrics
        metrics = test.calculate_metrics(tolerance, actual_duration)
        metrics["duration_seconds"] = actual_duration
        metrics["component"] = component_name
        metrics["tolerance"] = tolerance
        metrics["mode"] = mode

        # Print results
        print_results(metrics)

        return {"config": config, "metrics": metrics}

    finally:
        await robot.close()


def print_results(metrics: dict):
    """Print formatted test results."""
    mode = metrics.get("mode", "mixed")
    print("\n")
    print("=" * 60)
    print(f"  Wake Word Accuracy Test Results ({mode} mode)")
    print("=" * 60)
    print()
    print(f"  Test Duration:     {metrics['duration_seconds']:.0f} seconds")
    print(f"  Component:         {metrics['component']}")
    print(f"  Tolerance Window:  ±{metrics['tolerance']} seconds")
    print()

    if mode == "fp-only":
        print("-" * 60)
        print("  FALSE POSITIVE RESULTS")
        print("-" * 60)
        print(f"  False Positives:      {metrics['detections']}")
        print(f"  FP per minute:        {metrics['fp_per_minute']:.1f}")
        print()
    elif mode == "recall-only":
        print("-" * 60)
        print("  RECALL RESULTS")
        print("-" * 60)
        print(f"  Wake word attempts:   {metrics['attempts']}")
        print(f"  Detected:             {metrics['true_positives']}")
        print(f"  Missed:               {metrics['false_negatives']}")
        recall_pct = metrics['recall'] * 100
        print(f"  Recall:               {recall_pct:5.1f}%")
        print()
    else:
        print("-" * 60)
        print("  RAW COUNTS")
        print("-" * 60)
        print(f"  Wake word attempts (key presses):    {metrics['attempts']}")
        print(f"  Detections by filter:                {metrics['detections']}")
        print()
        print("-" * 60)
        print("  MATCHED RESULTS")
        print("-" * 60)
        print(f"  True Positives:     {metrics['true_positives']:3d}   (correctly detected)")
        print(f"  False Positives:    {metrics['false_positives']:3d}   (detected without wake word)")
        print(f"  False Negatives:    {metrics['false_negatives']:3d}   (missed detections)")
        print()
        print("-" * 60)
        print("  METRICS")
        print("-" * 60)
        precision_pct = metrics['precision'] * 100
        recall_pct = metrics['recall'] * 100
        print(f"  Precision:            {precision_pct:5.1f}%   (of detections, how many were real)")
        print(f"  Recall:               {recall_pct:5.1f}%   (of attempts, how many were caught)")
        print()
        print(f"  False Positive Rate:  {metrics['fp_per_minute']:.1f} per minute")
        print(f"  False Negative Rate:  {metrics['fn_rate_percent']:.1f}%")

    print()
    print("=" * 60)
    print()


async def run_all(args: argparse.Namespace):
    """Run one or more test approaches interactively."""
    all_results: list[dict] = []

    while True:
        config = await fetch_config_attributes(
            api_key=args.api_key,
            api_key_id=args.api_key_id,
            robot_part_id=args.robot_part_id,
            component_name=args.component,
        )

        result = await run_test(
            robot_address=args.robot_address,
            api_key=args.api_key,
            api_key_id=args.api_key_id,
            component_name=args.component,
            duration=args.duration,
            tolerance=args.tolerance,
            config=config,
            mode=args.mode,
        )
        all_results.append(result)

        another = input("\n  Run another approach? (y/n): ").strip().lower()
        if another != "y":
            break

        print("\n  Adjust the robot config now if needed, then press ENTER to continue.")
        input("  Press ENTER when ready...")

    # Print comparison table if multiple runs
    if len(all_results) > 1:
        print_comparison(all_results)


    output_file = args.output or f"accuracy_results_{int(time.time())}.json"
    if len(all_results) == 1:
        output_data = all_results[0]
    else:
        output_data = {"approaches": all_results}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def main():
    """CLI entry point for wake word accuracy testing."""
    parser = argparse.ArgumentParser(
        description="Wake Word Accuracy Testing CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/accuracy_test.py \\
    --robot-address my-robot.viam.cloud:443 \\
    --api-key <key> \\
    --api-key-id <key-id> \\
    --robot-part-id <part-id>

  # Custom test settings
  python scripts/accuracy_test.py \\
    --robot-address my-robot.viam.cloud:443 \\
    --api-key <key> \\
    --api-key-id <key-id> \\
    --robot-part-id <part-id> \\
    --component my-wake-filter \\
    --duration 120 \\
    --tolerance 3.0 \\
    --output results.json
""",
    )

    parser.add_argument(
        "--robot-address",
        required=True,
        help="Robot address (e.g., 'localhost:8080' or 'my-robot.viam.cloud:443')",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VIAM_API_KEY"),
        help="Viam API key (or use VIAM_API_KEY env var)",
    )
    parser.add_argument(
        "--api-key-id",
        default=os.environ.get("VIAM_API_KEY_ID"),
        help="Viam API key ID (or use VIAM_API_KEY_ID env var)",
    )
    parser.add_argument(
        "--robot-part-id",
        required=True,
        help="Robot part ID (used to fetch component config from Viam App)",
    )
    parser.add_argument(
        "--component",
        default="filter",
        help="Wake word filter component name (default: wake-word-filter)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 180)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=4.0,
        help="Detection/keypress matching window in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--mode",
        choices=["mixed", "fp-only", "recall-only"],
        default="mixed",
        help="Test mode: 'fp-only' (background noise, no wake words), "
             "'recall-only' (wake words, quiet environment), "
             "'mixed' (default, both)",
    )
    parser.add_argument(
        "--output",
        help="Save detailed results to JSON file",
    )

    args = parser.parse_args()

    try:
        asyncio.run(run_all(args))
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
