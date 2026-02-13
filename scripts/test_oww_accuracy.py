#!/usr/bin/env python3
"""
openWakeWord Standalone Accuracy Test

Connects to a Viam robot, streams raw PCM16 audio from the source microphone,
and runs openWakeWord detection locally with a user-provided .onnx model.
Reports recall and false positive metrics.

Usage:
    python scripts/test_oww_accuracy.py \
      --robot-address localhost:8080 \
      --api-key <key> \
      --api-key-id <key-id> \
      --microphone <mic-component-name> \
      --model path/to/okay_gambit.onnx \
      --threshold 0.5 \
      --duration 60 \
      --mode recall-only
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

import numpy as np
import openwakeword
from openwakeword.model import Model

# Fix macOS Python SSL certificate verification
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# Download openwakeword's bundled preprocessing models if missing
openwakeword.utils.download_models()

from pynput import keyboard
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


def print_results(metrics: dict):
    """Print formatted test results."""
    mode = metrics.get("mode", "mixed")
    print("\n")
    print("=" * 60)
    print(f"  openWakeWord Accuracy Test Results ({mode} mode)")
    print("=" * 60)
    print()
    print(f"  Test Duration:     {metrics['duration_seconds']:.0f} seconds")
    print(f"  Model:             {metrics['model']}")
    print(f"  Threshold:         {metrics['threshold']}")
    print(f"  Microphone:        {metrics['microphone']}")
    print(f"  Tolerance Window:  ±{metrics['tolerance']}s")
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
        recall_pct = metrics["recall"] * 100
        print(f"  Recall:               {recall_pct:5.1f}%")
        print()
    else:
        print("-" * 60)
        print("  RAW COUNTS")
        print("-" * 60)
        print(f"  Wake word attempts (key presses):    {metrics['attempts']}")
        print(f"  Detections by openWakeWord:          {metrics['detections']}")
        print()
        print("-" * 60)
        print("  MATCHED RESULTS")
        print("-" * 60)
        print(
            f"  True Positives:     {metrics['true_positives']:3d}   (correctly detected)"
        )
        print(
            f"  False Positives:    {metrics['false_positives']:3d}   (detected without wake word)"
        )
        print(
            f"  False Negatives:    {metrics['false_negatives']:3d}   (missed detections)"
        )
        print()
        print("-" * 60)
        print("  METRICS")
        print("-" * 60)
        precision_pct = metrics["precision"] * 100
        recall_pct = metrics["recall"] * 100
        print(
            f"  Precision:            {precision_pct:5.1f}%   (of detections, how many were real)"
        )
        print(
            f"  Recall:               {recall_pct:5.1f}%   (of attempts, how many were caught)"
        )
        print()
        print(f"  False Positive Rate:  {metrics['fp_per_minute']:.1f} per minute")
        print(f"  False Negative Rate:  {metrics['fn_rate_percent']:.1f}%")

    print()
    print("=" * 60)
    print()


async def run_test(
    robot_address: str,
    api_key: Optional[str],
    api_key_id: Optional[str],
    microphone_name: str,
    model_path: str,
    threshold: float,
    duration: int,
    tolerance: float,
    mode: str = "mixed",
) -> dict:
    """
    Run the openWakeWord accuracy test.

    Streams raw PCM16 audio from the source microphone and runs
    openWakeWord detection locally.
    """
    test = AccuracyTest()

    # Load the openWakeWord model
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)

    print(f"\nLoading openWakeWord model: {model_path}")
    oww_model = Model(wakeword_models=[model_path], inference_framework="onnx")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"Model loaded. Wake word name: '{model_name}'")

    # Connect to robot
    print(f"\nConnecting to robot at {robot_address}...")

    if api_key and api_key_id:
        opts = RobotClient.Options.with_api_key(
            api_key=api_key,
            api_key_id=api_key_id,
        )
    else:
        opts = RobotClient.Options(
            refresh_interval=0,
            dial_options=DialOptions(insecure=True),
        )

    robot = await RobotClient.at_address(robot_address, opts)
    print("Connected to robot")

    try:
        # Get the source microphone (NOT the wake word filter)
        mic = AudioIn.from_robot(robot, microphone_name)
        print(f"Found microphone: {microphone_name}")

        # Print instructions
        print("\n" + "=" * 60)
        print(f"  openWakeWord ACCURACY TEST — {mode.upper()} MODE")
        print("=" * 60)
        print(f"\n  Duration:  {duration} seconds")
        print(f"  Model:     {model_name}")
        print(f"  Threshold: {threshold}")
        print(f"  Mic:       {microphone_name}")
        print(f"  Tolerance: ±{tolerance}s")
        print("\n  Instructions:")
        if mode == "fp-only":
            print("  - DO NOT say the wake word")
            print("  - Play background conversation / noise")
            print("  - Any detections are false positives")
        elif mode == "recall-only":
            print("  - Press SPACE right as you START saying the wake word")
            print("  - Wait a few seconds between attempts")
            print("  - No background noise — quiet environment")
            print("  - Measures how many attempts are detected")
        else:
            print("  - Press SPACE right as you START saying the wake word")
            print("  - Wait a few seconds between attempts")
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

        # Stream audio and run openWakeWord detection
        try:
            audio_stream = await mic.get_audio("pcm16", duration, 0)

            async for chunk in audio_stream:
                if not test.running:
                    break

                elapsed = time.time() - test.start_time
                if elapsed >= duration:
                    test.running = False
                    break

                audio_data = chunk.audio.audio_data
                if not audio_data:
                    continue

                # Convert raw PCM16 bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # Run openWakeWord prediction
                prediction = oww_model.predict(audio_array)
                score = prediction.get(model_name, 0.0)

                if score >= threshold:
                    test.detections.append(time.time())
                    print(
                        f"  [DETECTED] Wake word at {elapsed:.1f}s "
                        f"(score: {score:.3f})"
                    )
                    subprocess.Popen(
                        ["afplay", "/System/Library/Sounds/Glass.aiff"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    # Reset model state to avoid repeated triggers on the
                    # same utterance
                    oww_model.reset()

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
        metrics["microphone"] = microphone_name
        metrics["model"] = model_name
        metrics["model_path"] = model_path
        metrics["threshold"] = threshold
        metrics["tolerance"] = tolerance
        metrics["mode"] = mode

        print_results(metrics)

        return metrics

    finally:
        await robot.close()


def main():
    """CLI entry point for openWakeWord accuracy testing."""
    parser = argparse.ArgumentParser(
        description="openWakeWord Standalone Accuracy Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recall test — say the wake word 20+ times
  python scripts/test_oww_accuracy.py \\
    --robot-address localhost:8080 \\
    --microphone mic \\
    --model path/to/okay_gambit.onnx \\
    --threshold 0.5 \\
    --duration 120 \\
    --mode recall-only

  # False positive test — play background conversation
  python scripts/test_oww_accuracy.py \\
    --robot-address my-robot.viam.cloud:443 \\
    --api-key <key> \\
    --api-key-id <key-id> \\
    --microphone mic \\
    --model path/to/okay_gambit.onnx \\
    --threshold 0.5 \\
    --duration 300 \\
    --mode fp-only
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
        "--microphone",
        required=True,
        help="Source microphone component name (the raw mic, not the wake word filter)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to openWakeWord .onnx model file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)",
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
        result = asyncio.run(
            run_test(
                robot_address=args.robot_address,
                api_key=args.api_key,
                api_key_id=args.api_key_id,
                microphone_name=args.microphone,
                model_path=args.model,
                threshold=args.threshold,
                duration=args.duration,
                tolerance=args.tolerance,
                mode=args.mode,
            )
        )

        # Save results to JSON
        output_file = args.output or f"oww_accuracy_{int(time.time())}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
