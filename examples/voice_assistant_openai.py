"""
Voice assistant using OpenAI Whisper (STT), GPT-4 (completion), and OpenAI TTS.

Requires: pip install openai
Set environment variable: OPENAI_API_KEY
"""

import asyncio
import os
import wave
from io import BytesIO

from viam.robot.client import RobotClient
from viam.components.audio_in import AudioIn, AudioCodec
from viam.components.audio_out import AudioOut, AudioInfo
from openai import OpenAI


class OpenAIVoiceAssistant:
    """Voice assistant powered entirely by OpenAI."""

    def __init__(self, robot: RobotClient, trigger_name: str = "filter", audioout_name: str = "speaker"):
        self.robot = robot
        self.trigger_name = trigger_name
        self.audioout_name = audioout_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.trigger = None
        self.audioout = None
        self.conversation_history = []

    async def start(self):
        """Initialize robot components."""
        self.trigger = AudioIn.from_robot(self.robot, self.trigger_name)
        self.audioout = AudioOut.from_robot(self.robot, self.audioout_name)
        print(f"Connected to filtered microphone: {self.trigger_name}")
        print(f"Connected to speaker: {self.audioout_name}")

    def speech_to_text(self, audio_data: bytes) -> str:
        """Convert audio to text using OpenAI Whisper."""
        try:
            # Create WAV file in memory, openAI expects a .wav file
            audio_file = BytesIO()
            with wave.open(audio_file, 'wb') as wav:
                wav.setnchannels(1)  # mono
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(16000)  # 16kHz
                wav.writeframes(audio_data)

            audio_file.seek(0)
            audio_file.name = "audio.wav"

            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcript.text
        except Exception as e:
            print(f"Error in speech-to-text: {e}")
            return ""

    def get_completion(self, user_text: str) -> str:
        """Get GPT-4 completion."""
        if not user_text:
            return "I didn't catch that."

        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_text})

            # Get completion
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise and conversational."},
                    *self.conversation_history
                ],
                max_tokens=150,
                temperature=0.7
            )

            assistant_message = response.choices[0].message.content

            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            # Keep only last 10 messages to avoid token limits
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return assistant_message
        except Exception as e:
            print(f"Error getting completion: {e}")
            return "Sorry, I had trouble processing that."

    async def speak(self, text: str):
        """Text to speech using OpenAI TTS."""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
                input=text
            )

            # Get audio bytes
            audio_bytes = response.content

            # OpenAI returns MP3 by default
            audio_info = AudioInfo(codec=AudioCodec.MP3, sample_rate_hz=24000, num_channels=1)
            await self.audioout.play(audio_bytes, audio_info)
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    async def run(self):
        """Continuously listen and respond."""
        print("Listening for wake word...")

        audio_stream = await self.trigger.get_audio("pcm16", 0, 0)
        audio_buffer = bytearray()
        stream_timeout = 0.5  # seconds to wait for next chunk before processing

        try:
            while True:
                try:
                    # Wait for next chunk with timeout
                    response = await asyncio.wait_for(audio_stream.__anext__(), timeout=stream_timeout)
                    audio_chunk = response.audio.audio_data

                    if len(audio_chunk) == 0:
                        continue

                    # First chunk of a new speech segment
                    if len(audio_buffer) == 0:
                        print("\nWake word detected! Collecting audio...")

                    audio_buffer.extend(audio_chunk)

                except asyncio.TimeoutError:
                    # No chunks arrived within timeout - speech segment ended
                    if len(audio_buffer) > 0:
                        print(f"Speech segment ended. Processing {len(audio_buffer)} bytes...")

                        user_text = self.speech_to_text(bytes(audio_buffer))

                        if user_text:
                            print(f"You: {user_text}")

                            # Get GPT-4 response
                            response_text = self.get_completion(user_text)
                            print(f"Assistant: {response_text}")

                            # Speak response
                            await self.speak(response_text)
                        else:
                            print("No speech recognized")

                        # Clear buffer and continue listening
                        audio_buffer.clear()
                        print("\nListening for next trigger...\n")

                except StopAsyncIteration:
                    # Stream ended
                    break

        except KeyboardInterrupt:
            print("\n\nStopping...")


async def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return

    opts = RobotClient.Options.with_api_key(
        api_key='<API_KEY>',
        api_key_id='<API_KEY_ID>'
    )
    robot = await RobotClient.at_address('<ADDRESS>', opts)

    try:
        assistant = OpenAIVoiceAssistant(robot, "filter", "speaker")
        await assistant.start()
        await assistant.run()
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
