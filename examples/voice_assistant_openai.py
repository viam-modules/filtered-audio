"""
Voice assistant using OpenAI Whisper, GPT, and OpenAI TTS.

Requires: pip install viam-sdk openai
Set environment variable: OPENAI_API_KEY
generate openAI key: https://platform.openai.com/api-keys
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
    """Voice assistant powered by OpenAI services."""

    def __init__(
        self,
        robot: RobotClient,
        filter_name: str = "filter",
        audioout_name: str = "speaker",
    ):
        self.robot = robot
        self.filter_name = filter_name
        self.audioout_name = audioout_name
        self.filter = None
        self.audioout = None

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = (
            "You are a helpful voice assistant. "
            "Keep responses concise and conversational."
        )
        self.chat_history = []

    async def start(self):
        self.filter = AudioIn.from_robot(self.robot, self.filter_name)
        self.audioout = AudioOut.from_robot(self.robot, self.audioout_name)
        print(f"Connected to filtered microphone: {self.filter_name}")
        print(f"Connected to speaker: {self.audioout_name}")

    def speech_to_text(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Convert audio to text using Whisper."""
        # Add WAV header for raw PCM data
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"
        response = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_buffer,
        )
        return response.text

    def get_response(self, user_text: str) -> str:
        """Generate response using GPT."""
        if not user_text:
            return "I didn't catch that."

        try:
            # Build messages for chat completion
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.chat_history)
            messages.append({"role": "user", "content": user_text})

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )

            assistant_message = response.choices[0].message.content

            # Update chat history
            self.chat_history.append({"role": "user", "content": user_text})
            self.chat_history.append({"role": "assistant", "content": assistant_message})

            return assistant_message
        except Exception as e:
            print(f"Error getting GPT response: {e}")
            return "Sorry, I had trouble processing that."

    async def speak(self, text: str):
        """Text to speech using OpenAI TTS."""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
            )
            mp3_data = response.content
            audio_info = AudioInfo(codec=AudioCodec.MP3)
            await self.audioout.play(mp3_data, audio_info)
        except Exception as e:
            print(f"Error in text to speech: {e}")

    async def run(self):
        """Continuously listen and respond."""
        print("Listening for wake word...")

        while True:
            # Start continuous stream
            try:
                audio_stream = await self.filter.get_audio("pcm16", 0, 0)
            except Exception as e:
                print(f"Error starting audio stream: {e}, retrying...")
                await asyncio.sleep(1)
                continue

            try:
                segment = bytearray()

                async for chunk in audio_stream:
                    audio_data = chunk.audio.audio_data

                    if len(audio_data) == 0:
                        # Empty chunk = segment ended, process it
                        if segment:
                            print(f"\nWake word detected! Processing {len(segment)} bytes...")
                            try:
                                user_text = self.speech_to_text(bytes(segment))
                                if user_text:
                                    print(f"You: {user_text}")
                                    response_text = self.get_response(user_text)
                                    print(f"Bot: {response_text}")
                                    await self.speak(response_text)
                                else:
                                    print("No speech recognized")
                            except Exception as e:
                                print(f"Error processing speech: {e}")

                            segment.clear()
                            print("Listening for next wake word...\n")
                    else:
                        # Accumulate audio data
                        segment.extend(audio_data)

            except KeyboardInterrupt:
                print("\n\nStopping...")
                return
            except Exception as e:
                print(f"Stream disconnected: {e}, reconnecting...")
                await asyncio.sleep(1)
                continue


async def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return

    opts = RobotClient.Options.with_api_key(
        api_key='',
        api_key_id=''
    )
    robot = await RobotClient.at_address('', opts)

    try:
        assistant = OpenAIVoiceAssistant(robot, "filter", "speaker")
        await assistant.start()
        await assistant.run()
    finally:
        await robot.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped by user")
