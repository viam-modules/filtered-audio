"""
Voice assistant using Google Speech-to-Text, Gemini, and Google TTS.

Requires: pip install google-genai SpeechRecognition gTTS
Set environment variable: GOOGLE_API_KEY
Generate an API key at https://aistudio.google.com/app/u/1/api-keys
"""

import asyncio
import os

from viam.robot.client import RobotClient
from viam.components.audio_in import AudioIn, AudioCodec
from viam.components.audio_out import AudioOut, AudioInfo
import speech_recognition as sr
from gtts import gTTS
from google import genai


class GeminiVoiceAssistant:
    """Voice assistant powered by Google Cloud services."""

    def __init__(self, robot: RobotClient, filter_name: str = "filter", audioout_name: str = "speaker"):
        self.robot = robot
        self.filter_name = filter_name
        self.audioout_name = audioout_name
        self.recognizer = sr.Recognizer()
        self.tts_lang = 'en'
        self.filter = None
        self.audioout = None

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.system_prompt = "You are a helpful voice assistant. Keep responses concise and conversational."
        self.chat_history = []


    async def start(self):
        self.filter = AudioIn.from_robot(self.robot, self.filter_name)
        self.audioout = AudioOut.from_robot(self.robot, self.audioout_name)
        print(f"Connected to filtered microphone: {self.filter_name}")
        print(f"Connected to speaker: {self.audioout_name}")


    def speech_to_text(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Convert audio to text."""
        audio = sr.AudioData(audio_data, sample_rate, 2)
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except:
            return ""

    def get_response(self, user_text: str) -> str:
        """Generate response using Gemini."""
        if not user_text:
            return "I didn't catch that."

        try:
            # Build conversation context
            messages = [self.system_prompt] + self.chat_history + [f"User: {user_text}"]

            # Send message to Gemini
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents='\n'.join(messages)
            )

            # Update chat history
            self.chat_history.append(f"User: {user_text}")
            self.chat_history.append(f"Assistant: {response.text}")

            return response.text
        except Exception as e:
            print(f"Error getting Gemini response: {e}")
            return "Sorry, I had trouble processing that."

    async def speak(self, text: str):
        """Text to speech."""
        temp_mp3 = "/tmp/tts.mp3"
        gTTS(text=text, lang=self.tts_lang).save(temp_mp3)

        with open(temp_mp3, 'rb') as f:
            mp3_data = f.read()
        os.remove(temp_mp3)

        audio_info = AudioInfo(codec=AudioCodec.MP3)
        await self.audioout.play(mp3_data, audio_info)

    async def run(self):
        """Continuously listen and respond."""
        print("Listening for wake word...")

        # Start continuous stream
        try:
            audio_stream = await self.filter.get_audio("pcm16", 0, 0)
            print("Audio stream started successfully")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            return

        try:
            segment = bytearray()

            async for chunk in audio_stream:
                audio_data = chunk.audio.audio_data

                if len(audio_data) == 0:
                    # Empty chunk = segment ended, process it
                    if segment:
                        print(f"\nWake word detected! Processing {len(segment)} bytes...")

                        user_text = self.speech_to_text(bytes(segment))

                        if user_text:
                            print(f"You: {user_text}")
                            response_text = self.get_response(user_text)
                            print(f"Bot: {response_text}")
                            await self.speak(response_text)
                        else:
                            print("No speech recognized")

                        segment.clear()
                        print("Listening for next wake word...\n")
                else:
                    # Accumulate audio data
                    segment.extend(audio_data)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()


async def main():
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Set it with: export GOOGLE_API_KEY='your-api-key'")
        return


    opts = RobotClient.Options.with_api_key(
        api_key='yn2v121yklqjfduvse718fn582dskzi1',
        api_key_id='d04e49b3-7799-4afe-ba3a-a5b35d802b17'
    )


    robot = await RobotClient.at_address('xarm-main.aqb785vhl4.viam.cloud', opts)

    try:
        assistant = GeminiVoiceAssistant(robot, "filter", "speaker")
        await assistant.start()
        await assistant.run()
    finally:
        await robot.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped by user")
