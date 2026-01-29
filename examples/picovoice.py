import pvporcupine
import sounddevice as sd
import struct
import librosa
import numpy as np

ACCESS_KEY = "5SJHeQ9iLZulbX0VnXniB9rsNC07DqZ00dVZiXXvtdayasJjUoCRYg=="
KEYWORD_PATH = "/home/viam/Hey-Gambit.ppn"

porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=[KEYWORD_PATH],
)

def audio_callback(indata, frames, time, status):
    # Resample to porcupine.sample_rate
    audio = indata[:, 0].astype(np.float32)
    audio_resampled = librosa.resample(audio, orig_sr=44100, target_sr=porcupine.sample_rate)
    # convert to int16 PCM
    pcm = (audio_resampled * 32767).astype(np.int16).tobytes()
    pcm_tuple = struct.unpack_from("h" * porcupine.frame_length, pcm)
    if porcupine.process(pcm_tuple) >= 0:
        print("WAKE WORD DETECTED")

with sd.InputStream(
    samplerate=44100,
    blocksize=porcupine.frame_length,
    channels=1,
    dtype="int16",
    callback=audio_callback,
):
    print("Listening...")
    while True:
        pass
