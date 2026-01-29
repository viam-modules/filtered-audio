import numpy as np
import onnxruntime as ort
import sounddevice as sd
import resampy
import time

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = "hey_gambit_final.onnx"
THRESHOLD = 0.7
SAMPLE_RATE = 16000
DEBOUNCE_SEC = 1.0
last_trigger_time = 0

# ------------------------------
# LOAD ONNX MODEL
# ------------------------------
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ------------------------------
# WAKE WORD DETECTION
# ------------------------------
def check_wake_word(audio_bytes: bytes, sample_rate: int) -> bool:
    global last_trigger_time

    # PCM16 → float32
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        audio = resampy.resample(audio, sample_rate, SAMPLE_RATE)

    # ONNX expects shape [1, num_samples]
    audio_input = audio[np.newaxis, :].astype(np.float32)

    # Run ONNX model
    score = session.run([output_name], {input_name: audio_input})[0][0]

    # Debounce
    now = time.time()
    if score >= THRESHOLD and (now - last_trigger_time) > DEBOUNCE_SEC:
        last_trigger_time = now
        print(f"Wake word detected! Score: {score:.2f}")
        return True

    return False

# ------------------------------
# AUDIO CALLBACK
# ------------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    check_wake_word(indata.tobytes(), SAMPLE_RATE)

# ------------------------------
# LISTENING LOOP
# ------------------------------
if __name__ == "__main__":
    print("Listening for 'hey gambit'... Press Ctrl+C to stop.")
    try:
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped listening.")

