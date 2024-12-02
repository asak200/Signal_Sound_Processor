import pyaudio
import wave

# Parameters
chunk = 1024  # Buffer size
format = pyaudio.paInt16  # 16-bit audio
channels = 1  # Mono
sample_rate = 44100  # 44.1 kHz
duration = 2.1  # Recording duration
output_file = "mystery.wav"

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream
stream = p.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)

print("Recording...")
frames = []

# Record in chunks
for _ in range(0, int(sample_rate / chunk * duration)):
    data = stream.read(chunk)
    frames.append(data)

print("Recording finished!")

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Save to file
with wave.open(output_file, 'wb') as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))

print(f"Audio saved to {output_file}")