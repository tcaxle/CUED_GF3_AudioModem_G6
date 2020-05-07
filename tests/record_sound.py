"""
A simple script to record the sound from your microphone for three seconds and play it back at you.
"""
import sounddevice as sd

SAMPLE_RATE = 44100
DURATION = 3

samples = int(SAMPLE_RATE * DURATION)

# Record in Mono
myrecording = sd.rec(samples, SAMPLE_RATE, channels=1)
sd.wait()

# Play it back
sd.play(myrecording, SAMPLE_RATE)
sd.wait()
