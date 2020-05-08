"""
A simple script to record the sound from your microphone for three seconds and play it back at you.
"""
import sounddevice as sd

def record_sound(sample_rate=44100, duration=1):
    samples = int(sample_rate * duration)
    print("RECORDING")
    sound = sd.rec(samples, sample_rate, channels=1)
    sd.wait()
    return sound

def play_sound(sound, sample_rate=44100):
    print("PLAYING")
    sd.play(sound, sample_rate)
    sd.wait()

play_sound(record_sound(44100, 5), 44100)
