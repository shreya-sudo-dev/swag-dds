import numpy as np
import wave
import struct

FS = 44100
D = 0.15
C = 343.0

def create_bang(angle):
    n = 4096
    
    noise = np.random.randn(n) * 0.1  
    sig = noise.astype(np.float32)
    
    center_idx = n // 2
    sig[center_idx] = 1.0 
    
    t = np.linspace(0, 1, n // 2)
    decay = np.exp(-10 * t)
    
    sig[center_idx:] += decay * 0.5 

    tau = (D / C) * np.sin(np.deg2rad(angle))
    d1 = int(round(tau * FS))
    d2 = int(0.5 * d1)
    
    c0 = sig
    c1 = np.roll(sig, d1)
    c2 = np.roll(sig, d2)
    
    return c0, c1, c2

def create_silence(seconds):
    n = int(FS * seconds)
    z = np.zeros(n, dtype=np.float32)
    return z, z, z

print("Generating 'mission_simulation.wav'...")

audio_c0 = []
audio_c1 = []
audio_c2 = []

s0, s1, s2 = create_silence(2.0)
audio_c0.extend(s0); audio_c1.extend(s1); audio_c2.extend(s2)

b0, b1, b2 = create_bang(-45)
audio_c0.extend(b0); audio_c1.extend(b1); audio_c2.extend(b2)

s0, s1, s2 = create_silence(3.0)
audio_c0.extend(s0); audio_c1.extend(s1); audio_c2.extend(s2)

b0, b1, b2 = create_bang(60)
audio_c0.extend(b0); audio_c1.extend(b1); audio_c2.extend(b2)

full_c0 = np.array(audio_c0)
full_c1 = np.array(audio_c1)
full_c2 = np.array(audio_c2)

interleaved = np.empty((full_c0.size + full_c1.size + full_c2.size,), dtype=np.float32)
interleaved[0::3] = full_c0
interleaved[1::3] = full_c1
interleaved[2::3] = full_c2

interleaved = (interleaved * 32767).astype(np.int16)

with wave.open("mission_simulation.wav", "wb") as f:
    f.setnchannels(3)
    f.setsampwidth(2) 
    f.setframerate(FS)
    f.writeframes(interleaved.tobytes())

print("Done! File saved: mission_simulation.wav")