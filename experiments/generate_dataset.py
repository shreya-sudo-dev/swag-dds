import numpy as np
import os

os.makedirs("dataset/gunshot", exist_ok=True)
os.makedirs("dataset/noise", exist_ok=True)

N = 1000
FRAME = 4096

def generate_gunshot():
    x = np.zeros(FRAME)
    center = FRAME // 2
    x[center] = 1.0
    decay = np.exp(-np.linspace(0,5,FRAME-center))
    x[center:] += decay * 0.4
    x += 0.05*np.random.randn(FRAME)
    return x.astype(np.float32)

def generate_noise():
    x = 0.3*np.random.randn(FRAME)

    # add random bursts
    idx = np.random.randint(0, FRAME - 10)
    x[idx:idx+10] += np.random.randn(10)*0.5

    # add low frequency hum
    t = np.linspace(0,1,FRAME)
    x += 0.2*np.sin(2*np.pi*60*t)

    return x.astype(np.float32)

def generate_impulse_noise():
    x = np.zeros(FRAME)

    idx = np.random.randint(0, FRAME)
    x[idx] = np.random.uniform(0.4, 1.0)

    x += 0.05*np.random.randn(FRAME)

    return x.astype(np.float32)

print("Generating dataset...")

for i in range(N):

    g = generate_gunshot()
    np.save(f"dataset/gunshot/g_{i}.npy", g)

    n = generate_noise()
    np.save(f"dataset/noise/n_{i}.npy", n)

    imp = generate_impulse_noise()
    np.save(f"dataset/noise/i_{i}.npy", imp)

print("Dataset created:", N*2, "samples")