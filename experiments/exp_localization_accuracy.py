import numpy as np
import csv
import core

FS = 44100
C = 343.0
D = 0.15

dsp = core.DSPEngine()

def generate_signal(n=4096):
    x = np.zeros(n)
    x[n//2] = 1
    x += 0.02*np.random.randn(n)
    return x.astype(np.float32)

def simulate_mics(signal, angle):
    tau = (D/C)*np.sin(np.deg2rad(angle))
    delay = int(round(tau*FS))

    m0 = signal
    m1 = np.roll(signal, delay)
    m2 = np.roll(signal, int(delay*0.5))

    return m0,m1,m2

results = []

for angle in range(-90,91,10):

    signal = generate_signal()

    m0,m1,m2 = simulate_mics(signal, angle)

    dsp.push(m0,m1,m2)

    if dsp.ready():

        est_angle, conf = dsp.process()

        error = abs(angle - est_angle)

        results.append([angle, est_angle, error])

with open("results/localization_results.csv","w") as f:
    writer = csv.writer(f)
    writer.writerow(["true_angle","estimated_angle","error"])
    writer.writerows(results)

print("Localization experiment completed")