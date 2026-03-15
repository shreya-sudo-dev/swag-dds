import numpy as np
import time
import csv
import core

dsp = core.DSPEngine()

def generate_signal(n=4096):
    x = np.zeros(n)
    x[n//2] = 1
    return x.astype(np.float32)

results = []

for i in range(100):

    sig = generate_signal()

    m0 = sig
    m1 = np.roll(sig,2)
    m2 = np.roll(sig,4)

    dsp.push(m0,m1,m2)

    if dsp.ready():

        start = time.time()

        dsp.process()

        latency = time.time() - start

        results.append(latency)

with open("results/latency_results.csv","w") as f:
    writer = csv.writer(f)
    writer.writerow(["latency"])
    for r in results:
        writer.writerow([r])

print("Latency experiment completed")