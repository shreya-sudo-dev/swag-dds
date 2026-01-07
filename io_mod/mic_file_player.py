import wave
import numpy as np

CHUNK = 1024

class FileStream:
    def __init__(self, filename="mission_simulation.wav"):
        self.wf = wave.open(filename, 'rb')
        print(f"Playing scenario: {filename}")

    def get_frame(self):
        data = self.wf.readframes(CHUNK)
        if len(data) < CHUNK * 3 * 2: # End of file
            self.wf.rewind()
            data = self.wf.readframes(CHUNK)
            
        raw = np.frombuffer(data, dtype=np.int16)
        
        c0 = raw[0::3].astype(np.float32) / 32768.0
        c1 = raw[1::3].astype(np.float32) / 32768.0
        c2 = raw[2::3].astype(np.float32) / 32768.0
        
        c0 *= 100.0
        c1 *= 100.0
        c2 *= 100.0

        return c0, c1, c2

    def close(self):
        self.wf.close()