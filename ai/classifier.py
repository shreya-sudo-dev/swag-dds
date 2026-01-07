import numpy as np

class GunshotClassifier:
    def __init__(self):
        
        self.w_energy   = 0.00116704
        self.w_zcr      = 0.00107874
        self.w_peak     = 0.00208320
        self.w_centroid = -0.99290136
        self.bias       = 0.0
        
        
        self.fs = 44100 

    def is_gunshot(self, audio: np.ndarray) -> bool:

        max_val = np.max(np.abs(audio)) + 1e-6
        audio = audio.astype(np.float32) / max_val

        
        energy = np.sum(audio**2)
        zcr = ((audio[:-1] * audio[1:]) < 0).sum()
        peak = np.max(np.abs(audio))
        
        
        spectrum = np.abs(np.fft.rfft(audio))
        
        
        frequencies = np.linspace(0, self.fs/2, len(spectrum))
        
        sum_mag = np.sum(spectrum) + 1e-6
        centroid = np.sum(frequencies * spectrum) / sum_mag
        
        centroid_norm = centroid / (self.fs/2)

        score = (
            (self.w_energy   * energy) + 
            (self.w_zcr      * zcr) + 
            (self.w_peak     * peak) + 
            (self.w_centroid * centroid_norm) + 
            self.bias
        )

        return score > 0