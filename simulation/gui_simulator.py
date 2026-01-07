import sys
import os
import numpy as np
import time

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import core
from ai.classifier import GunshotClassifier


FS = 44100
FRAME = 1024
C = 343.0
D = 0.06



def sectorize(angle):
    if angle <= -60: return "FAR LEFT"
    if angle <= -15: return "LEFT"
    if angle < 15:   return "FRONT"
    if angle < 60:   return "RIGHT"
    return "FAR RIGHT"


def generate_gunshot(n):
    x = np.zeros(n, np.float32)
    x[n//4] = 1.0
    x += 0.02 * np.random.randn(n)
    return x.astype(np.float32)


def simulate_3mic(frame, angle_deg):
    tau = (D / C) * np.sin(np.deg2rad(angle_deg))
    d01 = int(round(tau * FS))
    d02 = int(round(0.5 * d01))

    m0 = frame
    m1 = np.roll(frame, d01)
    m2 = np.roll(frame, d02)

    return m0, m1, m2


class Compass(QWidget):
    def __init__(self):
        super().__init__()
        self.angle = 0

    def set_angle(self, angle):
        self.angle = angle
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        center = w // 2, h // 2
        radius = min(w, h) // 3

        # Circle
        painter.setPen(QPen(Qt.black, 2))
        painter.drawEllipse(center[0]-radius, center[1]-radius,
                             radius*2, radius*2)

        # Arrow
        painter.setPen(QPen(Qt.red, 3))
        rad = np.deg2rad(-self.angle)
        x = center[0] + radius * np.sin(rad)
        y = center[1] - radius * np.cos(rad)
        painter.drawLine(center[0], center[1], int(x), int(y))



class DoAGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3-Mic Gunshot DoA Simulator")
        self.setGeometry(200, 200, 600, 500)

        self.dsp = core.DSPEngine()
        self.detector = GunshotClassifier()

        layout = QVBoxLayout()

        self.angle_label = QLabel("Source Angle: 0°")
        self.angle_label.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-90, 90)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_angle)

        self.estimate_label = QLabel("Estimated: ---")
        self.estimate_label.setAlignment(Qt.AlignCenter)

        self.sector_label = QLabel("Sector: ---")
        self.sector_label.setAlignment(Qt.AlignCenter)

        self.conf_label = QLabel("Confidence: ---")
        self.conf_label.setAlignment(Qt.AlignCenter)

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)

        self.compass = Compass()
        self.compass.setFixedHeight(250)

        self.run_btn = QPushButton("Simulate Gunshot")
        self.run_btn.clicked.connect(self.run_simulation)

        layout.addWidget(self.angle_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.estimate_label)
        layout.addWidget(self.sector_label)
        layout.addWidget(self.conf_label)
        layout.addWidget(self.conf_bar)
        layout.addWidget(self.compass)

        self.setLayout(layout)

    def update_angle(self):
        self.angle_label.setText(f"Source Angle: {self.slider.value()}°")

    def run_simulation(self):
        angle = self.slider.value()

        frame = generate_gunshot(FRAME)
        m0, m1, m2 = simulate_3mic(frame, angle)

        self.dsp.push(m0, m1, m2)

        if self.dsp.ready() and self.detector.is_gunshot(frame):
            est_angle, confident = self.dsp.process()

            self.estimate_label.setText(f"Estimated: {est_angle}°")
            self.sector_label.setText(f"Sector: {sectorize(est_angle)}")
            self.conf_label.setText(
                "Confidence: HIGH" if confident else "Confidence: LOW"
            )
            self.conf_bar.setValue(100 if confident else 30)
            self.compass.set_angle(est_angle)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DoAGUI()
    win.show()
    sys.exit(app.exec_())
