import curses
import time
import sys
import os
import numpy as np
import collections

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import core
from ai.classifier import GunshotClassifier
from io_mod.mic_file_player import FileStream as LaptopStream


NOISE_GATE_THRESHOLD = 1.0 

def draw_radar(stdscr, angle, y, x):
    """Draws a simple ASCII indicator of direction."""
    try:
        stdscr.addstr(y, x,     r"      [ FRONT ]      ", curses.A_BOLD)
        stdscr.addstr(y+1, x,   r"    o           o    ")
        pos = 10 
        marker_line = " " * pos + "X" + " " * (20 - pos)
        stdscr.addstr(y+2, x,   f"   [{marker_line}]   ", curses.color_pair(1))
        stdscr.addstr(y+3, x,   f"      ALERT!      ", curses.color_pair(1) | curses.A_BOLD)
    except curses.error:
        pass 

def main(stdscr):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)   # Alert (Red)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK) # Active (Green)
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)  # UI (Cyan)
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)# Sleep (Yellow)
    
    curses.curs_set(0)
    stdscr.nodelay(True) 

    dsp = core.DSPEngine()
    ai = GunshotClassifier()
    mic = LaptopStream() 
    
    history = collections.deque(maxlen=8)
    last_detection_time = 0
    frame_count = 0
    last_active_time = 0
    system_state = "INIT" 

    try:
        while True:
            max_y, max_x = stdscr.getmaxyx()

            m0, m1, m2 = mic.get_frame()
            
            energy = np.sum(m0**2)
            
            if energy < NOISE_GATE_THRESHOLD:
                system_state = "SLEEP (Power Save)"
                state_color = curses.color_pair(4)
                
                time.sleep(0.05) 
                                
            else:
                system_state = "ACTIVE (Processing)"
                state_color = curses.color_pair(2)
                
                dsp.push(m0, m1, m2)
                is_threat = ai.is_gunshot(m0)
                
                current_angle = 0 
                
                if is_threat:
                    if dsp.ready():
                        result_angle, confident = dsp.process()
                        last_angle = result_angle
                        conf_str = "CONF" if confident else "LOW-CONF"

                        last_detection_time = time.time()
                    
                        history.append(f"[{time.strftime('%H:%M:%S')}] SHOT @ {result_angle}° {conf_str} (E={int(energy)})")
                        time.sleep(1.0)

            
            stdscr.erase()
            try:
                stdscr.addstr(0, 0, "=== SWAG-DDS ===", curses.color_pair(3) | curses.A_BOLD)
                
                stdscr.addstr(1, 0, f"Status: ", curses.color_pair(3))
                stdscr.addstr(f"{system_state}", state_color | curses.A_BOLD)
                stdscr.addstr(f" | Energy: {int(energy)}", curses.color_pair(3))
                
                stdscr.hline(2, 0, '-', max_x)

                if (time.time() - last_detection_time) < 1.0:
                    draw_radar(stdscr, last_angle, 4, 2)
                    stdscr.addstr(9, 2, "!!! THREAT DETECTED !!!", curses.color_pair(1) | curses.A_BLINK)
                else:
                    if energy < NOISE_GATE_THRESHOLD:
                        stdscr.addstr(5, 8, "zZz...", curses.color_pair(4))
                    else:
                        stdscr.addstr(5, 8, "ANALYZING...", curses.color_pair(2))

                if max_x > 50:
                    stdscr.addstr(4, 40, "EVENT LOG:", curses.color_pair(3))
                    for i, log in enumerate(reversed(history)):
                        if 5 + i >= max_y - 2: break 
                        stdscr.addstr(5 + i, 40, log, curses.color_pair(1))

                stdscr.addstr(max_y - 1, 0, "Press 'q' to exit", curses.color_pair(3))

            except curses.error:
                pass
            
            stdscr.refresh()
            
            key = stdscr.getch()
            if key == ord('q'): break
            frame_count += 1

    finally:
        mic.close()

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass