#!/usr/bin/env python3
"""Reset an ESP32 (or Pico) via DTR/RTS and capture its fresh boot serial output.
Needed because the SISR sketches print once in setup() and leave loop() empty, so a
monitor that attaches *after* boot sees nothing. We open the port, pulse the auto-reset
line to re-run the app, then read until the sketch's final "Done." or a timeout.

    .venv/bin/python capture_serial.py /dev/cu.usbserial-0001 [secs] [baud]
"""
import sys, time
import serial

port = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
baud = int(sys.argv[3]) if len(sys.argv) > 3 else 115200

ser = serial.Serial()
ser.port = port
ser.baudrate = baud
ser.timeout = 0.2
ser.dtr = False          # GPIO0 high -> normal boot (run app, not bootloader)
ser.rts = False
ser.open()

# ESP32 auto-reset "run" pulse: assert EN (RTS) low briefly, keep DTR inactive.
ser.dtr = False
ser.rts = True
time.sleep(0.12)
ser.reset_input_buffer()
ser.rts = False

t0 = time.time()
saw_done = False
while time.time() - t0 < secs:
    line = ser.readline()
    if not line:
        if saw_done:
            break
        continue
    s = line.decode("utf-8", "replace")
    sys.stdout.write(s)
    sys.stdout.flush()
    if "Done." in s:
        saw_done = True          # one more drain pass, then stop
ser.close()
sys.stderr.write(f"[capture] {time.time()-t0:.1f}s elapsed, done={saw_done}\n")
