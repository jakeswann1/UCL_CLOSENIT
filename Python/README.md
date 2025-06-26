# closeNIT-hackathon-2025 - Python interface
Repository for people working on the closeNIT hackathon 2025 - codebase features Python interface for elemind EEG headband and documentation

## How to use
1. Connect the Elemind headband to PC running Python 3.12 (with packages in **[Requirements.txt](./Requirements.txt)** installed).
2. Check what COM port the headband is using (in Windows, use Device Manager for this information - or ls /dev/tty* on Mac/Linux).
3. Replace the 'port' variable of the ['*main.py*'](./main.py) value with a string corresponding to the correct COM port. Example:
```python
port = "COM3"  # Windows example
port = "/dev/ttyUSB0"  # Linux example
port = "/dev/tty.usbmodem14401"  # Mac example
```
4. Replace the 'group_num' variable of the ['*main.py*'](./main.py) value with an int corresponding to the correct group number. Replace the 'subject_num' variable of the ['*main.py*'](./main.py) value with an int corresponding to the correct subject number.
5. Run the ['*main.py*'](./main.py) script to begin streaming EEG.

## What it does
It will start streaming EEG to a log file that will get saved in .txt format. The ['*_process_eeg_sample()*'](./main.py) function is where EEG is processed sample by sample and also where code can be introduced for closed-loop control.

## How to send a command
To send a command to the device, use the '*send_command*' function. For example:
```python
audio_vol = 0.5
self.send_command(f"audio_pink_volume {audio_vol:.2f}", False) # Skip command-window print for speed
```
