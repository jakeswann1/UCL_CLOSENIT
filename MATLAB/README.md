# closeNIT-hackathon-2025 - MATLAB interface
Repository for people working on the closeNIT hackathon 2025 - codebase features MATLAB interface for elemind EEG headband and documentation

## How to use
1. Connect the Elemind headband to PC running MATLAB (**with [parallel computing toolbox](https://uk.mathworks.com/help/parallel-computing/getting-started-with-parallel-computing-toolbox.html) installed**).
2. Check what COM port the headband is using (in Windows, use Device Manager for this information - or ls /dev/tty* on Mac/Linux).
3. Replace the 'port' variable of the ['*main.m*'](./main.m) value with a string corresponding to the correct COM port. Example:
```MATLAB
port = "COM3"; % Windows example
port = "/dev/ttyUSB0";  % Linux example
port = "/dev/tty.usbmodem14401";  % Mac example
```
4. Replace the 'group_num' variable of the ['*main.m*'](./main.m) value with an int corresponding to the correct group number. Replace the 'subject_num' variable of the ['*main.m*'](./main.m) value with an int corresponding to the correct subject number.
5. Run the ['*main.m*'](./main.m) script to begin streaming EEG.

## What it does
It will start streaming EEG to a log file that will get saved in .txt format. The ['*elemindLogProcess.m*'](./elemindLogProcess.m) function is where EEG is processed sample by sample and also where code can be introduced for closed-loop control.

## How to send a command
To send a command to the device, use the '*elemindSendCmd*' function. For example:
```MATLAB
audio_vol = 0.5
elemindSendCmd("audio_pink_volume "+ num2str(audio_vol, '%.2f'), 0); % Skip command-window print for speed
```
