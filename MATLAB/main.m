% MATLAB Codebase written by David Wang for interfacing with the Elemind Headband.
% Edits by Junheng and Thomas to enable real-time processing with
% logging and data visualisation.
clear;
close all;
clc;

%% Recording parameters
team_num = 0; % CHANGE TO YOUR TEAM NUMBER
subject_num = 0; % CHAGNE TO YOUR SUBJECT NUMBER
logfilename = "Team_" + num2str(team_num) + "_sid_" + num2str(subject_num) + ".txt"; % Give it some name

global timeStart; % Time after recording starts to begin processing
global closedloopStim; % Flag for closed loop control

baselinetime = 60; % Time before stimulation starts for baseline EEG
stimulationTime = 2*60; % Time for stimulation to run
SamplingDurationSecs = stimulationTime + baselinetime; % Duration (in seconds) for the sampling loop to run
timeStart = baselinetime; % Discard window of seconds at start due to startup noise with filter transients
timeEnd = SamplingDurationSecs; % Time to end the recording

%% List all available COM ports
availablePorts = serialportlist;

%% Display the available ports
disp('Available COM ports:');
disp(availablePorts);

%% Set up serial port communication using serialport
% Replace with your actual COM port
% Check in /dev/tty* for Mac/Linux, or Device Manager for Windows

% port = "/dev/ttyUSB0";  % Linux example
% port = "/dev/tty.usbmodem14401";  % Mac example
port = 'COM3'; % Windows example
elemindSerialPortOpen(port);

%% Init Global Variables
% w* variables are for IIR filters, perSampleBPFilter, perSampleBSFilter and
% perSampleHPFilter. b_* and a_* variables are for the filter transfer
% function coefficients
global w1_hpf;
global w2_hpf;
global w1_bpf;
global w2_bpf;
global w3_bpf;
global w4_bpf;
global w1_bsf;
global w2_bsf;
global w3_bsf;
global w4_bsf;
global b_bpf;
global a_bpf;
global b_bsf;
global a_bsf;
global b_hpf;
global a_hpf;

% Flags for turning filters on/off (1/0 respectively)
global hpf_on;
global bsf_on;
global bpf_on;

global j; % iterable variable for keeping track of EEG sample count
global Ts; % Time between samples

% Real-time plotting variables
global enableRealTimePlotting; % Flag to enable/disable plotting
global plotUpdateCounter; % Counter for plot updates
global plotUpdateInterval; % Update every N samples (much less frequent)
global simpleEegBuffer; % Simple buffer for recent EEG data
global simpleFigure; % Single figure handle
global bufferT; % Time axis for live plot data

Fs = 250; % Default Sampling rate
Ts = 1/Fs; % Default Sampling time

bandpassCentreFreq = 10; % Centre frequency for bandpass phase tracking
[b_hpf, a_hpf] = butter(2, 0.5/(Fs/2), 'high'); % remove low frequency artefacts from EEG
[b_bpf, a_bpf] = butter(2, [bandpassCentreFreq*(1-0.25) bandpassCentreFreq*(1+0.25)]/(Fs/2)); % Alpha bandpass filter
[b_bsf, a_bsf] = butter(2, [45 55]/(Fs/2), 'stop'); % Line noise filter

% Intialise all to 0
w1_hpf = 0;
w2_hpf = 0;
w1_bpf = 0;
w2_bpf = 0;
w3_bpf = 0;
w4_bpf = 0;
w1_bsf = 0;
w2_bsf = 0;
w3_bsf = 0;
w4_bsf = 0;

bsf_on = 1; % Turn on line noise filter
hpf_on = 1; % Turn on low frequency drift filter
bpf_on = 0; % Turn off alpha filter

j = 1;

%% Initialize real-time plotting
enableRealTimePlotting = 1; % Set to 0 to disable plotting entirely
plotUpdateCounter = 0; % Count how many times the plot updated
plotUpdateInterval = 250; % Update plots every 250 samples (once per second at 250Hz)

if enableRealTimePlotting
    % Simple buffers - only keep last 1000 samples (4 seconds at 250Hz)
    bufferSize = 1000;
    bufferT = Ts.*(-(bufferSize-1):0); % Time axis data for live EEG
    simpleEegBuffer = zeros(bufferSize, 3); % Buffer for live EEG data
    simplePowerBuffer = zeros(bufferSize, 1); % Buffer for live power data

    % Create single simple figure
    simpleFigure = figure('Name', 'Real-time EEG Monitor', 'Position', [100, 300, 1000, 600]);

    % Create plot
    plot(bufferT, zeros(bufferSize, 3), 'LineWidth', 1);hold on;
    title('Real-time Filtered EEG (Last 4 seconds)');
    xlabel('Time (s)');
    ylabel('Voltage (V)');
    legend({'Fp1', 'Fpz', 'Fp2'}, 'Location', 'northeast');
    grid on;

    drawnow;

    fprintf('Real-time plotting enabled (updates every %.1f seconds)\n', plotUpdateInterval/Fs);
else
    fprintf('Real-time plotting disabled for maximum performance\n');
end

%% Start Logging
[logAbsPath, logFID] = elemindLogOpen(logfilename);

%% Audio Configuration
fprintf('Configuring audio output...\n');

% Set master volume first
elemindSendCmd('audio_set_volume 128', 1); % 50% master volume

% Example code to test audio output briefly
fprintf('Testing audio output...\n');
elemindSendCmd('audio_play_test 440', 1); % Play 440Hz test tone
pause(1);
elemindSendCmd('audio_stop_test', 1);
fprintf('Audio test complete.\n');
% end of example code to test audio output briefly

%% Ask for streaming data
elemindSendCmd('stream eeg 1', 1); % Stream eeg enabled
elemindSendCmd('stream accel 0', 1); % Don't need
elemindSendCmd('stream audio 0', 1); % Doesn't work
elemindSendCmd('stream leadoff 1', 1); % unknown units

% Turn on on-device filters
elemindSendCmd('therapy_enable_line_filters 1', 1);
elemindSendCmd('therapy_enable_az_filters 1', 1);
elemindSendCmd('therapy_enable_ac_filters 1', 1);

%% Start session
elemindSendCmd('eeg_start', 1);
elemindSendCmd('accel_start', 1);

%% Continuously read and write data for X seconds
fprintf('=== DATA ACQUISITION STARTED ===\n');
disp(['Reading and writing data to serial port for ', num2str(SamplingDurationSecs), ' seconds...']);

if enableRealTimePlotting
    disp('Real-time plots will update every second.');
end

% Progress indicator
tic;
startTime = datetime('now');
lastProgressUpdate = 0;

while toc < SamplingDurationSecs
    % Update progress every 5 seconds
    elapsed = toc;
    if elapsed - lastProgressUpdate >= 5
        remaining = SamplingDurationSecs - elapsed;
        fprintf('Progress: %.1f/%.1f seconds (%.1f%% complete, %.1f seconds remaining)\n', ...
            elapsed, SamplingDurationSecs, 100*elapsed/SamplingDurationSecs, remaining);
        lastProgressUpdate = elapsed;
    end

    pause(1); % Check every second
end

fprintf('=== DATA ACQUISITION COMPLETED ===\n');

%% Stop session
elemindSendCmd('eeg_stop', 1);
elemindSendCmd('accel_stop', 1);

elemindSendCmd('audio_pink_volume 1', 1); % Restore pink noise volume
elemindSendCmd('audio_pink_fade_out 0', 1); % Set to stop immediately
elemindSendCmd('audio_pink_stop', 1); % Stop the pink noise
elemindSendCmd('audio_pink_unmute', 1); % Turn off mute for pink noise

elemindSendCmd('audio_bg_fade_out 0', 1); % Set background wav to stop immediately
elemindSendCmd('audio_bgwav_stop', 1); % Stop background wave file

%% Stop logging
pause(1)
beep
elemindLogClose();

%% Close the Serial Port
elemindSerialPortClose();

%% Display summary
fprintf('\n=== SESSION SUMMARY ===\n');
fprintf('Total samples collected: %d\n', j);
fprintf('Duration: %.2f s\n', j/Fs);
fprintf('Log file: %s\n', logAbsPath);

%% Analyze the log (existing post-processing code)
fprintf('Analyzing recorded data...\n');
parsedData = elemindLogParseData(logAbsPath);

%% Plot EEG and Accelerometer Data
if isfield(parsedData, 'eeg') && ~isempty(parsedData.eeg)
    eegTime = (parsedData.eeg(:,1) - parsedData.eeg(1,1)) / 1e6; % convert us to s
    sample_times = diff(eegTime); % Compute time between samples.

    % Compute actual time between samples and actual sampling rate
    ts = mean(sample_times);
    if ts < 0
        ts = mode(sample_times);
    end
    std_ts = sqrt((1/length(sample_times)).*sum(abs(sample_times - ts).^2));
    Fs = 1/ts;

    rawParsedData = parsedData;

    % Create EEG figure
    figure('Name', 'EEG Time Data - PostProcessed');
    if bsf_on == 1
        parsedData.eeg(:, 2:4) = filter(b_bsf, a_bsf, parsedData.eeg(:, 2:4));
    end
    if hpf_on == 1
        parsedData.eeg(:, 2:4) = filter(b_hpf, a_hpf, parsedData.eeg(:, 2:4));
    end
    if bpf_on == 1
        parsedData.eeg(:, 2:4) = filter(b_bpf, a_bpf, parsedData.eeg(:, 2:4));
    end

    % Plot each EEG channel from timestart to timeend
    eeg = parsedData.eeg(round(timeStart*Fs)+1:round(timeEnd*Fs), 2:4);
    N = length(eeg);
    t = ts.*(0:N-1);
    plot(t, eeg);
    title('Time Data');
    xlabel('Time (seconds)');
    ylabel('Voltage (V)');
    legend({'Fp1', 'Fpz', 'Fp2'});
    grid on;

    % Create frequency figure
    figure('Name', 'EEG Freq Data');

    % Convert timestamps from microseconds to seconds
    EEG = fft(eeg)./(N/2);
    EEG_mag = abs(EEG);
    EEG_pow = 20.*log10(EEG_mag);

    f = (Fs/N).*(0:N-1);

    % Plot frequency power
    plot(f, EEG_pow);
    xlim([0 65])
    title('Frequency Data');
    xlabel('Frequency (Hertz)');
    ylabel('Power (dbV)');
    legend({'Fp1', 'Fpz', 'Fp2'});
    grid on;
end

fprintf('Analysis complete!\n');
