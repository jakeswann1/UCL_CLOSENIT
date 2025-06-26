% Written by David Wang with modifications by Junheng and Thomas for 
% computing alpha and theta powers.

function elemindLogProcess(fid, direction, data)
% elemindLogWrite - Logs received or sent data to an open file
%
% fid: File handle for the log file
% direction: 'Recv' or 'Sent'
% data: The data to log and process (string)
global bsf_on;
global bpf_on;
global hpf_on;

global j; % iterable variable for keeping track of EEG sample count
global timeStart; % Time after recording starts to begin processing

% Real-time plotting globals
global enableRealTimePlotting;

try
    if fid > 0
        % Get the current timestamp
        timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');

        % Write the log entry to the open file
        fprintf(fid, '[%s] %s: %s\n', timestamp, direction, data);
    else
        % If fid is invalid, skip logging
        warning('elemindLogWrite: Invalid file handle. Skipping log entry.');
    end

    % Conversion factors for streamed data to volts
    eeg_adc2volts_numerator = (4.5/(8388608.0-1));
    eeg_gain_default = 24;
    eeg_adc2volts = eeg_adc2volts_numerator/eeg_gain_default;
    match = regexp(data, 'V \((\d+)\) data_log_(\w+): ([\d\.\-\s]+)', 'tokens');

    if ~isempty(match)
        dataType = match{1}{2}; % Extract the specific data type (e.g., inst_amp_phs, eeg, etc.)
        values = str2num(match{1}{3}); % Extract and convert the values to a numeric array

        if ~isempty(values)
            value_data = values(2:end); % The remaining numbers are the sensor data
        end

        % Check the dataType and handle each case
        if strcmp(dataType, 'eeg') % Convert to Volts
            value_data = eeg_adc2volts * value_data;

            % Store raw EEG for later use
            raw_eeg_sample = value_data(:);

            eeg_filted = value_data(:);
            if bsf_on == 1
                eeg_filted = perSampleBSFilter(eeg_filted);
            end
            if hpf_on == 1
                eeg_filted = perSampleHPFilter(eeg_filted);
            end
            if bpf_on == 1
                eeg_filted = perSampleBPFilter(eeg_filted);
            end

            % Update plotting buffers (much less frequently)
            if enableRealTimePlotting
                updateSimplePlottingBuffers(eeg_filted);
            end

            % Print progress every 1000 samples (4 seconds at 250 Hz)
            if mod(j, 1000) == 0
                elapsed_time = j / 250; % Approximate time in seconds
                fprintf('EEG samples: %d (%.1f seconds)\n', j, elapsed_time);
            end

            % START YOUR CLOSED LOOP CONTROL HERE
            
            % END YOUR CLOSED LOOP CONTROL HERE

            j = j+1; % Update index iterable variable
        end
    end

catch ME
    % Print errors for debugging
    fprintf('Error in elemindLogProcess: %s\n', ME.message);
    if enableRealTimePlotting
        fprintf('Disabling real-time plotting due to error\n');
        enableRealTimePlotting = 0;
    end
end
end

%% Function to update plotting buffers 
function updateSimplePlottingBuffers(filteredEeg)
    global plotUpdateCounter;
    global plotUpdateInterval;
    global simpleEegBuffer;

    try
        % Always update buffers (this is fast)
        simpleEegBuffer(1:end-1, :) = simpleEegBuffer(2:end, :);
        simpleEegBuffer(end, :) = filteredEeg;
        
        % Increment counter
        plotUpdateCounter = plotUpdateCounter + 1;
        
        % Only update plots at specified interval (much less frequently)
        if mod(plotUpdateCounter, plotUpdateInterval) == 0
            updateSimplePlots();
        end
        
    catch ME
        % Don't let plotting errors affect data acquisition
        fprintf('Plot buffer update error (sample %d): %s\n', plotUpdateCounter, ME.message);
    end
end

%% Function to update the actual plots
function updateSimplePlots()
    global simpleEegBuffer;
    global simpleFigure;
    global plotUpdateCounter;
    global bufferT;
    global Ts;
    global j;
    
    try
        % Check if figure still exists
        if ~ishandle(simpleFigure) || ~isvalid(simpleFigure)
            return; % Exit silently if figure was closed
        end

        bufferT = Ts.*(j-length(bufferT)+1:j);
        
        % Update EEG subplot
        cla; % Clear and redraw (simple approach)
        plot(bufferT, simpleEegBuffer, 'LineWidth', 1);
        title(sprintf('Real-time Filtered EEG (Time %d s)', plotUpdateCounter*Ts));
        xlabel('Time (s)')
        ylabel('Voltage (V)');
        legend({'Fp1', 'Fpz', 'Fp2'}, 'Location', 'northeast');
        grid on;
        
        % Force update but limit rate
        drawnow limitrate;
        
    catch ME
        % Silently handle plot errors to avoid disrupting data collection
        fprintf('Plot update error: %s\n', ME.message);
end
end
