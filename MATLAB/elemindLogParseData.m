function parsedData = elemindLogParseData(logFIDOrStringPath)
% elemindLogParseData - Parse data from a log file
%
% Input:
%   logFIDOrStringPath - Either a file ID (number) or path to log file (string)
% Output:
%   parsedData - Structure containing parsed data arrays

%% Setup gain scalars for raw data
eeg_adc2volts_numerator = (4.5/(8388608.0-1));
eeg_gain_default = 24;
eeg_adc2volts = eeg_adc2volts_numerator/eeg_gain_default;

accel_raw2g = 1/(2^14);

% Handle input argument
if ischar(logFIDOrStringPath) || isstring(logFIDOrStringPath)
    % Open the file if path was provided
    fid = fopen(logFIDOrStringPath, 'r');
    if fid == -1
        error('Failed to open log file: %s', logFIDOrStringPath);
    end
    closeFile = true;
else
    % Use provided file ID
    fid = logFIDOrStringPath;
    closeFile = false;
end

% Initialize output structure
parsedData = struct();

try
    % Read file line by line
    while ~feof(fid)
        % Read a line from the file
        rawData = fgetl(fid);

        % Skip empty lines
        if isempty(rawData)
            continue;
        end

        % Parse the data (look for lines containing "data_log_")
        match = regexp(rawData, 'V \((\d+)\) data_log_(\w+): ([\d\.\-\s]+)', 'tokens');
        if ~isempty(match)
            rtos_timestamp = str2double(match{1}{1}); % Extract and convert the timestamp
            dataType = match{1}{2}; % Extract the specific data type
            values = str2num(match{1}{3}); % Extract and convert the values

            if ~isempty(values)
                value_timestamp = values(1); % First number is timestamp [microseconds]
                value_data = values(2:end); % Remaining numbers are sensor data

                % Initialize field if it doesn't exist
                if ~isfield(parsedData, dataType)
                    parsedData.(dataType) = [];
                end

                % Process data based on type
                switch dataType
                    case 'eeg'
                        value_data = value_data * eeg_adc2volts;
                        parsedRow = [value_timestamp value_data];
                    case 'accel'
                        value_data = value_data * accel_raw2g;
                        parsedRow = [value_timestamp value_data];
                    case {'audio', 'leadoff'}
                        parsedRow = [value_timestamp value_data];
                    case 'inst_amp_phs'
                        try
                            value_data(1) = value_data(1) * eeg_adc2volts;
                            value_data(2) = rad2deg(value_data(2));
                            parsedRow = [value_timestamp value_data];
                        catch ME
                            warning(num2str(value_timestamp) + ", NaN data");
                            value_data = [0 0];
                            parsedRow = [value_timestamp value_data];
                        end
                    otherwise
                        fprintf('Unknown data type: %s. Skipping...\n', dataType);
                        continue;
                end

                % Append the data
                parsedData.(dataType) = [parsedData.(dataType); parsedRow];
            end
        end
    end

catch ME
    % Clean up and rethrow error
    if closeFile
        fclose(fid);
    end
    rethrow(ME);
end

% Close file if we opened it
if closeFile
    fclose(fid);
end
end
