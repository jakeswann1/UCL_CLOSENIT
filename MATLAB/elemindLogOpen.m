% Written by David Wang. This function sets up the threads
% for handling incoming data, i.e. logging and processing.

function [logAbsPath, logFID] = elemindLogOpen(varargin)
    % elemindLogOpen - Opens log file and starts logging serial data
    %
    % Optional input:
    %   logFilePath - Path to log file. If not provided, generates default path
    %
    % Outputs:
    %   logFID - File identifier for the opened log file
    %   logAbsPath - Absolute path to the log file
    
    if nargin > 0
        logFilePath = varargin{1};
        % Convert to absolute path if relative
        if ~isabsolute(logFilePath)
            logFilePath = fullfile(pwd, logFilePath);
        end

        if isfile(logFilePath)
            filesplit = strsplit(logFilePath, '.');
            timestamp = datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd_HH-mm-ss_Z');
            logFilePath = filesplit{1} + "_" + char(timestamp) + "." + filesplit{2};
        end
    else
        % Generate unique filename with timestamp in current directory
        timestamp = datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd_HH-mm-ss_Z');
        filename = ['elemind_matlab_serial_log_', char(timestamp), '.txt'];
        filename = strrep(filename, ':', '-');
        logFilePath = fullfile(pwd, filename);
    end
    
    % Ensure directory exists
    [pathstr,~,~] = fileparts(logFilePath);
    if ~exist(pathstr, 'dir')
        mkdir(pathstr);
    end
    
    % Open log file
    logFID = fopen(logFilePath, 'a');
    if logFID == -1
        error('Failed to open log file: %s', logFilePath);
    end
    
    % Get receive queue
    recvQueue = evalin('base', 'recvQueue');
    
    % Set up processing/logging callback
    afterEach(recvQueue, @(data)elemindLogProcess(logFID, 'Recv', data));
    
    % Save file handle and path to base workspace
    assignin('base', 'logFID', logFID);
    assignin('base', 'logAbsPath', logFilePath);
    
    % Get canonical absolute path
    logAbsPath = char(java.io.File(logFilePath).getCanonicalPath());
    
    fprintf('Logging started to file: "%s"\n', logAbsPath);
end

function isabs = isabsolute(filepath)
    % Helper function to determine if path is absolute
    if ispc
        % Windows: Check for drive letter or UNC path
        isabs = ~isempty(regexp(filepath, '^([a-zA-Z]:)|^(\\\\)', 'once'));
    else
        % Unix-like: Check for leading slash
        isabs = strcmp(filepath(1), '/');
    end
end