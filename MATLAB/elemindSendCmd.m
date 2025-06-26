% Written by David Wang - slightly modified by Thomas to control whether the command is printed to
% MATLAB console

function elemindSendCmd(command, print)
% elemindSendCmd - Sends command to serial port via worker thread
%
% command: Command string to send

try
    % Get command queue
    cmdQueue = evalin('base', 'sendQueue');

    % Send command
    send(cmdQueue, command);

    if print == 1
        fprintf('elemindSendCmd "%s"\n', command);
    end

    % Log command if logging is enabled
    try
        fid = evalin('base', 'logFid');
        timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');
        fprintf(fid, '[%s] Sent: %s\n', timestamp, command);
    catch
        % Not logging, ignore
    end

catch ME
    error('Failed to send command: %s', ME.message);
end
end
