function elemindLogClose()
    % elemindLogClose - Closes log file
    
    try
        % Get file handle from base workspace
        fid = evalin('base', 'logFID');
        
        % Close file
        fclose(fid);
        
        % Clear from workspace
        evalin('base', 'clear logFID');
        
        disp('Log file closed');
    catch ME
        warning('Error closing log file: %s', ME.message);
    end
end