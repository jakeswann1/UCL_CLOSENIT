function elemindSerialPortClose()
    % elemindCloseSerialPort - Closes serial port and terminates worker
    
    try
        % Get worker from base workspace
        worker = evalin('base', 'serialWorker');
        
        % Cancel worker
        cancel(worker);
        
        % Clear variables from base workspace
        evalin('base', 'clear serialWorker cmdQueue recvQueue');
        
        disp('Serial port worker stopped');
    catch ME
        warning('Error closing serial port: %s', ME.message);
    end
end