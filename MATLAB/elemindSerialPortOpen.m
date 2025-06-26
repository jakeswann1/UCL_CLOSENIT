function elemindSerialPortOpen(port)
    % elemindOpenSerialPort - Opens serial port in worker thread
    %
    % port: COM port to open (e.g. 'COM3')
    
    % Check if worker already exists
    if ~isempty(evalin('base', 'who(''serialWorker'')'))
        error('Serial worker already running. Close it first with elemindCloseSerialPort');
    end

    % Create a queue to get data received from the serial port out of the
    % worker.
    recvQueue = parallel.pool.DataQueue;
    assignin('base', 'recvQueue', recvQueue);

    % Create a passbackQueue to get data from the worker
    passbackQueue = parallel.pool.PollableDataQueue;
    
    % Start worker with queues as arguments
    worker = parfeval(@serialWorkerFunction, 0, port, recvQueue, passbackQueue);
    afterEach(worker, @(worker) disp(worker.Diary), 0, 'PassFuture', true);

    sendQueue = poll(passbackQueue,60);
    assignin('base', 'sendQueue', sendQueue);
    
    % Save worker to base workspace
    assignin('base', 'serialWorker', worker);    
    disp(['Serial port worker started on port ', port]);
end

function serialWorkerFunction(port, recvQueue, passbackQueue)
    % Worker function that handles serial port communication
    fprintf('serialWorkerFunction starting on port %s\n', port);

    try
        % Open serial port
        s = serialport(port, 115200);
        configureTerminator(s, "LF");
        s.DataBits = 8;
        s.StopBits = 1;
        s.Parity = 'none';
        s.Timeout = 1;

        sendQueue = parallel.pool.PollableDataQueue;
        send(passbackQueue,sendQueue);
        % Set up command handling
        % afterEach(sendQueue, @(data)handleCommand(s, data));

        fprintf('Serial port opened successfully on port %s\n', port);

        % Main loop
        while true
            % Check for incoming serial data
            while s.NumBytesAvailable > 0
                data = readline(s);
                send(recvQueue, data);
                fprintf('Received: %s\n', data);
            end
            % Check for data to write
            while true
                data = poll(sendQueue, 0);
                if (~isempty(data))
                    handleCommand(s,data);
                else
                    break;
                end
            end
            % pause(0.001); % Small delay
        end
        
    catch ME
        fprintf('Serial worker error: %s\n', ME.message);
        rethrow(ME);
    end
end


function handleCommand(s, data)
    % Handle commands within the worker thread
    try
        s.writeline(data);
    catch ME
        warning('Failed to send command: %s', ME.message);
    end
end