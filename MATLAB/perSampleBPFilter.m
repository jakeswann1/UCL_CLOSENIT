% Written by Thomas - does sample by sample band pass filtering
% of incoming EEG data

function y = perSampleBPFilter(eeg)
global w1_bpf;
global w2_bpf;
global w3_bpf;
global w4_bpf;

global b_bpf;
global a_bpf;

x = eeg;
y = b_bpf(1)*x+w1_bpf;
w1_bpf = b_bpf(2)*x+w2_bpf-a_bpf(2)*y;
w2_bpf = b_bpf(3)*x+w3_bpf-a_bpf(3)*y;
w3_bpf = b_bpf(4)*x+w4_bpf-a_bpf(4)*y;
w4_bpf = b_bpf(5)*x-a_bpf(5)*y;
end

