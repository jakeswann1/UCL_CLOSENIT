% Written by Thomas - does sample by sample high pass filtering
% of incoming EEG data

function y = perSampleHPFilter(eeg)
global w1_hpf;
global w2_hpf;
global b_hpf;
global a_hpf;

x = eeg;
y = b_hpf(1)*x+w1_hpf;
w1_hpf = b_hpf(2)*x+w2_hpf-a_hpf(2)*y;
w2_hpf = b_hpf(3)*x-a_hpf(3)*y;
end
