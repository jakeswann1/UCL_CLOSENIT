% Written by Thomas - does sample by sample band stop filtering
% of incoming EEG data

function y = perSampleBSFilter(eeg)
global w1_bsf;
global w2_bsf;
global w3_bsf;
global w4_bsf;

global b_bsf;
global a_bsf;

x = eeg;
y = b_bsf(1)*x+w1_bsf;
w1_bsf = b_bsf(2)*x+w2_bsf-a_bsf(2)*y;
w2_bsf = b_bsf(3)*x+w3_bsf-a_bsf(3)*y;
w3_bsf = b_bsf(4)*x+w4_bsf-a_bsf(4)*y;
w4_bsf = b_bsf(5)*x-a_bsf(5)*y;
end

