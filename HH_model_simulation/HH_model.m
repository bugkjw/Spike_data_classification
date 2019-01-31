%%
clear all; close all;
% Integration time step: 10ms
DT = 0.01; TIME = 300;
%% Input current model
time_series = 1:DT:TIME; TS = 20;
% KCl
KCl_AMP = 10;
KCl_input = ones(size(time_series))*KCl_AMP;
% GABA
GABA_AMP = 1; GABA_DUR = 7; GABA_TAU = 4;
GABA_STEP = zeros(size(time_series));
GABA_STEP(TS/DT:(GABA_DUR+TS)/DT) = GABA_AMP;
GABA_CONVFUNC = exp(-(1:DT:20)/GABA_TAU)/GABA_TAU^2.*(1:DT:20);
GABA_input = conv(imgaussfilt(GABA_STEP,1/DT),GABA_CONVFUNC,'same');
% Cap
Cap_AMP = 3; Cap_DUR = 13; Cap_TAU = 4;
Cap_STEP = zeros(size(time_series));
Cap_STEP(TS/DT:(Cap_DUR+TS)/DT) = Cap_AMP;
Cap_CONVFUNC = exp(-(1:DT:20)/Cap_TAU)/Cap_TAU^2.*(1:DT:20);
Cap_input = conv(imgaussfilt(Cap_STEP,3/DT),Cap_CONVFUNC,'same');

figure;
subplot(131); plot(time_series,KCl_input); title("Input current induced by KCl");
xlabel("time(s)");
xlim([0 100]); ylim([0 max(Cap_input)]); set(gca, 'YDir','reverse');
subplot(132); plot(time_series, GABA_input); title("Input current induced by GABA");
xlabel("time(s)");
xlim([0 100]); ylim([0 max(Cap_input)]); set(gca, 'YDir','reverse');
subplot(133); plot(time_series, Cap_input); title("Input current induced by Cap");
xlabel("time(s)");
xlim([0 100]); ylim([0 max(Cap_input)]); set(gca, 'YDir','reverse');
%% HH model response
v = -65; mi = 0.4; hi =  0.2; ni = 0.5;
[V_KCl,m,h,n,t] = hhrun(KCl_input,100, v, mi, hi, ni,0);
[V_GABA,m,h,n,t] = hhrun(GABA_input,100, v, mi, hi, ni,0);
[V_Cap,m,h,n,t] = hhrun(Cap_input,100, v, mi, hi, ni,0);
figure;
subplot(131); plot(t,V_KCl); title("Response to KCl");
xlabel("time(s)"); ylabel("Membrane voltage (mV)");
subplot(132); plot(t,V_GABA); title("Response to GABA");
xlabel("time(s)");
subplot(133); plot(t,V_Cap); title("Response to Cap");
xlabel("time(s)");