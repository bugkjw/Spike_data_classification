%%
clear all; close all;
Display = false;
%% Initialize
load("TrainData.mat"); load("TestData.mat");
dt = 10/1000; % Time bin = 10ms
max_time = 300; % Time limit = 300s
num_train = 49; num_test = 15;
START_TIME = 0; END_TIME = 300;
SPLIT_1 = 80; SPLIT_2 = 190;
%% Selected spike train features
% Training set features
TRAIN_MEAN_RATE = zeros(num_train,1);
TRAIN_TRANSIENT_RATE = zeros(num_train,1); TRAIN_MID_RATE = zeros(num_train,1); TRAIN_END_RATE = zeros(num_train,1);
TRAIN_PEAK_MAX = zeros(num_train,1); TRAIN_PEAK_MEAN = zeros(num_train,1);
TRAIN_PEAK_STD = zeros(num_train,1); TRAIN_PEAK_INV = zeros(num_train,1); TRAIN_NUM_PEAKS = zeros(num_train,1);
TRAIN_ISI_MEAN = zeros(num_train,1); TRAIN_ISI_STD = zeros(num_train,1);
TRAIN_SR_MEAN = zeros(num_train,1); TRAIN_SR_STD = zeros(num_train,1);
TRAIN_L_MEAN = zeros(num_train,1); TRAIN_L_STD = zeros(num_train,1);
% Test set features
TEST_MEAN_RATE = zeros(num_test,1);
TEST_TRANSIENT_RATE = zeros(num_test,1); TEST_MID_RATE = zeros(num_test,1); TEST_END_RATE = zeros(num_test,1);
TEST_PEAK_MAX = zeros(num_test,1); TEST_PEAK_MEAN = zeros(num_test,1);
TEST_PEAK_STD = zeros(num_test,1); TEST_PEAK_INV = zeros(num_test,1); TEST_NUM_PEAKS = zeros(num_test,1);
TEST_ISI_MEAN = zeros(num_test,1); TEST_ISI_STD = zeros(num_test,1);
TEST_SR_MEAN = zeros(num_test,1); TEST_SR_STD = zeros(num_test,1);
TEST_L_MEAN = zeros(num_test,1); TEST_L_STD = zeros(num_test,1);
Feature_names = ["Mean firing rate",...
    "Max firing rate (instantaneous)",...
    "Mean firing rate (instantaneous)",...
    "Peak STD (instantaneous)",...
    "Peak interval (instantaneous)",...
    "# of peaks (instantaneous)",...
    "ISI Mean",...
    "ISI STD",...
    "Spikelet regularity mean",...
    "Spikelet regularity STD",...
    "Spikelet length mean",...
    "Spikelet length STD",...
    "Transient firing rate",...
    "Steady firing rate",...
    "Decaying firing rate"];
%% Spike train preprocessing
% Training data processing
TRAIN_BINARY = zeros(num_train,max_time/dt);
TRAIN_IMAGE = zeros(num_train,121);
TRAIN_LABEL = Ytrain;
for ii = 1:num_train
    % Extract spike timing data and stimulus label
    temp_label = TRAIN_LABEL(ii);
    temp_spike_timing = cell2mat(Xtrain(ii));
    temp_spike_timing = temp_spike_timing(temp_spike_timing>START_TIME);
    temp_spike_timing = temp_spike_timing(temp_spike_timing<END_TIME);
    % Digitize the spikes into binary array
    temp_spike_timing_digitized = unique(round(temp_spike_timing/dt));
    TRAIN_BINARY(ii,temp_spike_timing_digitized) = 1/dt;
    % Extract features from the spike train
    % Mean firing rate (Hz)
    TRAIN_MEAN_RATE(ii) = length(temp_spike_timing)/max(temp_spike_timing);
    TRAIN_TRANSIENT_RATE(ii) = length(temp_spike_timing(temp_spike_timing<SPLIT_1))/SPLIT_1;
    TRAIN_MID_RATE(ii) = length(temp_spike_timing(temp_spike_timing>SPLIT_1 & temp_spike_timing<SPLIT_2))/(SPLIT_2-SPLIT_1);
    TRAIN_END_RATE(ii) = length(temp_spike_timing(temp_spike_timing>SPLIT_2))/(max_time-SPLIT_2);
    % Instantaneous firing rate (Gaussian window convolusion)
    temp_curve = imgaussfilt(TRAIN_BINARY(ii,:),5/dt);
    % Maximum firing rate
    TRAIN_PEAK_MAX(ii) = max(temp_curve);
    % Number of peaks
    [pks,locs] = findpeaks(temp_curve,'MinPeakDistance',100,'MinPeakProminence',0.05);
    TRAIN_NUM_PEAKS(ii) = size(locs,2);
    TRAIN_PEAK_MEAN(ii) = mean(pks);
    TRAIN_PEAK_STD(ii) = std(pks);
    if length(locs)~=1
        TRAIN_PEAK_INV(ii) = mean(locs(2:end)-locs(1:end-1))*dt;
    else
        TRAIN_PEAK_INV(ii) = 0;
    end
    % Inter-spike interval, mean and std
    temp_ISI = temp_spike_timing(2:end) - temp_spike_timing(1:end-1);
    TRAIN_ISI_MEAN(ii) = mean(temp_ISI);
    TRAIN_ISI_STD(ii) = std(temp_ISI);
    % Spikelet regularity, mean and std
    temp_L = temp_spike_timing(3:end) - temp_spike_timing(1:end-2);
    temp_SR = 2*(temp_spike_timing(3:end) - temp_spike_timing(2:end-1))...
        ./(temp_spike_timing(3:end) - temp_spike_timing(1:end-2));
    TRAIN_SR_MEAN(ii) = mean(temp_SR); TRAIN_SR_STD(ii) = std(temp_SR);
    TRAIN_L_MEAN(ii) = mean(temp_L); TRAIN_L_STD(ii) = std(temp_L);
    % Image feature
    %IMG = hist3([temp_L temp_SR],'edges',{0:1.5:15 0:0.2:2}); TRAIN_IMAGE(ii,:) = IMG(:);
    % Display
    if Display
        CurrentData = figure;
        subplot(311); plot([temp_spike_timing temp_spike_timing],[0 1],'k'); xlim([0 max_time]);
        title("Data "+num2str(ii)+", label: "+num2str(temp_label)+", mean rate "+num2str(TRAIN_MEAN_RATE(ii))+"Hz");
        subplot(312); plot(temp_curve); hold on; plot(locs,pks,'ko');
        title("Firing rate curve, max rate "+num2str(max(temp_curve))+"Hz");
        subplot(313); histogram(temp_ISI,20);
        title("ISI histogram, mean "+num2str(mean(temp_ISI))+"sec, std "+num2str(std(temp_ISI))+"sec");
        drawnow; saveas(CurrentData,"Label "+num2str(temp_label)+", training data "+num2str(ii)+".png");
        close(CurrentData);
    end
end
TRAIN_FEATURE_MATRIX = ...
    [TRAIN_MEAN_RATE,TRAIN_PEAK_MAX,TRAIN_PEAK_MEAN,TRAIN_PEAK_STD,...
    TRAIN_PEAK_INV,TRAIN_NUM_PEAKS,...
    TRAIN_ISI_MEAN,TRAIN_ISI_STD,...
    TRAIN_SR_MEAN,TRAIN_SR_STD,...
    TRAIN_L_MEAN,TRAIN_L_STD,...
    TRAIN_TRANSIENT_RATE,TRAIN_MID_RATE,TRAIN_END_RATE];
% Scale between 0 and 1
scale_min = min(TRAIN_FEATURE_MATRIX,[],1);
scale_max = max(TRAIN_FEATURE_MATRIX,[],1);
TRAIN_FEATURE_MATRIX = (TRAIN_FEATURE_MATRIX-scale_min)./(scale_max-scale_min);
% Test data processing
TEST_BINARY = zeros(num_test,max_time/dt);
TEST_IMAGE = zeros(num_train,121);
for ii = 1:num_test
    % Extract spike timing data
    temp_spike_timing = cell2mat(Xtest(ii));
    temp_spike_timing = temp_spike_timing(temp_spike_timing>START_TIME);
    temp_spike_timing = temp_spike_timing(temp_spike_timing<END_TIME);
    % Digitize the spikes into binary array
    temp_spike_timing_digitized = unique(round(temp_spike_timing/dt));
    TEST_BINARY(ii,temp_spike_timing_digitized) = 1/dt;
    % Extract features from the spike train
    % Mean firing rate (Hz)
    TEST_MEAN_RATE(ii) = length(temp_spike_timing)/max(temp_spike_timing);
    TEST_TRANSIENT_RATE(ii) = length(temp_spike_timing(temp_spike_timing<SPLIT_1))/SPLIT_1;
    TEST_MID_RATE(ii) = length(temp_spike_timing(temp_spike_timing>SPLIT_1 & temp_spike_timing<SPLIT_2))/(SPLIT_2-SPLIT_1);
    TEST_END_RATE(ii) = length(temp_spike_timing(temp_spike_timing>SPLIT_2))/(max_time-SPLIT_2);
    % Instantaneous firing rate (Gaussian window convolusion)
    temp_curve = imgaussfilt(TEST_BINARY(ii,:),5/dt);
    % Maximum firing rate
    TEST_PEAK_MAX(ii) = max(temp_curve);
    % Number of peaks
    [pks,locs] = findpeaks(temp_curve,'MinPeakDistance',100,'MinPeakProminence',0.05);
    TEST_NUM_PEAKS(ii) = size(locs,2);
    TEST_PEAK_MEAN(ii) = mean(pks);
    TEST_PEAK_STD(ii) = std(pks);
        if length(locs)~=1
        TEST_PEAK_INV(ii) = mean(locs(2:end)-locs(1:end-1))*dt;
    else
        TEST_PEAK_INV(ii) = 0;
    end
    % Inter-spike interval, mean and std
    TEST_ISI_MEAN(ii) = mean(temp_ISI);
    TEST_ISI_STD(ii) = std(temp_ISI);
    % Spikelet regularity, mean and std
    temp_L = temp_spike_timing(3:end) - temp_spike_timing(1:end-2);
    temp_SR = 2*(temp_spike_timing(3:end) - temp_spike_timing(2:end-1))...
        ./(temp_spike_timing(3:end) - temp_spike_timing(1:end-2));
    TEST_SR_MEAN(ii) = mean(temp_SR); TEST_SR_STD(ii) = std(temp_SR);
    TEST_L_MEAN(ii) = mean(temp_L); TEST_L_STD(ii) = std(temp_L);
    % Image feature
    %IMG = hist3([temp_L temp_SR],'edges',{0:1.5:15 0:0.2:2}); TEST_IMAGE(ii,:) = IMG(:);
    % Display
    if Display
        CurrentData = figure;
        subplot(311); plot([temp_spike_timing temp_spike_timing],[0 1],'k'); xlim([0 max_time]);
        title("Data "+num2str(ii)+", mean rate "+num2str(TEST_MEAN_RATE(ii))+"Hz");
        subplot(312); plot(temp_curve); hold on; plot(locs,pks,'ko');
        title("Firing rate curve, max rate "+num2str(max(temp_curve))+"Hz");
        subplot(313); histogram(temp_ISI,20);
        title("ISI histogram, mean "+num2str(mean(temp_ISI))+"sec, std "+num2str(std(temp_ISI))+"sec");
        drawnow; saveas(CurrentData,"Test data "+num2str(ii)+".png");
        close(CurrentData);
    end
end
TEST_FEATURE_MATRIX = ...
    [TEST_MEAN_RATE,TEST_PEAK_MAX,TEST_PEAK_MEAN,TEST_PEAK_STD,...
    TEST_PEAK_INV,TEST_NUM_PEAKS,...
    TEST_ISI_MEAN,TEST_ISI_STD,...
    TEST_SR_MEAN,TEST_SR_STD,...
    TEST_L_MEAN,TEST_L_STD,...
    TEST_TRANSIENT_RATE,TEST_MID_RATE,TEST_END_RATE];
%Scale according to training data distribution
TEST_FEATURE_MATRIX = (TEST_FEATURE_MATRIX-scale_min)./(scale_max-scale_min);
%% Raster plot
COLOR = parula(3);
[~,LABELSORT] = sort(TRAIN_LABEL);
figure;
for ii = 1:num_train
    index = LABELSORT(ii);
    temp_spike_timing = cell2mat(Xtrain(index));
    plot([temp_spike_timing temp_spike_timing],[-ii-0.5 -ii+0.5],'Color',COLOR(TRAIN_LABEL(index),:)); hold on;
end
yticks([]);
%% Unsupervised PCA
PCA_res = pca(TRAIN_FEATURE_MATRIX);
TRAIN_PC1 = TRAIN_FEATURE_MATRIX*PCA_res(:,1);
TRAIN_PC2 = TRAIN_FEATURE_MATRIX*PCA_res(:,2);
TRAIN_PC3 = TRAIN_FEATURE_MATRIX*PCA_res(:,3);
PCA_res = pca(TEST_FEATURE_MATRIX);
TEST_PC1 = TEST_FEATURE_MATRIX*PCA_res(:,1);
TEST_PC2 = TEST_FEATURE_MATRIX*PCA_res(:,2);
TEST_PC3 = TEST_FEATURE_MATRIX*PCA_res(:,3);
%% N-fold CV scheme
N = 5;
data_randseq = randperm(num_train);
divide_index = [1 round((1:(N-1))*num_train/N) num_train+1];
CV_LDA = []; CV_QDA = []; CV_RF = [];
for ii = 1:N
    temp_ind = divide_index(ii):divide_index(ii+1)-1;
    valid_ind = data_randseq(temp_ind);
    train_ind = data_randseq(setdiff((1:num_train),temp_ind));
    TRAIN_SET = TRAIN_FEATURE_MATRIX(train_ind,:); TRAIN_SET_LABEL = TRAIN_LABEL(train_ind);
    VALID_SET = TRAIN_FEATURE_MATRIX(valid_ind,:); VALID_SET_LABEL = TRAIN_LABEL(valid_ind);
    % Validation data classification
    Mdl = TreeBagger(200,TRAIN_SET,TRAIN_SET_LABEL,'OOBPrediction','On','Method','classification',...
        'NumPredictorsToSample',round(sqrt(size(TRAIN_SET,2))));
    VALID_RF_CLASS = str2num(cell2mat(predict(Mdl,VALID_SET)));
    CV_RF = [CV_RF sum(VALID_SET_LABEL~=VALID_RF_CLASS')/length(VALID_SET_LABEL)];
    disp(sum(VALID_SET_LABEL~=VALID_RF_CLASS')/length(VALID_SET_LABEL));
    disp(valid_ind(VALID_SET_LABEL~=VALID_RF_CLASS'));
end
%disp(mean(CV_LDA));
%disp(mean(CV_QDA));
disp("Mean CV error " + num2str(mean(CV_RF)));
%% Final prediction
%rng(1); % For reproducibility
Mdl = TreeBagger(200,TRAIN_FEATURE_MATRIX,TRAIN_LABEL,'OOBPrediction','On',...
    'Method','classification','OOBPredictorImportance','on');
view(Mdl.Trees{1},'Mode','graph');
TRAIN_RF_CLASS = str2num(cell2mat(predict(Mdl,TRAIN_FEATURE_MATRIX)));
error = sum(TRAIN_RF_CLASS~=TRAIN_LABEL')/length(TRAIN_RF_CLASS);
figure; subplot(121); scatter3(TRAIN_PC1,TRAIN_PC2,TRAIN_PC3,30,TRAIN_LABEL);
hold on; text(TRAIN_PC1+0.02,TRAIN_PC2+0.02,TRAIN_PC3+0.02,num2str((1:num_train)'),'fontsize',8);
axis xy image; xlabel("PC1"); ylabel("PC2"); zlabel("PC3");
title("Train data labels"); cbh = colorbar('h'); set(cbh,'YTick',[1 2 3]);
set(gca,'Color',[0.4 0.4 0.4]);
subplot(122); scatter3(TRAIN_PC1,TRAIN_PC2,TRAIN_PC3,30,TRAIN_RF_CLASS);
hold on; text(TRAIN_PC1+0.02,TRAIN_PC2+0.02,TRAIN_PC3+0.02,num2str((1:num_train)'),'fontsize',8);
axis xy image; xlabel("PC1"); ylabel("PC2"); zlabel("PC3");
title("Train data classification by RF ensemble"); cbh = colorbar('h'); set(cbh,'YTick',[1 2 3]);
set(gca,'Color',[0.4 0.4 0.4]);
TEST_RF_CLASS = str2num(cell2mat(predict(Mdl,TEST_FEATURE_MATRIX)));
figure; scatter3(TEST_PC1,TEST_PC2,TEST_PC3,30,TEST_RF_CLASS);
hold on; text(TEST_PC1+0.02,TEST_PC2+0.02,TEST_PC3+0.02,num2str((1:num_test)'),'fontsize',8);
axis xy image; xlabel("PC1"); ylabel("PC2"); zlabel("PC3");
title("Test data classificaion by RF ensemble"); cbh = colorbar('h'); set(cbh,'YTick',[1 2 3]);
set(gca,'Color',[0.4 0.4 0.4]);

disp(TEST_RF_CLASS');
%% Variable importance
importance = Mdl.OOBPermutedPredictorDeltaError;
[~,ind] = sort(importance,'descend');
figure; bar(importance(ind));
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = Feature_names(ind);
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
% Group2_ans = [1,3,2,1,2,2,3,3,2,2,1,1,1,1,2];