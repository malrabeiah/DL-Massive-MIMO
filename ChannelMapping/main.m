%===============================================================%
% Author: 
% Muhammad Alrabeiah
% School of ECEE, ASU
% Tempe, AZ, USA
%===============================================================%

clc
clear
close all

sampling = [8,1]                
R_predAve = [];
R_targAve = [];
R_lowAve = []; 
datasetSizeRatio = [0.001,0.005,0.01,0.05,0.1,0.3,0.5,0.7,1];
options.valNMSE = [];
options.figCount = 0;
datasetSize = [];
options.mask = [];
for s = 1:size(datasetSizeRatio,2)
    options.antDim = [16,4];
    options.inputType = 'FCN';
    options.dataWhitening = false;
    options.numOfSub = 16;
    options.sampling = sampling;
    options.numOfSamples = [];
    options.scalingFactor = 1;% A factor for scaling the standardized data smaples
    options.numOfChannels = 2;% third dimension of the input sample
    options.valPercentage = 0.2;
    options.corruption = 'FCNMasking';
    options.fixedMask = 1;
    options.rawDataFile1 = '';% Path to dataset, example: RawData_64DistAnt_Indoor2_4GHz_16Sub_1Path_from1to502row.mat'
    options.expTag = '16Sub_64DistAnt_1Path_Case_Spatial24GHz_FCNModel_Exp3';
    options.trainedNetDir = '~/Documents/MATLAB/MassiveMIMO/Networks/';% path to where to store the trained model
    options.learningRate = 1e-3;
    options.dropFactor = 0.1;
    options.weightDecay = 1e-4;
    options.learnRateSch = 100;
    options.maxNumEpochs = 17;
    options.batchSize = 1000;
    options.valFreq = 50;
    switch options.inputType
        case 'Vectorized'
            options.inputSize = [options.antDim(1)*options.antDim(2),...
                                 options.numOfSub, 2];
        case 'Planner'
            options.inputSize = options.antDim;
        case 'FCN'
            options.inputSize = [1,1,options.antDim(1)*options.antDim(2)*options.numOfSub*2];
    end
    
    % Preparing dataset:
    % ------------------

    fprintf('Experiment: %s --- Sampling of %f \n', options.expTag, options.sampling(1));
    load(options.rawDataFile1);
    options.numOfSamples = size(rawData.channel, 3);
    options.datasetSizeRatio = datasetSizeRatio(s);
    datasetSize(s) = floor(options.datasetSizeRatio*options.numOfSamples);
    fprintf('Preprocessing: preparing training and testing data using %f of train samples \n', options.datasetSizeRatio);
    [dataset,options] = dataPrep(rawData.channel, options);
    options.stats = dataset.stats;

    % Network Construction:
    % --------------------

    net = buildNet('FCN',... % cnnAuto2_VectInput_AllConv, FCN
                   false,...
                   options.inputSize,...
                   options.numOfSub);

    % Network training:
    % -----------------
    trainOpt = trainingOptions('adam', ...
                'InitialLearnRate',options.learningRate,...
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropFactor',options.dropFactor, ...
                'LearnRateDropPeriod',options.learnRateSch, ...
                'L2Regularization',options.weightDecay,...
                'MaxEpochs',options.maxNumEpochs, ...
                'MiniBatchSize',options.batchSize, ...
                'Shuffle','every-epoch',...
                'ValidationData',dataset.val,...
                'ValidationFrequency',options.valFreq,...
                'ExecutionEnvironment','gpu',...
                'ValidationPatience', 5,...% Disables automatic training break-off
                'Plots','none');

    gpuDevice(1)
    [trainedNet, trainingInfo] = trainNetwork(dataset.train{1,:}, dataset.train{2,:}, net, trainOpt);
    nanLoc = isnan(trainingInfo.ValidationLoss);
    valNMSE = trainingInfo.ValidationLoss(nanLoc);
    options.valNMSE{s} = valNMSE;

    % Performance Evaluation:
    % -----------------------

    disp('Computing rates')
    X = dataset.val{1,1};
    Y = dataset.val{2,1};
    numOfSamples = size(X, 4);
    R_pred = [];
    R_targ = [];
    R_lower = [];
    Ind = zeros(1,numOfSamples);% Beam-vector index
    switch options.inputType            
        case 'FCN'
            half = options.inputSize(3)/2;
            inputChMatComp = squeeze( complex(X(:,:,1:half,:),X(:,:,half+1:end,:)) );
            predChVol = trainedNet.predict(X);
            predChMatComp = transpose(complex(predChVol(:,1:half),predChVol(:,half+1:end)));
            targChMatComp = transpose(complex(Y(:,1:half), Y(:,half+1:end)));
            inputH = reshape(inputChMatComp, [prod(options.antDim),options.numOfSub,numOfSamples]);
            predH = reshape(predChMatComp, [prod(options.antDim),options.numOfSub,numOfSamples]);
            targH = reshape(targChMatComp, [prod(options.antDim),options.numOfSub,numOfSamples]);
            for sample = 1:numOfSamples
                pred = diag( abs(targH(:,:,sample)'*predH(:,:,sample)).^2)./diag( abs(predH(:,:,sample)'*predH(:,:,sample)) );
                targ = diag( abs(targH(:,:,sample)'*targH(:,:,sample)).^2)./diag( abs(targH(:,:,sample)'*targH(:,:,sample)) );
                lower = diag( abs(targH(:,:,sample)'*inputH(:,:,sample)).^2)./diag( abs(inputH(:,:,sample)'*inputH(:,:,sample)) );
                R_pred(sample) = mean( log2(1 + pred) );
                R_targ(sample) = mean( log2(1 + targ) );
                R_lower(sample) = mean( log2(1 + lower) );
            end
    end
    R_predAve(end+1) = mean(R_pred);
    R_targAve(end+1) = mean(R_targ);
    R_lowAve(end+1) = mean(R_lower);
    fprintf('Average achievable rate(predicted channel): %4.2f. Average achievable rate (upper bound): %4.2f. Average lower bound: %4.2f \n',...
             R_predAve(end), R_targAve(end), R_lowAve(end))
   
         

end

 options.figCount = options.figCount + 1;
 h = figure(options.figCount);
 plot(datasetSize, R_predAve, '-bo', datasetSize, R_targAve, '--r', datasetSize, R_lowAve, '--k')
 title(['Model: FCAE ', options.expTag]); xlabel('Dataset size'); ylabel('Achievable rate')
 savefig(h,'DaatasetVsRate')

