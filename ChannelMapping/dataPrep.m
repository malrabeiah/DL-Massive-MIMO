function [dataset, options] = dataPrep(rawData, options)
%==========================================================================
%
% INPUTS:
%   rawData: unprocessed channel complex-valued data, comes in the form of 
%   a 3D array: # of Antennas x # of subcarriers x # of samples.
%   options: a structure defining the basic experiment setting and
%   data-related parameters.
% OUTPUTS:
%   dataset: a structure with three fields that:
%   training, validation, and stats.
%   each is a 2x1 cell array. The first entry of the cell array is a 4D  
%   array of inputs and the second is a 3D array of GT outputs. 
%   options: updated options structure.
%
%==========================================================================

dataset = struct();

fprintf('Normalizing the absolute of the raw data to the range -1 to 1 \n');
absRawData = abs(rawData);
dataset.stats.absMaxValue = max( absRawData(:) );
rawData = rawData/dataset.stats.absMaxValue;

% Preparing dataset:
% ------------------

fprintf('Separating real from imaginary\n')
switch options.inputType        
    case 'FCN'
        for sample = 1:options.numOfSamples
            x = rawData(:,:,sample);
            newX = cat(3, real(x), imag(x));
            dataset.sample{sample} = newX;
        end
        
end

% Generating training and validation datasets:
% --------------------------------------------

shuffledInd = randperm(options.numOfSamples);
dataset.sample = dataset.sample(shuffledInd);% Shuffled dataset
options.numOfTrain = floor( (1-options.valPercentage)*options.numOfSamples );
options.numOfVal = options.numOfSamples - options.numOfTrain;
train = cell(2,1);
val = cell(2,1);
trainTemp = dataset.sample( 1:options.numOfTrain );
valTemp = dataset.sample( options.numOfTrain+1:end );
if options.datasetSizeRatio ~= 1 % Using a fraction of the data points
     shuffleTrain = randperm(options.numOfTrain);
     options.numOfTrain = floor( options.datasetSizeRatio*options.numOfTrain );% Reduce number of data samples
     fprintf('New training set size: %d \n', options.numOfTrain)
     trainTemp = trainTemp( shuffleTrain(1:options.numOfTrain) );
end
dataset.sample = [];

switch options.corruption
    case 'FCNMasking'
        numOfAnt = prod( options.antDim );
        numOfActElem = floor( numOfAnt/options.sampling(1,1) );
        if options.fixedMask == 0 
            maskVec = zeros(numOfAnt,1);
            oneLoc = randperm(length(maskVec));
            maskVec(oneLoc(1:numOfActElem)) = 1;
            maskMat = repmat(maskVec,[1,options.numOfSub]);
            mask(:,:,1) = maskMat;
            mask(:,:,2) = maskMat;
            options.mask = mask;
        elseif options.fixedMask == 1 && isempty(options.mask)
            maskVec = zeros(numOfAnt,1);
            oneLoc = randperm(length(maskVec));
            maskVec(oneLoc(1:numOfActElem)) = 1;
            maskMat = repmat(maskVec,[1,options.numOfSub]);
            mask(:,:,1) = maskMat;
            mask(:,:,2) = maskMat;
	    options.mask = mask;
            loc = find(options.mask(:,1,1) == 1)
        else
            mask = options.mask;
            loc = find(options.mask(:,1,1) == 1)
        end
        fprintf('number of ones: %d \n',sum(options.mask(:,1,1)));
        options.figCount = options.figCount+1;
        figure(options.figCount); imshow(mask(:,:,1)); title('Visualization of elements mask')
        disp(['Create training pairs using ', options.corruption, ' corruption'])
        X = zeros(1,1,options.inputSize(3),options.numOfTrain);
        Y = zeros(options.numOfTrain,options.inputSize(3));
        for i = 1:options.numOfTrain% looping over data samples
            x = trainTemp{i};
            X1 = mask.*x(:,1:options.numOfSub,:);
            X(1,1,:,i) = [reshape(X1(:,:,1),[numOfAnt*options.numOfSub,1]);...
                reshape(X1(:,:,2),[numOfAnt*options.numOfSub,1])];%
            Y1 = x(:,1:options.numOfSub,:);
            Y(i,:) = [reshape(Y1(:,:,1),[1,numOfAnt*options.numOfSub]),...
                      reshape(Y1(:,:,2),[1,numOfAnt*options.numOfSub])];% 
        end
        train{1,1} = single( X );
        train{2,1} = single( Y );
        disp('Create validation set')
        X = zeros(1,1,options.inputSize(3),options.numOfVal);
        Y = zeros(options.numOfVal,options.inputSize(3));
        for i = 1:options.numOfVal% looping over data samples
            x = valTemp{i};
            X1 = mask.*x(:,1:options.numOfSub,:);
            X(1,1,:,i) = [reshape(X1(:,:,1),[numOfAnt*options.numOfSub,1]);...
                          reshape(X1(:,:,2),[numOfAnt*options.numOfSub,1])];%
            Y1 = x(:,1:options.numOfSub,:);
            Y(i,:) = [reshape(Y1(:,:,1),[1,numOfAnt*options.numOfSub]),...
                      reshape(Y1(:,:,2),[1,numOfAnt*options.numOfSub])];% 
        end
        val{1,1} = single( X );
        val{2,1} = single( Y );

end

dataset.train = train;
dataset.val = val;

end% End of function
