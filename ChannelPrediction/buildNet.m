function net = buildNet(type, preTrained, inputSize, numOfSub)
%==========================================================================
% buildNet constucts the neural network
%
%
%==========================================================================

if preTrained
   disp('No preTrained models available')
end

% Building Network:
% -----------------

switch type
    case 'FCN'
        vecInput = imageInputLayer(inputSize, 'Name', 'input');
        fc1 = fullyConnectedLayer(1024,'Name','fc1');% o/p size is 16x16x128
        relu1 = reluLayer('Name','relu1');
        dropOut1 = dropoutLayer('Name', 'dropOut1');
        fc2 = fullyConnectedLayer(4096,'Name','fc2');
        relu2 = reluLayer('Name','relu2');
        dropOut2 = dropoutLayer('Name', 'dropOut2');
        fc3 = fullyConnectedLayer(4096,'Name','fc3');
        relu3 = reluLayer('Name','relu3');
        dropOut3 = dropoutLayer('Name', 'dropOut3');
        fc5 = fullyConnectedLayer(2048, 'Name', 'fc5');
        regOutput = nmseReg('regOutput');

        layers = [...
                  vecInput
                  fc1
                  relu1
                  fc2
                  relu2
                  fc3
                  relu3
                  fc5
                  regOutput]

        net = layerGraph(layers);

end
