classdef nmseReg < nnet.layer.RegressionLayer
    properties
%         mask;
    end
    
    methods
        function layer = nmseReg(name)           
            % (Optional) Create a myRegressionLayer.
            % Layer constructor function goes here.
            layer.Name = name;
            layer.Description = 'NMSE regression';
        end

        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the 
            % training targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T

            % Layer forward loss function goes here.
            Y = squeeze(Y);
            T = squeeze(T);
            diff = Y - T;
            num = diag( transpose(diff)*diff );
            den = diag( transpose(T)*T );
            nmseVec = num./den;
            loss = ( sum(nmseVec) )/(2*size(T,2));
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Backward propagate the derivative of the loss function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network (size is equal to 1x1xRxN if the output comes from FC layer)
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the predictions Y        

            % Layer backward loss function goes here.
            N = size(Y,4);
            Y_squ = squeeze(Y);
            T_squ = squeeze(T);
            diff = Y_squ - T_squ;
            normResp = diag( transpose(T_squ)*T_squ );% norm of responses
            invNormResp = 1./(N*normResp);
            D = diag(invNormResp);
            dLdY = diff*D;
            dLdY = reshape(dLdY,size(Y));

        end
    end
end
