classdef depthSeparationLayer < nnet.layer.Layer %#codegen
    %DEPTHSEPARATION2DLAYER
    %
    % Requirements: MATLAB R2020b
    %
    % Copyright (c) 2020-2021, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    
    properties
        % (Optional) Layer properties.
        
        % Layer properties go here.
    end
    
    methods
        function layer = depthSeparationLayer(numOutputs,varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Name','')
            parse(p,varargin{:})
            
            if numOutputs < 2
                error('numOutputs should be greater than one.');
            end
            
            % Layer constructor function goes here.
            layer.Name = p.Results.Name;
            for idx = 1:numOutputs
                layer.OutputNames{idx} = [ 'out' num2str(idx)];
            end
        end
        
        function varargout = predict(layer, X)
           numOutputs = length(layer.OutputNames);
           varargout = cell(numOutputs,1);
           for idx = 1:numOutputs
                varargout{idx} = X(:,:,idx,:);
           end
        end
        
    end
    
end

