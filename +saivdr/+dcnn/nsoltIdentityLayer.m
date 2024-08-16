classdef nsoltIdentityLayer < nnet.layer.Layer %#codegen
    %NSOLTIDENTITYLAYER
    %
    % Exported and modified from SaivDr package
    %
    %    https://github.com/msiplab/SaivDr    
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
        
    methods
        function layer = nsoltIdentityLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Name','')
            %addParameter(p,'InputNames',{'in'});
            %addParameter(p,'OutputNames',{'out'});
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Name = p.Results.Name;
            %layer.InputNames = p.Results.InputNames;
            %layer.OutputNames = p.Results.OutputNames;
            layer.Description =  "Identity";
            layer.Type = '';
            
        end
        
        function Z = predict(~, X)           
            % Layer forward function for prediction goes here.
            Z = X;
        end
        function [dLdX] = backward(~,~,~,dLdZ,~)
            % Layer backward function for prediction goes here.
            dLdX = dLdZ;
        end
    end
    
end

