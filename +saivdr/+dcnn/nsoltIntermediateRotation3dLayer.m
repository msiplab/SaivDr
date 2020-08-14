classdef nsoltIntermediateRotation3dLayer < nnet.layer.Layer
    %NSOLTINTERMEDIATEROTATION2DLAYER
    %
    % Requirements: MATLAB R2020a
    %
    % Copyright (c) 2020, Shogo MURAMATSU
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
        NumberOfChannels
        Mus
        Mode
        
        % Layer properties go here.
    end
    
    properties (Learnable)
        Angles
    end
    
    methods
        function layer = nsoltIntermediateRotation3dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'NumberOfChannels',[])
            addParameter(p,'Angles',[])
            addParameter(p,'Mus',[])
            addParameter(p,'Mode','Synthesis')
            addParameter(p,'Name','')
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            layer.Name = p.Results.Name;
            layer.Mode = p.Results.Mode;
            layer.Angles = p.Results.Angles;
            layer.Mus = p.Results.Mus;
            layer.Description = layer.Mode ...
                + " NSOLT intermediate rotation " ...
                + "(ps,pa) = (" ...
                + layer.NumberOfChannels(1) + "," ...
                + layer.NumberOfChannels(2) + ")";
            layer.Type = '';
            
            if isempty(layer.Angles)
                nChsTotal = sum(layer.NumberOfChannels);
                nAngles = (nChsTotal-2)*nChsTotal/8;
                layer.Angles = zeros(nAngles,1);
            end
            
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data (n: # of components)
            % Outputs:
            %         Z           - Outputs of layer forward function
            %  
            import saivdr.dcnn.*
            
            % Layer forward function for prediction goes here.
            nrows = size(X,1);
            ncols = size(X,2);
            nlays = size(X,3);
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);
            nSamples = size(X,5);
            %
            if isempty(layer.Mus)
                musU = 1;
            else
                musU = layer.Mus;
            end
            if isempty(layer.Angles)
                Un = musU*eye(pa);
            else
                anglesU = layer.Angles;
                Un = fcn_orthonormalmatrixgenerate(anglesU,musU);
            end
            Y = permute(X,[4 1 2 3 5]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            if strcmp(layer.Mode,'Analysis')
                Za = Un*Ya;
            elseif strcmp(layer.Mode,'Synthesis')
                Za = Un.'*Ya;
            else
                throw(MException('NsoltLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
            end
            Y(ps+1:ps+pa,:,:,:,:) = reshape(Za,pa,nrows,ncols,nlays,nSamples);
            Z = ipermute(Y,[4 1 2 3 5]);
        end
        
    end

end

