classdef DecimationSystem < ...
        saivdr.degradation.linearprocess.AbstLinearSystem %#codegen
    %DECIMATIONSYSTEM Decimation rpocess
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2017, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    properties (Nontunable)
        HorizontalDecimationFactor = 2;
        VerticalDecimationFactor = 2;
        DepthDecimationFactor = 2;
        BlurType = 'Identical';
        BoundaryOption = 'Value'
        % Parameters available only for Gaussian
        SizeOfGaussianKernel
        SigmaOfGaussianKernel
        CustomKernel = [];
        % Parameter available only for Value
        BoundaryValue = 0
    end
    
    properties (Hidden, Transient)
        BlurTypeSet = ...
            matlab.system.StringSet({...
            'Identical',...
            'Gaussian',...
            'Average',...
            'Custom'})
        BoundaryOptionSet = ...
            matlab.system.StringSet({...
            'Value',...
            'Symmetric',...
            'Replicate',...
            'Circular'})
    end
    
    properties(Access = protected)
        blurKernel = 1;
        decimationFactor
        offset
    end
    
    methods
        
        % Constractor
        function obj = DecimationSystem(varargin)
            obj = ...
                obj@saivdr.degradation.linearprocess.AbstLinearSystem(...
                varargin{:});
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@...
                saivdr.degradation.linearprocess.AbstLinearSystem(obj);
            s.blurKernel = obj.blurKernel;
            s.decimationFactor = obj.decimationFactor;
            s.offset = obj.offset;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.blurKernel = s.blurKernel;
            obj.decimationFactor = s.decimationFactor;
            obj.offset = s.offset;
            loadObjectImpl@...
                saivdr.degradation.linearprocess.AbstLinearSystem(obj,s,wasLocked);
        end
        
        function flag = isInactiveSubPropertyImpl(obj,propertyName)
            if strcmp(propertyName,'SizeOfGaussianKernel')
                flag = ~strcmp(obj.BlurType,'Gaussian');
            elseif strcmp(propertyName,'SigmaOfGaussianKernel')
                flag = ~strcmp(obj.BlurType,'Gaussian');
            elseif strcmp(propertyName,'CustomKernel')
                flag = ~strcmp(obj.BlurType,'Custom');
            elseif strcmp(propertyName,'BoundaryValue')
                flag = ~strcmp(obj.BoundaryOption,'Value');
            else
                flag = false;
            end
        end
        
        function setupImpl(obj,input)
            
            if strcmp(obj.DataType,'Image')
                obj.decimationFactor = ...
                    [obj.VerticalDecimationFactor obj.HorizontalDecimationFactor];
                switch obj.BlurType
                    case {'Identical'}
                        obj.blurKernel = 1;
                    case {'Average'}
                        obj.blurKernel = fspecial('average',obj.decimationFactor);
                    case {'Gaussian'}
                        if isempty(obj.SizeOfGaussianKernel)
                            obj.SizeOfGaussianKernel  = 4*obj.decimationFactor+1;
                        end
                        if isempty(obj.SigmaOfGaussianKernel)
                            obj.SigmaOfGaussianKernel = ...
                                max(obj.decimationFactor);
                            % (max(obj.decimationFactor)/pi)*sqrt(2*log(2));
                        end
                        obj.blurKernel = fspecial('gaussian',...
                            obj.SizeOfGaussianKernel, obj.SigmaOfGaussianKernel);
                    case {'Custom'}
                        if isempty(obj.CustomKernel)
                            me = MException('SaivDr:InvalidOption',...
                                'CustomKernel should be specified.');
                            throw(me);
                        else
                            obj.blurKernel = obj.CustomKernel;
                        end
                    otherwise
                        me = MException('SaivDr:InvalidOption',...
                            'Invalid blur type');
                        throw(me);
                end
                obj.offset = mod(size(obj.blurKernel)+1,2);                
            else % Volumetric Data
                obj.decimationFactor = [...
                    obj.VerticalDecimationFactor ...
                    obj.HorizontalDecimationFactor ...
                    obj.DepthDecimationFactor ];
                switch obj.BlurType
                    case {'Identical'}
                        obj.blurKernel = 1;
                    case {'Average'}
                        obj.blurKernel = ...
                            ones(obj.decimationFactor)/...
                            prod(obj.decimationFactor);
                    case {'Gaussian'}
                        if isempty(obj.SizeOfGaussianKernel)
                            obj.SizeOfGaussianKernel = ...
                                4*obj.decimationFactor+1;
                        end
                        if isempty(obj.SigmaOfGaussianKernel)
                            obj.SigmaOfGaussianKernel = ...
                                max(obj.decimationFactor);
                        end
                        hs = (obj.SizeOfGaussianKernel-1)/2;
                        sg = obj.SigmaOfGaussianKernel;
                        [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
                        kernel_ = exp(-(X.^2+Y.^2+Z.^2)/(2*sg^2));
                        obj.blurKernel = kernel_/sum(kernel_(:));
                    case {'Custom'}
                        if isempty(obj.CustomKernel)
                            me = MException('SaivDr:InvalidOption',...
                                'CustomKernel should be specified.');
                            throw(me);
                        else
                            obj.blurKernel = obj.CustomKernel;
                        end
                    otherwise
                        me = MException('SaivDr:InvalidOption',...
                            'Invalid blur type');
                        throw(me);
                end
                obj.offset(1) = mod(size(obj.blurKernel,1)+1,2);
                obj.offset(2) = mod(size(obj.blurKernel,2)+1,2);
                obj.offset(3) = mod(size(obj.blurKernel,3)+1,2);
            end
            %
            setupImpl@saivdr.degradation.linearprocess. ...
                AbstLinearSystem(obj,input)
        end
        
        function output = normalStepImpl(obj,input)
            if strcmp(obj.BoundaryOption,'Value')
                v = imfilter(input,obj.blurKernel,'conv',...
                    obj.BoundaryValue);
            elseif strcmp(obj.BlurType,'Gaussian')
                if strcmp(obj.DataType,'Image')
                    v = imgaussfilt(input,...
                        obj.SigmaOfGaussianKernel,...
                        'FilterSize',obj.SizeOfGaussianKernel,...
                        'Padding',lower(obj.BoundaryOption));
                else
                    v = imgaussfilt3(input,...
                        obj.SigmaOfGaussianKernel,...
                        'FilterSize',obj.SizeOfGaussianKernel,...
                        'Padding',lower(obj.BoundaryOption));
                end
            else
                v = imfilter(input,obj.blurKernel,'conv',...
                    lower(obj.BoundaryOption));
            end
            if strcmp(obj.DataType,'Image')
                output = ...
                    permute(downsample(...
                    permute(downsample(v,...
                    obj.VerticalDecimationFactor),[2 1 3]),...
                    obj.HorizontalDecimationFactor),[2 1 3]); % Downsampling
            else
                output = ...
                    shiftdim(downsample(...
                    shiftdim(downsample(...
                    shiftdim(downsample(v,...
                    obj.VerticalDecimationFactor),1),...
                    obj.HorizontalDecimationFactor),1),...
                    obj.DepthDecimationFactor),1);
            end
        end
        
        function output = adjointStepImpl(obj,input)
            if strcmp(obj.DataType,'Image')
                v = permute(upsample(...
                    permute(upsample(input,...
                    obj.VerticalDecimationFactor,obj.offset(1)),[2 1 3]),...
                    obj.HorizontalDecimationFactor,obj.offset(2)),[2 1 3]); % Upsampling
            else % Volumetric Data
                v = shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(input,...
                    obj.VerticalDecimationFactor,obj.offset(1)),1),...
                    obj.HorizontalDecimationFactor,obj.offset(2)),1),...
                    obj.DepthDecimationFactor,obj.offset(3)),1);
            end
            if strcmp(obj.BoundaryOption,'Value')
                output = imfilter(v,obj.blurKernel,'corr',...
                    obj.BoundaryValue);
            elseif strcmp(obj.BlurType,'Gaussian')
                if strcmp(obj.DataType,'Image')
                    output = imgaussfilt(v,...
                        obj.SigmaOfGaussianKernel,...
                        'FilterSize',obj.SizeOfGaussianKernel,...
                        'Padding',lower(obj.BoundaryOption));
                else
                    output = imgaussfilt3(v,...
                        obj.SigmaOfGaussianKernel,...
                        'FilterSize',obj.SizeOfGaussianKernel,...
                        'Padding',lower(obj.BoundaryOption));
                end
            else
                output = imfilter(v,obj.blurKernel,'corr',...
                    lower(obj.BoundaryOption));
            end
        end
        
        function originalDim = getOriginalDimension(obj,observedDim)
            if strcmp(obj.DataType,'Image')
                originalDim = ...
                    obj.decimationFactor.*observedDim(1:2);
            else
                originalDim = ...
                    obj.decimationFactor.*observedDim(1:3);
            end
        end
        
    end
    
end
