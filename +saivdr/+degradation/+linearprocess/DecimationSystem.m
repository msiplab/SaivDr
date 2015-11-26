classdef DecimationSystem < ...
        saivdr.degradation.linearprocess.AbstLinearSystem %#codegen
    %DECIMATIONSYSTEM Decimation rpocess
    %   
    % SVN identifier:
    % $Id: DecimationSystem.m 715 2015-07-31 01:53:27Z sho $
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2014-2015, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627    
    %       
    properties (Nontunable)
        HorizontalDecimationFactor = 2;
        VerticalDecimationFactor = 2;
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
           %
           setupImpl@saivdr.degradation.linearprocess. ...
               AbstLinearSystem(obj,input)
        end
        
        function output = normalStepImpl(obj,input)
            nCmps  = size(input,3);
            nRows  = ceil(size(input,1)/obj.VerticalDecimationFactor);
            nCols  = ceil(size(input,2)/obj.HorizontalDecimationFactor);
            output = zeros(nRows,nCols,nCmps);
            for iCmp = 1:nCmps
                cmpin = input(:,:,iCmp);
                if strcmp(obj.BoundaryOption,'Value')
                    v = imfilter(cmpin,obj.blurKernel,'conv',...
                        obj.BoundaryValue); 
                else
                    v = imfilter(cmpin,obj.blurKernel,'conv',...
                        lower(obj.BoundaryOption)); 
                end
                cmpout = downsample(downsample(v,obj.VerticalDecimationFactor).',...
                    obj.HorizontalDecimationFactor).'; % Downsampling
                output(:,:,iCmp) = cmpout;
            end
        end
        
        function output = adjointStepImpl(obj,input)
            nCmps  = size(input,3);
            nRows  = size(input,1)*obj.VerticalDecimationFactor;
            nCols  = size(input,2)*obj.HorizontalDecimationFactor;
            output = zeros(nRows,nCols,nCmps);            
            for iCmp = 1:nCmps            
                cmpin = input(:,:,iCmp);
                v = upsample(upsample(cmpin,obj.VerticalDecimationFactor,obj.offset(1)).',...
                    obj.HorizontalDecimationFactor,obj.offset(2)).'; % Upsampling
                if strcmp(obj.BoundaryOption,'Value')
                    cmpout = imfilter(v,obj.blurKernel,'corr',...
                        obj.BoundaryValue); 
                else
                    cmpout = imfilter(v,obj.blurKernel,'corr',...
                        lower(obj.BoundaryOption));
                end 
                output(:,:,iCmp) = cmpout;
            end
        end
        
        function originalDim = getOriginalDimension(obj,ovservedDim)
                originalDim = ...
                    obj.decimationFactor.*ovservedDim(1:2);
        end
        
    end
    
end
