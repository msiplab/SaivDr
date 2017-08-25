classdef ComplexBlurSystem < ...
        saivdr.degradation.linearprocess.AbstLinearSystem %#codegen
    %COMPLEXBLURSYSTEM Bluring process
    %   
    % SVN identifier:
    % $Id: BlurSystem.m 714 2015-07-30 21:44:30Z sho $
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
        BlurType = 'Identical'
        BoundaryOption = 'Value'
        % Parameters available only for Gaussian
        SizeOfKernel
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
    
    properties(SetAccess = protected, GetAccess = public)
        BlurKernel = 1
    end

    properties(Access = protected)
        offset
    end
    
    methods
        % Constractor
        function obj = ComplexBlurSystem(varargin)
            obj = ...
                obj@saivdr.degradation.linearprocess.AbstLinearSystem(...
                varargin{:});
        end
                
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@...
                saivdr.degradation.linearprocess.AbstLinearSystem(obj);
            s.BlurKernel = obj.BlurKernel;
            s.offset     = obj.offset;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.BlurKernel = s.BlurKernel;
            obj.offset     = s.offset;    
            loadObjectImpl@...
                saivdr.degradation.linearprocess.AbstLinearSystem(obj,s,wasLocked);
        end
        
        function flag = isInactiveSubPropertyImpl(obj,propertyName)
            if strcmp(propertyName,'SizeOfKernel')
                flag = ~(strcmp(obj.BlurType,'Gaussian') || ...
                    strcmp(obj.BlurType,'Average'));
            elseif strcmp(propertyName,'SigmaOfGaussianKernel')
                flag = ~strcmp(obj.BlurType,'Gaussian');
            elseif strcmp(propertyName,'CostumKernel')
                flag = ~strcmp(obj.BlurType,'Custom');
            elseif strcmp(propertyName,'BoundaryValue')
                flag = ~strcmp(obj.BoundaryOption,'Value');
            else
                flag = false;
            end
        end
        
        function setupImpl(obj,input)
            
            switch obj.BlurType
                case {'Identical'}
                    obj.BlurKernel = 1;
                case {'Average'}
                    if isempty(obj.SizeOfKernel)
                        obj.SizeOfKernel  = [ 3 3 ];
                    end
                    obj.BlurKernel = fspecial('average',obj.SizeOfKernel);
                case {'Gaussian'}
                    if isempty(obj.SigmaOfGaussianKernel)
                        obj.SigmaOfGaussianKernel = 2.0;
                    end
                    if isempty(obj.SizeOfKernel)
                        obj.SizeOfKernel  = ...
                            2*round(4*obj.SigmaOfGaussianKernel)+1;
                    end
                    obj.BlurKernel = fspecial('gaussian',...
                        obj.SizeOfKernel, obj.SigmaOfGaussianKernel);
                case {'Custom'}
                    if isempty(obj.CustomKernel)
                        me = MException('SaivDr:InvalidOption',...
                            'CustomKernel should be specified.');
                        throw(me);
                    else
                        obj.BlurKernel = obj.CustomKernel;
                    end
                otherwise
                    me = MException('SaivDr:InvalidOption',...
                        'Invalid blur type');
                    throw(me);
            end
            obj.offset = mod(size(obj.BlurKernel)+1,2);
            %
            setupImpl@saivdr.degradation.linearprocess. ...
                AbstLinearSystem(obj,input);
        end
        
        function output = normalStepImpl(obj,input)
            if strcmp(obj.BoundaryOption,'Value')
                output = imfilter(input,obj.BlurKernel,'conv',...
                     obj.BoundaryValue);
            else
                output = imfilter(input,obj.BlurKernel,'conv',...
                    lower(obj.BoundaryOption));
            end
        end
        
        function output = adjointStepImpl(obj,input)
            if strcmp(obj.BoundaryOption,'Value')            
                output = imfilter(input,obj.BlurKernel,'corr',...
                    obj.BoundaryValue);
            else
                output = imfilter(input,obj.BlurKernel,'corr',...
                    lower(obj.BoundaryOption));                
            end
        end

        function originalDim = getOriginalDimension(~,ovservedDim)
            originalDim = ovservedDim;
        end        
      
    end
    
end
