classdef (Abstract) AbstLinearSystem < matlab.System %#codegen
    %ABSTLINEARSYSTEM Abstract class of linear system
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2020, Shogo MURAMATSU
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
        DataType             = 'Image';
        %
        TolOfPowerMethod     = 1e-8;
        MaxIterOfPowerMethod = 1e+6;
        UseFileForLambdaMax  = false;
        %
        FileNameForLambdaMax = 'lmax';
    end

    properties
        ProcessingMode = 'Normal'
    end
    
    properties (Hidden, Transient)
        ProcessingModeSet = ...
            matlab.system.StringSet({'Normal','Forward','Adjoint'})
        DataTypeSet = ...
            matlab.system.StringSet({'Image','Volumetric Data'})
    end
    
    properties (Hidden, Nontunable)
        ObservedDimension
        OriginalDimension
        LambdaMax
    end
    
    properties (Access = private)
        scurr
    end
    
    %{
    properties(DiscreteState)
        State
    end
    %}
    
    methods (Access = protected, Abstract = true)
        output = normalStepImpl(obj,input)
        output = adjointStepImpl(obj,input)
        flag = isInactiveSubPropertyImpl(obj,propertyName)
        originalDim = getOriginalDimension(obj,observedDim)
    end
    
    methods
        function obj = AbstLinearSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            obj.scurr = rng;
        end
    end
    
    methods (Access = protected)

        function flag = isInactivePropertyImpl(obj,propertyName)
            if strcmp(propertyName,'FileNameForLambdaMax')
                flag = ~obj.UseFileForLambdaMax;
            else
                flag = isInactiveSubPropertyImpl(obj,propertyName);
            end
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);    
            s.scurr = obj.scurr;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.scurr = s.scurr;
            loadObjectImpl@matlab.System(obj,s,wasLocked);           
        end

        function setupImpl(obj,input)
            if strcmp(obj.DataType,'Image')
                obj.ObservedDimension = [ size(input,1) size(input,2) ];
            else % Volumetric Data
                obj.ObservedDimension = [ ...
                    size(input,1) ...
                    size(input,2) ...
                    size(input,3)];
            end
            obj.OriginalDimension = getOriginalDimension(...
                obj,obj.ObservedDimension);
            obj.LambdaMax = getMaxEigenValueGram_(...
                obj,obj.OriginalDimension);            
        end
        
        function output = stepImpl(obj,input)
            if strcmp(obj.ProcessingMode,'Adjoint')
                output = adjointStepImpl(obj,input);
            else % Normal
                output = normalStepImpl(obj,input);
            end
        end
        
        function N = getNumInputsImpl(~)
            N = 1;
        end
        
        function N = getNumOutputsImpl(~)
            N = 1;
        end
        
    end
    
    methods (Access = private)
        
        function lmax = getMaxEigenValueGram_(obj,observedDim)
            origDim = getOriginalDimension(obj,observedDim);
            if obj.UseFileForLambdaMax
                lmaxfile = obj.FileNameForLambdaMax;
                if exist(lmaxfile,'file') == 2
                    data = load(lmaxfile,'lmax');
                    lmax = data.lmax;
                else
                    if labindex == 1
                        lmax = getLambdaMax_(obj,origDim);
                        save(lmaxfile,'lmax')
                    end
                    labBarrier
                    if labindex ~= 1
                        data = load(lmaxfile,'lmax');
                        lmax = data.lmax;
                    end
                end
            else
                lmax = getLambdaMax_(obj,origDim);
            end
        end
        
        function lmax = getLambdaMax_(obj,origDim)
            rng(obj.scurr);
            upst = rand(origDim);
            lpre = 1.0;
            err_ = Inf;
            cnt_ = 0;
            % Power method
            while ( err_ > obj.TolOfPowerMethod ) % || ...
                %cnt_ >= obj.MaxIterOfPowerMethod )
                cnt_ = cnt_ + 1;
                % upst = (P.'*P)*upre
                upre = upst/norm(upst(:));
                v    = normalStepImpl(obj,upre); % P
                upst = adjointStepImpl(obj,v);   % P.'
                n = (upst(:).'*upst(:));
                d = (upst(:).'*upre(:));
                lpst = n/d;
                err_ = abs(lpst-lpre)/abs(lpre);
                lpre = lpst;
                if cnt_ >= obj.MaxIterOfPowerMethod  
                    warning('# of iterations reached to MaxIterPowerMethod');
                    break;
                end
            end
            lmax = lpst;
        end
        
    end
    
end
