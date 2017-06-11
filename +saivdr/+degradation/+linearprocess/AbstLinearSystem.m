classdef AbstLinearSystem < matlab.System %#codegen
    %ABSTLINEARSYSTEM Abstract class of linear system
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
        DataType             = 'Image';
        %
        EpsOfPowerMethod     = 1e-6;
        UseFileForLambdaMax  = false;
        %
        FileNameForLambdaMax = 'lmax';
    end

    properties
        ProcessingMode = 'Normal'
    end
    
    properties (Hidden, Transient)
        ProcessingModeSet = ...
            matlab.system.StringSet({'Normal','Adjoint'})
        DataTypeSet = ...
            matlab.system.StringSet({'Image','Volumetric Data'})
    end
    
    properties (Hidden, Nontunable)
        ObservedDimension
        OriginalDimension
        LambdaMax
    end
    
    methods (Access = protected, Abstract = true)
        output = normalStepImpl(obj,input)
        output = adjointStepImpl(obj,input)
        flag = isInactiveSubPropertyImpl(obj,propertyName)
        originalDim = getOriginalDimension(obj,ovservedDim)
    end
    
    methods
        function obj = AbstLinearSystem(varargin)
            setProperties(obj,nargin,varargin{:})
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
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end

        function setupImpl(obj,input)
            obj.ObservedDimension = [ size(input,1) size(input,2) ];
            obj.OriginalDimension = getOriginalDimension(...
                obj,obj.ObservedDimension);
            obj.LambdaMax = getMaxEigenValueGram_(...
                obj,obj.OriginalDimension);            
        end
        
        function output = stepImpl(obj,input)
            if strcmp(obj.ProcessingMode,'Adjoint')
                output = adjointStepImpl(obj,input);
            else
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
        
        function lmax = getMaxEigenValueGram_(obj,ovservedDim)
            origDim = getOriginalDimension(obj,ovservedDim);
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
            upst = ones(origDim);
            lpre = 1.0;
            err_ = Inf;
            while ( err_ > obj.EpsOfPowerMethod ) % Power method
                % upst = (P.'*P)*upre
                upre = upst/norm(upst(:));
                v    = normalStepImpl(obj,upre); % P
                upst = adjointStepImpl(obj,v);  % P.'
                n = (upst(:).'*upst(:));
                d = (upst(:).'*upre(:));
                lpst = n/d;
                err_ = norm(lpst-lpre(:))^2;
                lpre = lpst;
            end
            lmax = lpst;
        end
        
    end
    
end
