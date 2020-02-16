classdef AbstIstaImRestoration < matlab.System %~#codegen
    %ABSTISTAIMRESTORATION ISTA-based image restoration
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
    
    properties(Nontunable)
        Synthesizer
        AdjOfSynthesizer
        LinearProcess
    end
    

    
    properties
        StepMonitor
        Eps0   = 1e-6
        Lambda
    end
    
    properties (Logical)
        UseParallel = false;
    end
    
    properties (PositiveInteger)
        MaxIter = 1000
    end
    
    properties (Access = protected,Nontunable)
        AdjLinProcess
    end
    
    properties (Access = protected)
        x
        valueL
        scales
    end
    
    methods
        function obj = AbstIstaImRestoration(varargin)
            setProperties(obj,nargin,varargin{:})
        end
    end
    
    methods(Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            %
            s.Synthesizer = ...
                matlab.System.saveObject(obj.Synthesizer);
            s.AdjOfSynthesizer = ...
                matlab.System.saveObject(obj.AdjOfSynthesizer);
            s.LinearProcess = ...
                matlab.System.saveObject(obj.LinearProcess);
            %
            s.AdjLinProcess = ...
                matlab.System.saveObject(obj.AdjLinProcess);
            s.valueL = obj.valueL;
            s.scales = obj.scales;
            s.x      = obj.x;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.x      = s.x;
            obj.valueL = s.valueL;
            obj.scales = s.scales;
            %
            loadObjectImpl@matlab.System(obj,s,wasLocked);
            %
            obj.Synthesizer = ...
                matlab.System.loadObject(s.Synthesizer);
            obj.AdjOfSynthesizer = ...
                matlab.System.loadObject(s.AdjOfSynthesizer);
            obj.LinearProcess = ...
                matlab.System.loadObject(s.LinearProcess);
            %
            obj.AdjLinProcess = ...
                matlab.System.loadObject(s.AdjLinProcess);
            
        end
        
        function validatePropertiesImpl(obj)
            if isempty(obj.Synthesizer)
                me = MException('SaivDr:InstantiationException',...
                    'Synthesizer must be given.');
                throw(me)
            end
            if isempty(obj.AdjOfSynthesizer)
                me = MException('SaivDr:InstantiationException',...
                    'AdjOfSynthesizer must be given.');
                throw(me)
            end
            if isempty(obj.LinearProcess)
                me = MException('SaivDr:InstantiationException',...
                    'LinearProcess must be given.');
                throw(me)
            end
            if ~strcmp(get(obj.LinearProcess,'ProcessingMode'),'Normal')
                error('SaivDr: Invalid processing mode')
            end
        end
        
        function setupImpl(obj,srcImg) 
            obj.AdjLinProcess = clone(obj.LinearProcess);
            set(obj.AdjLinProcess,'ProcessingMode','Adjoint');            
            obj.x = srcImg;
            obj.valueL  = getLipschitzConstant_(obj);            
        end        
        
        function value = getLipschitzConstant_(obj)
            B_ = get(obj.Synthesizer,'FrameBound');
            step(obj.LinearProcess,obj.x);
            value = B_*get(obj.LinearProcess,'LambdaMax');
        end       
        
        function N = getNumInputsImpl(~)
            N = 1;
        end
        
        function N = getNumOutputsimpl(~)
            N = 1;
        end
        
    end
    
    methods (Static = true, Access = protected)
        % Soft shrink
        function outputcf = softshrink_(inputcf,threshold)
            % Soft-thresholding shrinkage
            nc = abs(inputcf)-threshold;
            %nc(nc<0) = 0;
            nc = (nc+abs(nc))/2;
            outputcf = sign(inputcf).*nc;
        end
        
    end
    
end

