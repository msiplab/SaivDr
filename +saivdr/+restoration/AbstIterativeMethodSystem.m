classdef (Abstract) AbstIterativeMethodSystem < matlab.System
    %ABSTITERATIVERESTORATIONSYSTEM Abstract class for iterative methods
    %
    % Requirements: MATLAB R2018a
    %
    % Copyright (c) 2018, Shogo MURAMATSU
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

    methods (Abstract)
        [coefs,scales] = getCoefficients(obj)
    end
    
    properties (Nontunable, Abstract)
        GaussianDenoiser % Set of Gaussian denoisers {G_R(),...}
    end
    
    properties (Constant)
        FORWARD = 1
        ADJOINT = 2
    end

    properties
        Dictionary       % Set of synthesis and analysis dictionary {D, D'}
        Observation      % Observation y
    end

    properties (Hidden)
        Gamma            % Stepsize parameter(s)
    end
    
    properties (Nontunable)
        Lambda = 0       % Regulalization Parameter
        %
        MeasureProcess   % Measurment process P
        %
        DataType = 'Image'
        SplitFactor = []
        PadSize     = []
    end
    
    properties (Hidden, Transient)
        DataTypeSet = ...
            matlab.system.StringSet({'Image' 'Volumetric Data'});
    end    

    properties (GetAccess = public, SetAccess = protected)
        Result
        LambdaOriginal
    end
    
    properties(Nontunable, Access = protected)
        AdjointProcess
        ParallelProcess
    end    

    properties(Nontunable, Logical)
        IsIntegrityTest = true
        IsLambdaCompensation = false
        UseParallel = false
        UseGpu = false
    end

    properties(Nontunable,Logical, Hidden)
        Debug = false
    end

    properties(DiscreteState)
        Iteration
    end
    
    methods
        function obj = AbstIterativeMethodSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            if isempty(obj.MeasureProcess)
                import saivdr.degradation.linearprocess.IdenticalMappingSystem
                obj.MeasureProcess = IdenticalMappingSystem();
            end
            if isempty(obj.PadSize)
                if strcmp(obj.DataType,'Volumetric Data')
                    obj.PadSize = zeros(1,3);
                else
                    obj.PadSize = zeros(1,2);
                end
            end
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            %s.Var = obj.Var;
            %s.Obj = matlab.System.saveObject(obj.Obj);
            s.AdjointProcess = matlab.System.saveObject(obj.AdjointProcess);
            s.ParallelProcess = matlab.System.saveObject(obj.ParallelProcess);            
            if isLocked(obj)
                s.Iteration = obj.Iteration;
            end
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            if wasLocked
                obj.Iteration = s.Iteration;
            end
            obj.AdjointProcess = matlab.System.loadObject(s.AdjointProcess);            
            obj.ParallelProcess = matlab.System.loadObject(s.ParallelProcess);              
            %obj.Obj = matlab.System.loadObject(s.Obj);
            %obj.Var = s.Var;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        %{
        function validateInputsImpl(~,x)
            if ~isnumeric(x)
                error('Input must be numeric');
            end
        end
        %}
        
        function validatePropertiesImpl(obj)
            if isempty(obj.Observation)
                me = MException('SaivDr:InstantiationException',...
                    'Observation must be given.');
                throw(me)
            end
        end
        
        function processTunedPropertiesImpl(obj)
            propChange = isChangedProperty(obj,'Dictionary');
            if propChange && isLocked(obj)
                msrProc = obj.MeasureProcess;
                fwdDic = obj.Dictionary{obj.FORWARD};
                framebound = fwdDic.FrameBound;
                obj.Gamma = 1/(framebound*msrProc.LambdaMax);
                gamma  = obj.Gamma;
                lambda = obj.Lambda;
                obj.GaussianDenoiser.Sigma = sqrt(gamma*lambda);
                if isa(obj.ParallelProcess,'saivdr.restoration.AbstOlsOlaProcess')
                    import saivdr.restoration.*
                    gdn = obj.GaussianDenoiser;
                    cm = CoefsManipulator(...
                        'Manipulation',...
                        @(t,cpre) gdn.step(cpre-gamma*t));
                    obj.ParallelProcess.CoefsManipulator = cm;                    
                end
            end
        end
        
        function setupImpl(obj)
            if obj.IsLambdaCompensation
                vObs = obj.Observation;
                adjProc = obj.MeasureProcess.clone();
                adjProc.ProcessingMode = 'Adjoint';     
                adjDic = obj.Dictionary{obj.ADJOINT};
                obj.LambdaOriginal = obj.Lambda; 
                sizeM = numel(vObs); % Data size of measurement 
                src   = adjProc.step(vObs);
                coefs = adjDic.step(src); % Data size of coefficients
                sizeL = numel(coefs);
                obj.Lambda = obj.LambdaOriginal*(sizeM^2/sizeL);
            end
        end
        
        function stepImpl(obj)
            obj.Iteration = obj.Iteration + 1;
        end

        function resetImpl(obj)
            obj.Iteration = 0;
        end
        
    end
    
    methods (Static)

        function z = rmse(x,y)
            z = norm(x(:)-y(:),2)/sqrt(numel(x));
        end
        
    end
    
end

