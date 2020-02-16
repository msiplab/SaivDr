classdef NsoltDictionaryLearningPnP < matlab.System
    %NSOLTDICTIONARYLERNINGPNP NSOLT dictionary learning (Plug and Play)
    %
    % Requirements: MATLAB R2015b
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
    properties
        TrainingImages
        SparseApproximater
        DictionaryUpdater
    end
    
    properties (Nontunable)
        DecimationFactor = [2 2]
        NumberOfChannels = [2 2]
        PolyPhaseOrder   = [0 0]
        NumberOfVanishingMoments = 0
        NumberOfLevels = 1
        %
        DataType = 'Image'
    end
    
    properties (Hidden, Transient)
        DataTypeSet = ...
            matlab.system.StringSet({'Image','Volumetric Data'});
    end
    
    properties (Nontunable, Logical)
        IsRandomInit = false
    end
    
    properties (Hidden)
        StdOfAngRandomInit    = 1e-2;
        PrbOfFlipMuRandomInit = 0
    end
    
    properties(DiscreteState)
        Count
    end
    
    
    properties(GetAccess=public,SetAccess=private)
        OvsdLpPuFb
    end
    
    properties (Access = protected, Nontunable)
        nImgs
    end
    
    methods
        function obj = NsoltDictionaryLearningPnP(varargin)
            setProperties(obj,nargin,varargin{:});
            %
            import saivdr.dictionary.nsoltx.NsoltFactory
            args = { ...
                'DecimationFactor', obj.DecimationFactor,...
                'NumberOfChannels', obj.NumberOfChannels,...
                'PolyPhaseOrder', obj.PolyPhaseOrder,...
                'NumberOfVanishingMoments',obj.NumberOfVanishingMoments,...
                'OutputMode','ParameterMatrixSet'};
            if strcmp(obj.DataType,'Image')
                obj.OvsdLpPuFb = ...
                    NsoltFactory.createOvsdLpPuFb2dSystem(args{:});
            else
                obj.OvsdLpPuFb = ...
                    NsoltFactory.createOvsdLpPuFb3dSystem(args{:});
            end
        end
    end
    
    methods (Access=protected)
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@matlab.System(obj);
            % Save the child System objects
            s.OvsdLpPuFb = matlab.System.saveObject(obj.OvsdLpPuFb);
            % Save the protected & private properties
            s.nImgs = obj.nImgs;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load protected and private properties
            obj.nImgs = s.nImgs;
            % Load the child System objects
            obj.OvsdLpPuFb = matlab.System.loadObject(s.OvsdLpPuFb);
            % Call base class method to load public properties
            loadObjectImpl@matlab.System(obj,s,wasLocked);             
        end
        function flag = isInactivePropertyImpl(obj,propertyName)
            flag = false;
            if strcmp(propertyName,'StdOfAngRandomInit') || ...
                    strcmp(propertyName,'PrbOfFlipMuRandomInit')
                flag = ~obj.IsRandomInit;
            end
        end
        
        function validatePropertiesImpl(obj)
            if isempty(obj.SparseApproximater)
                me = MException('SaivDr:InstantiationException',...
                    'SparseApproximater must be given.');
                throw(me)
            end
            if isempty(obj.DictionaryUpdater)
                me = MException('SaivDr:InstantiationException',...
                    'DictionaryUpdater must be given.');
                throw(me)
            end
        end
        
        function setupImpl(obj, varargin)
            
            % Random initialization
            if obj.IsRandomInit
                %
                sdv = obj.StdOfAngRandomInit;
                angles_ = obj.OvsdLpPuFb.Angles;
                sizeAngles = size(angles_);
                angles_ = angles_ + sdv*randn(sizeAngles);
                obj.OvsdLpPuFb.Angles = angles_;
                %
                thr = obj.PrbOfFlipMuRandomInit;
                mus_ = obj.OvsdLpPuFb.Mus;
                sizeMus = size(mus_);
                mus_ = mus_ .* (2*(rand(sizeMus)>=thr)-1);
                obj.OvsdLpPuFb.Mus = mus_;
                %
            end
            
        end
        
        function varargout = stepImpl(obj,varargin)
            obj.Count = obj.Count + 1;
            if nargin > 0
                obj.TrainingImages = varargin{1};
            end
            if nargin > 1
                options = varargin{2};
            end
            obj.nImgs = length(obj.TrainingImages);
            
            % Sparse Approximation
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesisSystem(...
                obj.OvsdLpPuFb,'IsCloneLpPuFb',false);
            analyzer = NsoltFactory.createAnalysisSystem(...
                obj.OvsdLpPuFb,'IsCloneLpPuFb',false);
            analyzer.NumberOfLevels = obj.NumberOfLevels;
            obj.SparseApproximater.Dictionary = { synthesizer, analyzer};
            sprsCoefs   = cell(obj.nImgs,1);
            setOfScales = cell(obj.nImgs,1);
            for iImg = 1:obj.nImgs
                if ~isempty(obj.SparseApproximater.StepMonitor)
                    obj.SparseApproximater.StepMonitor.reset();
                    obj.SparseApproximater.StepMonitor.SourceImage...
                        = obj.TrainingImages{iImg};
                end
                [~, sprsCoefs{iImg}, setOfScales{iImg}] = ...
                    obj.SparseApproximater.step(obj.TrainingImages{iImg});
            end

            % Dictionary Update
            obj.DictionaryUpdater.TrainingImages = obj.TrainingImages;
            obj.DictionaryUpdater.SparseCoefficients = sprsCoefs;
            obj.DictionaryUpdater.SetOfScales = setOfScales;
            [ lppufb, fval, exitflag ] = ...
                obj.DictionaryUpdater.step(obj.OvsdLpPuFb,options);
            
            % Update
            obj.OvsdLpPuFb.Angles = lppufb.Angles;
            obj.OvsdLpPuFb.Mus    = lppufb.Mus;
            
            % Output
            if nargout > 0
                varargout{1} = obj.OvsdLpPuFb;
            end
            if nargout > 1
                varargout{2} = fval;
            end
            if nargout > 2
                varargout{3} = exitflag;
            end
            
        end
        
        function resetImpl(obj)
            obj.Count = 0;
        end
        
    end
    
end