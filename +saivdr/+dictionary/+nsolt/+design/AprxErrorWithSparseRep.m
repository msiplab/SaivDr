classdef AprxErrorWithSparseRep < matlab.System %#~codegen
    %APPROXIMATIONERRORWITHSPARSEREP Approximation error with sparse
    %representation
    %
    % SVN identifier:
    % $Id: AprxErrorWithSparseRep.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
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
    % http://msiplab.eng.niigata-u.ac.jp/    
    %
    properties  (Nontunable)
        SourceImages
        NumberOfTreeLevels = 1
        IsFixedCoefs = true
    end
 
    properties (Access=protected, Nontunable)
        sharedLpPuFb2d
        nImgs        
    end
    
    properties (Access=protected)
        setOfSubbandSizes
        analyzer
        synthesizer
    end
        
    methods
        function obj = AprxErrorWithSparseRep(varargin)
            setProperties(obj,nargin,varargin{:});
            obj.nImgs = length(obj.SourceImages);
        end
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@matlab.System(obj);
            
            % Save the child System objects
            s.analyzer = matlab.System.saveObject(obj.analyzer);
            s.synthesizer = matlab.System.saveObject(obj.synthesizer);
            s.sharedLpPuFb2d = matlab.System.saveObject(obj.sharedLpPuFb2d);
            
            % Save the protected & private properties
            s.setOfSubbandSizes = obj.setOfSubbandSizes;
            s.nImgs = obj.nImgs;

            % Save the state only if object locked
            %if isLocked(obj)
            %    s.state = obj.state;
            %end
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load child System objects
            obj.analyzer = matlab.System.loadObject(s.analyzer);
            obj.synthesizer = matlab.System.loadObject(s.synthesizer);
            obj.sharedLpPuFb2d = matlab.System.loadObject(s.sharedLpPuFb2d);
            
            % Load protected and private properties
            obj.setOfSubbandSizes = s.setOfSubbandSizes;
            obj.nImgs = s.nImgs;

            % Load the state only if object locked
            %if wasLocked
            %    obj.state = s.state;
            %end
            
            % Call base class method to load public properties
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
    
        function setupImpl(obj,lppufb_,~,~)
            import saivdr.dictionary.nsolt.NsoltFactory
            obj.sharedLpPuFb2d = lppufb_;
            if ~obj.IsFixedCoefs
                obj.analyzer = NsoltFactory.createAnalysisSystem(...
                    obj.sharedLpPuFb2d,...
                    'BoundaryOperation','Termination',...
                    'IsCloneLpPuFb2d',false);
            end
            obj.synthesizer = NsoltFactory.createSynthesisSystem(...
                obj.sharedLpPuFb2d,...
                'BoundaryOperation','Termination',...
                'IsCloneLpPuFb2d',false);                        
        end
        
        function resetImpl(~)
        end
        
        function cost = stepImpl(obj,lppufb_,sprsCoefs,setOfScales)
            paramMtx = step(lppufb_,[],[]);
            set(obj.sharedLpPuFb2d,'ParameterMatrixSet',paramMtx);
            cost = 0;
            for iImg = 1:obj.nImgs
                coefs_ = sprsCoefs{iImg};
                scales_ = setOfScales{iImg};
                if ~obj.IsFixedCoefs
                    masks_ = (coefs_ ~= 0);
                    coefs_ = ...
                        step(obj.analyzer,obj.SourceImages{iImg},obj.NumberOfTreeLevels);                    
                    coefs_ = masks_.*coefs_;
                end
                aprxImg_ = step(obj.synthesizer,coefs_,scales_);                
                subcost = norm(obj.SourceImages{iImg}(:) - aprxImg_(:))^2;
                cost = cost + subcost;
            end
        end
        
        function N = getNumInputsImpl(~)
            % Specify number of System inputs
            N = 3;
        end
        
        function N = getNumOutputsImpl(~)
            % Specify number of System outputs
            N = 1;
        end
        
    end
       
end
