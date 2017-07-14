classdef AprxErrorWithSparseRep < matlab.System %#~codegen
    %APPROXIMATIONERRORWITHSPARSEREP Approximation error with sparse
    %representation
    %
    % SVN identifier:
    % $Id: AprxErrorWithSparseRep.m 866 2015-11-24 04:29:42Z sho $
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
    properties  (Nontunable)
        SourceImages
        NumberOfTreeLevels = 1
        IsFixedCoefs = true
        GradObj      = 'off'
        BoundaryOperation = 'Termination'
        Stochastic   = 'off'
        Mse          = 'on'
    end
    
    properties (Hidden, Transient)
        GradObjSet = ...
            matlab.system.StringSet({'on','off'});
        StochasticSet = ...
            matlab.system.StringSet({'on','off'});        
        MseSet = ...
            matlab.system.StringSet({'on','off'});                
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Termination','Circular'});        
    end    
 
    properties (Access=protected, Nontunable)
        sharedLpPuFb
        nImgs        
    end
    
    properties (Access=protected)
        setOfSubbandSizes
        analyzer
        synthesizer
        evaluator
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
            s.evaluator = matlab.System.saveObject(obj.evaluator);
            
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
            obj.evaluator = matlab.System.loadObject(s.evaluator);
            
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
    
        function validatePropertiesImpl(obj)
            if ~obj.IsFixedCoefs && strcmp(obj.GradObj,'on')
                error('GradObj should be off unless IsFixedCoefs is true.');
            end
        end
        
        function setupImpl(obj,lppufb_,~,~,~)
            if ~obj.IsFixedCoefs
                setupSimCo_(obj,lppufb_)
            elseif strcmp(obj.GradObj,'on')
                setupGradObjOn_(obj,lppufb_)
            else
                setupNormal_(obj,lppufb_)
            end
        end
        
        function [cost,grad] = stepImpl(obj,lppufb_,sprsCoefs,setOfScales,...
                iImg)
            if strcmp(obj.Stochastic,'on') && ~isempty(iImg)
                imgSet = iImg;
            else
                imgSet = 1:obj.nImgs;
            end
            if nargout < 2
                isGrad = false;
            else
                isGrad = true;
            end
            
            %
            cost = 0;
            grad = 0;
            for iImg = imgSet
                srcImg_ = obj.SourceImages{iImg};
                coefs_  = sprsCoefs{iImg};
                scales_ = setOfScales{iImg};
                if ~obj.IsFixedCoefs
                    cost_ = stepSimCo_(obj,lppufb_,srcImg_,coefs_,scales_);
                    grad_ = [];
                elseif strcmp(obj.GradObj,'on') && isGrad
                    [cost_,grad_] = ...
                        stepGradObjOn_(obj,lppufb_,srcImg_,coefs_,scales_);
                else
                    cost_ = stepNormal_(obj,lppufb_,srcImg_,coefs_,scales_);
                    grad_ = [];
                end
                cost = cost + cost_;
                grad = grad + grad_;
            end
            if length(imgSet) > 1
                cost = cost/obj.nImgs;
                grad = grad/obj.nImgs;
            end            
        end
        
        function N = getNumInputsImpl(obj)
            % Specify number of System inputs
            if strcmp(obj.Stochastic,'on')
                N = 4;
            else
                N = 3;
            end
        end
        
        function N = getNumOutputsImpl(~)
            % Specify number of System outputs
            N = 2;
        end
        
    end
    
    methods (Access=private)

        %%
        function setupNormal_(obj,lppufb_,~,~)
            import saivdr.dictionary.cnsoltx.CnsoltFactory
            obj.sharedLpPuFb = clone(lppufb_);
            if isa(lppufb_,'saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dSystem')
                obj.analyzer = [];
                obj.synthesizer = CnsoltFactory.createSynthesis2dSystem(...
                    obj.sharedLpPuFb,...
                    'BoundaryOperation',obj.BoundaryOperation,...
                    'IsCloneLpPuFb2d',false);
            elseif isa(lppufb_,'saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dSystem')
                obj.analyzer = [];                
                obj.synthesizer = CnsoltFactory.createSynthesis3dSystem(...
                    obj.sharedLpPuFb,...
                    'BoundaryOperation',obj.BoundaryOperation,...
                    'IsCloneLpPuFb3d',false);
            else 
               error('Invalid OvsdLpPuFb was supplied.'); 
            end
        end
            
        function cost = stepNormal_(obj,lppufb_,srcImg_,coefs_,scales_)
            synthesizer_ = obj.synthesizer;
            %
            angs_  = get(lppufb_,'Angles');
            mus_   = get(lppufb_,'Mus');
            set(obj.sharedLpPuFb,'Angles',angs_);
            set(obj.sharedLpPuFb,'Mus',mus_);
            %
            aprxImg_ = step(synthesizer_,coefs_,scales_);
            %
            cost  = sum(abs(srcImg_(:)-aprxImg_(:)).^2);
            %
            if strcmp(obj.Mse,'on')
                cost = cost/numel(srcImg_);
            end
        end
        
        %%
        function setupSimCo_(obj,lppufb_,~,~)
            import saivdr.dictionary.cnsoltx.CnsoltFactory
            obj.sharedLpPuFb = clone(lppufb_);
            if isa(lppufb_,'saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dSystem')
                obj.analyzer = CnsoltFactory.createAnalysis2dSystem(...
                    obj.sharedLpPuFb,...
                    'BoundaryOperation',obj.BoundaryOperation,...
                    'IsCloneLpPuFb2d',false);
                obj.synthesizer = CnsoltFactory.createSynthesis2dSystem(...
                    obj.sharedLpPuFb,...
                    'BoundaryOperation',obj.BoundaryOperation,...
                    'IsCloneLpPuFb2d',false);
            elseif isa(lppufb_,'saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dSystem')
                obj.analyzer = CnsoltFactory.createAnalysis3dSystem(...
                    obj.sharedLpPuFb,...
                    'BoundaryOperation',obj.BoundaryOperation,...
                    'IsCloneLpPuFb3d',false);
                obj.synthesizer = CnsoltFactory.createSynthesis3dSystem(...
                    obj.sharedLpPuFb,...
                    'BoundaryOperation',obj.BoundaryOperation,...
                    'IsCloneLpPuFb3d',false);
            else
                error('Invalid OvsdLpPuFb was supplied.');
            end
        end
        
        function cost = stepSimCo_(obj,lppufb_,srcImg_,coefs_,scales_)
            nLevels_     = obj.NumberOfTreeLevels;
            analyzer_    = obj.analyzer;
            synthesizer_ = obj.synthesizer;
            %
            angs_  = get(lppufb_,'Angles');
            mus_   = get(lppufb_,'Mus');
            set(obj.sharedLpPuFb,'Angles',angs_);
            set(obj.sharedLpPuFb,'Mus',mus_);
            %
            masks_ = (coefs_ ~= 0);
            coefs_ = step(analyzer_,srcImg_,nLevels_);
            coefs_ = masks_.*coefs_;
            aprxImg_ = step(synthesizer_,coefs_,scales_);
            %
            cost = sum(abs(srcImg_(:)-aprxImg_(:)).^2);
            %
            if strcmp(obj.Mse,'on')
                cost = cost/numel(srcImg_);
            end            
        end        
        
        %%   
        function setupGradObjOn_(obj,lppufb_,~,~)
            obj.sharedLpPuFb = clone(lppufb_);
            if isa(lppufb_,'saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dTypeISystem')
                import saivdr.dictionary.cnsoltx.design.CplxOvsdLpPuFb2dTypeICostEvaluator
                obj.evaluator = CplxOvsdLpPuFb2dTypeICostEvaluator(...
                        'LpPuFb',obj.sharedLpPuFb,...
                        'BoundaryOperation',obj.BoundaryOperation);
            elseif isa(lppufb_,'saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dTypeISystem')
                import saivdr.dictionary.cnsoltx.design.CplxOvsdLpPuFb3dTypeICostEvaluator                
                obj.evaluator = CplxOvsdLpPuFb3dTypeICostEvaluator(...
                        'LpPuFb',obj.sharedLpPuFb,...
                        'BoundaryOperation',obj.BoundaryOperation);
            else 
               error('Invalid OvsdLpPuFb was supplied.'); 
            end
        end
        
        function [cost,grad] = stepGradObjOn_(obj,lppufb_,srcImg_,coefs_,scales_)
            evaluator_ = obj.evaluator;
            %
            angs_  = get(lppufb_,'Angles');
            mus_   = get(lppufb_,'Mus');
            set(obj.sharedLpPuFb,'Angles',angs_);
            set(obj.sharedLpPuFb,'Mus',mus_);
            %
            [cost,grad] = step(evaluator_,srcImg_,coefs_,scales_);
            %
            if strcmp(obj.Mse,'on')
                cost = cost/numel(srcImg_);
                grad = grad/numel(srcImg_);
            end                        
        end        
    end
    
end
