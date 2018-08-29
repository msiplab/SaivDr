classdef CoefsManipulator < matlab.System
    %COEFSMANIPULATOR Coefficient manipulator for OLS/OLA wrapper classes
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    
    properties (Nontunable)
        Manipulation
        InitialState
    end
    
    properties (Access = private)
        state
    end
    
    properties (Logical,Nontunable)
        IsFeedBack = false
        IsStateOutput = false
    end
    
    methods
        
        % Constractor
        function obj = CoefsManipulator(varargin)
            setProperties(obj,nargin,varargin{:})
        end
        
        function s = getState(obj)
            s = obj.state;
        end
        
        function setState(obj,s)
            obj.state = s;
        end        
        
    end
    
    methods(Access = protected)
        
        function resetImpl(obj)
            obj.state = obj.InitialState;
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            if isLocked(obj)
                s.state = obj.state;
                %class(obj.state)
            end
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            if wasLocked
                obj.state = s.state;
                %class(obj.state)                
            end
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function coefspst = stepImpl(obj,coefspre)
            isfeedback_ = obj.IsFeedBack;
            manipulation_ = obj.Manipulation;
            prestate_ = obj.state;
            
            if isempty(manipulation_)
                coefspst = coefspre;
                state_ = [];
            elseif iscell(coefspre)
                nChs = length(coefspre);
                coefspst = cell(1,nChs);
                if isfeedback_
                    if obj.IsStateOutput && iscell(prestate_)
                        state_ = cell(1,nChs);
                        for iCh = 1:nChs
                            [coefspst{iCh},state_{iCh}] = ...
                                manipulation_(coefspre{iCh},...
                                prestate_{iCh});
                        end
                    elseif obj.IsStateOutput && isscalar(prestate_)
                        state_ = cell(1,nChs);
                        for iCh = 1:nChs
                            [coefspst{iCh},state_{iCh}] = ...
                                manipulation_(coefspre{iCh},...
                                prestate_);
                        end
                    elseif ~obj.IsStateOutput && iscell(prestate_)
                        for iCh = 1:nChs
                            coefspst{iCh} = manipulation_(...
                                coefspre{iCh},prestate_{iCh});
                        end
                        state_ = coefspst;
                    elseif ~obj.IsStateOutput && isscalar(prestate_)
                        for iCh = 1:nChs
                            coefspst{iCh} = manipulation_(...
                                coefspre{iCh},prestate_);
                        end
                        state_ = coefspst;
                    else
                        id = 'SaivDr:IllegalStateInitialization';
                        message = 'State must be cell or scalar.';
                        throw(MException(id,message))
                    end
                else
                    for iCh = 1:nChs
                        coefspst{iCh} = manipulation_(coefspre{iCh});
                        state_ = [];
                    end
                end
            else
                if isfeedback_
                    if obj.IsStateOutput
                        [coefspst,state_] = ...
                            manipulation_(coefspre,prestate_);
                    else
                        coefspst = manipulation_(coefspre,prestate_);
                        state_ = coefspst;
                    end
                else
                    coefspst = manipulation_(coefspre);
                    state_ = [];
                end
            end
            obj.state = state_;
        end
    end
    
end