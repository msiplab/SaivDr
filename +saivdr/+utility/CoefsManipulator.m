classdef CoefsManipulator < matlab.System
    %COEFSMANIPULATOR Coefficient manipulator for OLS/OLA wrapper classes
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    
    properties (Nontunable)
        Manipulation
        InitialState
    end
    
    properties (DiscreteState)
        state
    end
    
    properties (Logical)
        IsFeedBack = false
        IsStateOutput = false
    end
    
    methods
        
        % Constractor
        function obj = CoefsManipulator(varargin)
            setProperties(obj,nargin,varargin{:})
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
            end
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            if wasLocked
                obj.state = s.state;
            end
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function coefspst = stepImpl(obj,coefspre)
            isfeedback_ = obj.IsFeedBack;
            manipulation_ = obj.Manipulation;
            
            if isempty(manipulation_)
                coefspst = coefspre;
            elseif iscell(coefspre)
                nChs = length(coefspre);
                coefspst = cell(1,nChs);
                if isfeedback_
                    if obj.IsStateOutput && iscell(obj.state)
                        state_ = cell(1,nChs);
                        for iCh = 1:nChs
                            [coefspst{iCh},state_{iCh}] = ...
                                manipulation_(coefspre{iCh},...
                                obj.state{iCh});
                        end
                    elseif obj.IsStateOutput && isscalar(obj.state)
                        state_ = cell(1,nChs);
                        for iCh = 1:nChs
                            [coefspst{iCh},state_{iCh}] = ...
                                manipulation_(coefspre{iCh},...
                                obj.state);
                        end
                    elseif ~obj.IsStateOutput && iscell(obj.state)
                        for iCh = 1:nChs
                            coefspst{iCh} = manipulation_(...
                                coefspre{iCh},obj.state{iCh});
                        end
                        state_ = coefspst;
                    elseif ~obj.IsStateOutput && isscalar(obj.state)
                        for iCh = 1:nChs
                            coefspst{iCh} = manipulation_(...
                                coefspre{iCh},obj.state);
                        end
                        state_ = coefspst;
                    else
                        id = 'SaivDr:IllegalStateInitialization';
                        message = 'State must be cell or scalar.';
                        throw(MException(id,message))
                    end
                    obj.state = state_;
                else
                    for iCh = 1:nChs
                        coefspst{iCh} = manipulation_(coefspre{iCh});
                    end
                end
            else
                if isfeedback_
                    if obj.IsStateOutput
                        [coefspst,state_] = ...
                            manipulation_(coefspre,obj.state);
                    else
                        coefspst = manipulation_(coefspre,obj.state);
                        state_ = coefspst;
                    end
                    obj.state = state_;
                else
                    coefspst = manipulation_(coefspre);
                end
            end
        end
    end
    
end

