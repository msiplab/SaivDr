classdef CoefsManipulator < matlab.System
    %COEFSMANIPULATOR Coefficient manipulator for OLS/OLA wrapper classes
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2018-2020, Shogo MURAMATSU
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
        Manipulation
    end
    
    properties
        State
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
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
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
                    if obj.IsStateOutput && iscell(obj.State)
                        state = cell(1,nChs);
                        for iCh = 1:nChs
                            [coefspst{iCh},state{iCh}] = ...
                                manipulation_(coefspre{iCh},...
                                obj.State{iCh});
                        end
                    elseif obj.IsStateOutput && isscalar(obj.State)
                        state = cell(1,nChs);
                        for iCh = 1:nChs
                            [coefspst{iCh},state{iCh}] = ...
                                manipulation_(coefspre{iCh},...
                                obj.State);
                        end                        
                    elseif ~obj.IsStateOutput && iscell(obj.State)
                        for iCh = 1:nChs
                            coefspst{iCh} = manipulation_(...
                                coefspre{iCh},obj.State{iCh});
                        end
                        state = coefspst;                        
                    elseif ~obj.IsStateOutput && isscalar(obj.State) 
                        for iCh = 1:nChs
                            coefspst{iCh} = manipulation_(...
                                coefspre{iCh},obj.State);
                        end
                        state = coefspst;
                    else
                        id = 'SaivDr:IllegalStateInitialization';
                        message = 'State must be cell or scalar.';
                        throw(MException(id,message))                        
                    end
                    obj.State = state;
                else
                    for iCh = 1:nChs
                        coefspst{iCh} = manipulation_(coefspre{iCh});
                    end
                end
            else
                if isfeedback_
                    if obj.IsStateOutput
                        [coefspst,state] = ...
                            manipulation_(coefspre,obj.State);
                    else
                        coefspst = manipulation_(coefspre,obj.State);
                        state = coefspst;
                    end
                    obj.State = state;
                else
                    coefspst = manipulation_(coefspre);
                end
            end
        end
    end

end

