classdef AbstOvsdLpPuFbCostEvaluator < matlab.System
    %ABSTOVSDLPPUFBCOSTEVALUATOR Abstract class of cost evaluator for NSOLT
    %
    % SVN identifier:
    % $Id: AbstOvsdLpPuFbCostEvaluator.m 828 2015-10-21 11:12:19Z sho $
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015, Shogo MURAMATSU
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
    properties (Nontunable, PositiveInteger)
        NumberOfSymmetricChannels
        NumberOfAntisymmetricChannels
    end
    
    properties (Nontunable)
        LpPuFb
        BoundaryOperation = 'Termination'
    end
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Termination','Circular'});
    end
    
    properties (Nontunable, Logical)
        IsCloneLpPuFb = true;
    end
    
    properties (Access = protected, Nontunable)
        decimationFactor
        polyPhaseOrder
    end
    
    properties (Access = protected)
        atomCncFcn
        gradFcn
    end
    
    properties (Access = protected, Logical)
        isMexFcn = false
    end
    
    methods (Abstract, Access = protected)
        dim = getDataDimension(obj);
    end
    
    methods
        
        % Constractor
        function obj = AbstOvsdLpPuFbCostEvaluator(varargin)
            setProperties(obj,nargin,varargin{:})
            %
            if isempty(obj.LpPuFb)
                import saivdr.dictionary.nsoltx.NsoltFactory
                if getDataDimension(obj) == 2
                    obj.LpPuFb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                        'NumberOfChannels', ...
                        [ obj.NumberOfSymmetricChannels ...
                        obj.NumberOfAntisymmetricChannels ], ...
                        'NumberOfVanishingMoments',1,...
                        'OutputMode','ParameterMatrixSet');
                elseif getDataDimension(obj) == 3
                    obj.LpPuFb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                        'NumberOfChannels', ...
                        [ obj.NumberOfSymmetricChannels ...
                        obj.NumberOfAntisymmetricChannels ], ...
                        'NumberOfVanishingMoments',1,...
                        'OutputMode','ParameterMatrixSet');
                else
                    error('Not yet supported')
                end
            end
            
            if obj.IsCloneLpPuFb
                obj.LpPuFb = clone(obj.LpPuFb);
            end
            
            if ~strcmp(get(obj.LpPuFb,'OutputMode'),'ParameterMatrixSet')
                release(obj.LpPuFb);
                set(obj.LpPuFb,'OutputMode','ParameterMatrixSet');
            end
            
            obj.decimationFactor = get(obj.LpPuFb,'DecimationFactor');
            obj.polyPhaseOrder   = get(obj.LpPuFb,'PolyPhaseOrder');
            nch = get(obj.LpPuFb,'NumberOfChannels');
            obj.NumberOfSymmetricChannels = nch(1);
            obj.NumberOfAntisymmetricChannels = nch(2);
            %
            %obj.FrameBound = 1;
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@matlab.System(obj);
            % Save the child System objects
            s.LpPuFb = matlab.System.saveObject(obj.LpPuFb);
            
            % Save the protected & private properties
            s.atomCncFcn       = obj.atomCncFcn;
            s.gradFcn          = obj.gradFcn;
            s.decimationFactor = obj.decimationFactor;
            s.polyPhaseOrder   = obj.polyPhaseOrder;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load protected and private properties
            obj.atomCncFcn       = s.atomCncFcn;
            obj.gradFcn          = s.gradFcn;
            obj.decimationFactor = s.decimationFactor;
            obj.polyPhaseOrder   = s.polyPhaseOrder;
            % Call base class method to load public properties
            loadObjectImpl@matlab.System(obj,s,wasLocked);
            % Load the child System objects
            obj.LpPuFb = matlab.System.loadObject(s.LpPuFb);
        end
    end
    
end
