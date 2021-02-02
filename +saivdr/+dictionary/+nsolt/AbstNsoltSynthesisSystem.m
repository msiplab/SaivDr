classdef AbstNsoltSynthesisSystem < ...
        saivdr.dictionary.AbstSynthesisSystem %#~codegen
    %ABSTNSOLTSYNTHESISSYSTEM Abstract class of NSOLT synthesis system
    %
    % SVN identifier:
    % $Id: AbstNsoltSynthesisSystem.m 683 2015-05-29 08:22:13Z sho $
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
    
    properties (Nontunable)
        LpPuFb2d
        BoundaryOperation = 'Termination'
    end

    properties (Nontunable, PositiveInteger)
        NumberOfSymmetricChannels 
        NumberOfAntisymmetricChannels 
    end
    
    properties (Nontunable, Logical)
        IsCloneLpPuFb2d = true
    end
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Termination','Circular'});
    end

    properties (Access = protected, Nontunable)
        nAllCoefs
        nAllChs
    end
    
    properties (Access = protected, Nontunable, PositiveInteger)
        decX
        decY
    end

    properties (Access = protected)
        paramMtx
        arrayCoefs
        ordX
        ordY
    end
    
    properties (Access = protected, PositiveInteger)
        nRows
        nCols
    end
        
    methods (Access = protected, Abstract = true)
        synthesize_(obj,subCoefs)
        ps = getDefaultNumberOfSymmetricChannels(obj)
        pa = getDefaultNumberOfAntisymmetricChannels(obj)        
    end
    
    methods
        
        % Constructor
        function obj = AbstNsoltSynthesisSystem(varargin)
            setProperties(obj,nargin,varargin{:});
            if isempty(obj.NumberOfSymmetricChannels)
                obj.NumberOfSymmetricChannels = ...
                    getDefaultNumberOfSymmetricChannels(obj);
            end
            if isempty(obj.NumberOfAntisymmetricChannels)
                obj.NumberOfAntisymmetricChannels = ...
                    getDefaultNumberOfAntisymmetricChannels(obj);
            end            
            if isempty(obj.LpPuFb2d)
                import saivdr.dictionary.nsolt.NsoltFactory
                obj.LpPuFb2d = NsoltFactory.createOvsdLpPuFb2dSystem(...
                    'NumberOfChannels',...
                    [obj.NumberOfSymmetricChannels obj.NumberOfAntisymmetricChannels],...
                    'NumberOfVanishingMoments',1,...
                    'OutputMode','ParameterMatrixSet');
            end            
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsolt.ChannelGroup
            dec = get(obj.LpPuFb2d,'DecimationFactor');
            obj.decX = dec(Direction.HORIZONTAL);
            obj.decY = dec(Direction.VERTICAL);
            ord = get(obj.LpPuFb2d,'PolyPhaseOrder');
            obj.ordX = int32(ord(Direction.HORIZONTAL));
            obj.ordY = int32(ord(Direction.VERTICAL));
            obj.FrameBound = 1;
        end
        
    end    
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj);
            
            % Save the child System objects
            s.LpPuFb2d = matlab.System.saveObject(obj.LpPuFb2d);
            
            % Save the protected & private properties
            %s.nRows = obj.nRows;
            %s.nCols = obj.nCols;
            s.arrayCoefs = obj.arrayCoefs;
            s.decX = obj.decX;
            s.decY = obj.decY;
            s.ordX = obj.ordX;
            s.ordY = obj.ordY;
            s.nAllCoefs = obj.nAllCoefs;
            s.nAllChs = obj.nAllChs;
            
            % Save the state only if object locked
            %if isLocked(obj)
            %    s.state = obj.state;
            %end
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load child System objects
            import saivdr.dictionary.nsolt.*
            
            % Load protected and private properties
            %obj.nRows = s.nRows;
            %obj.nCols = s.nCols;
            obj.arrayCoefs = s.arrayCoefs;
            obj.decX = s.decX;
            obj.decY = s.decY;
            obj.ordX = s.ordX;
            obj.ordY = s.ordY;
            obj.nAllCoefs = s.nAllCoefs;
            obj.nAllChs = s.nAllChs;
            
            % Load the state only if object locked
            %if wasLocked
            %    obj.state = s.state;
            %end
            
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj,s,wasLocked);
            %
            obj.LpPuFb2d = matlab.System.loadObject(s.LpPuFb2d);
        end
        
        function validatePropertiesImpl(obj)
            if ~isa(obj.LpPuFb2d,'saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dSystem')
                me = MException('Invalid:InstantiationException',...
                    'LpPuFb2d must be an object of saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dSystem.');
                throw(me)
            end
        end
        
        function setupImpl(obj, ~, scales)
            import saivdr.dictionary.nsolt.ChannelGroup
            nChs_ = get(obj.LpPuFb2d,'NumberOfChannels');
            obj.NumberOfSymmetricChannels = nChs_(ChannelGroup.UPPER);
            obj.NumberOfAntisymmetricChannels = nChs_(ChannelGroup.LOWER);
            obj.nAllChs = size(scales,1);
            obj.nAllCoefs = scales(:,1).*scales(:,1);
            %
            if obj.IsCloneLpPuFb2d
                obj.LpPuFb2d = clone(obj.LpPuFb2d);
            end
            if ~strcmp(get(obj.LpPuFb2d,'OutputMode'),'ParameterMatrixSet')
                warning('OutputMode of LpPuFb2d is recommended to be ParameterMatrixSet.');
                release(obj.LpPuFb2d);
                set(obj.LpPuFb2d,'OutputMode','ParameterMatrixSet');
            end
        end
        
        function resetImpl(~)
        end
        
        function recImg = stepImpl(obj, coefs, scales)
            obj.paramMtx = step(obj.LpPuFb2d,[],[]);
            %
            nChs_ = [ obj.NumberOfSymmetricChannels ...
                obj.NumberOfAntisymmetricChannels ];
            subCoefs = cell(sum(nChs_),1);
            nLevels = (size(scales,1)-1)/(sum(double(nChs_))-1);
            %
            iSubband = 1;
            sIdx = 1;
            eIdx = prod(scales(iSubband,:));
            %
            subCoefs{1} = reshape(coefs(sIdx:eIdx),scales(iSubband,:));
            for iLevel = 1:nLevels
                for iCh = 2:sum(nChs_)
                    iSubband = iSubband + 1;
                    sIdx = eIdx + 1;
                    eIdx = sIdx + prod(scales(iSubband,:))-1;
                    subCoefs{iCh} = reshape(coefs(sIdx:eIdx),scales(iSubband,:));
                end
                synthesize_(obj,subCoefs);
                recImg = obj.arrayCoefs;
                subCoefs{1} = recImg;
            end
            %{
            cNumCoefs = cumsum(prod(scales.').');
            eSubband = 1;
            recImg = coefs(1:sum(nChs_));
            eIdx = numel(recImg);
            for iLevel = 1:nLevels
                sSubband = eSubband + 1;
                eSubband = eSubband + sum(nChs_) - 2;
                sIdx = eIdx + 1;
                eIdx = sIdx + CNumCoefs(eSubband);
                finCoefs = coefs(sIdx:eIdx)
                subCoefs = [ recImg(:); finCoefs(:) ];
                subScales = [ size(recImg) ; scales(sSubband:eSubband,:) ];
                synthresize_(obj,subCoefs,subScales)
                recImg = obj.arrayCoefs;
            end
            %}
            
        end

    end
    
    methods (Access = protected, Static = true)
                
        function value = idct2_(x)
            value = idct2(x.data);
        end
        
        function value = getMatrix_(subCoefs,nchs)
            height = size(subCoefs{1},1)*nchs;
            width  = size(subCoefs{1},2);
            value  = zeros(height,width);
            for iSubband = 1:nchs
                value(iSubband:nchs:end,:) = subCoefs{iSubband};
            end
        end
        
    end
    
end
