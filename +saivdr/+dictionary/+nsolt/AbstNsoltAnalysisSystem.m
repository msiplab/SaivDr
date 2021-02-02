classdef AbstNsoltAnalysisSystem < ...
        saivdr.dictionary.AbstAnalysisSystem %#~codegen
    %ABSTNSOLTANALYSISSYSTEM Abstract class of NSOLT analysis system
    %
    % SVN identifier:
    % $Id: AbstNsoltAnalysisSystem.m 683 2015-05-29 08:22:13Z sho $
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
        analyze_(obj,srcImg)
        ps = getDefaultNumberOfSymmetricChannels(obj)
        pa = getDefaultNumberOfAntisymmetricChannels(obj)
    end
    
    methods
        
        % Constructor
        function obj = AbstNsoltAnalysisSystem(varargin)
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
            obj.ordX = ord(Direction.HORIZONTAL);
            obj.ordY = ord(Direction.VERTICAL);
        end
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@matlab.System(obj);
            
            % Save the child System objects            
            %s.paramMtx = matlab.System.saveObject(obj.paramMtx);
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
            %obj.paramMtx = matlab.System.loadObject(s.paramMtx);
           
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
            loadObjectImpl@matlab.System(obj,s,wasLocked);
            %
            obj.LpPuFb2d = matlab.System.loadObject(s.LpPuFb2d);            
        end
        
        function setupImpl(obj, srcImg, nLevels)
            import saivdr.dictionary.nsolt.ChannelGroup            
            nChs_ = get(obj.LpPuFb2d,'NumberOfChannels');
            obj.NumberOfSymmetricChannels = nChs_(ChannelGroup.UPPER);
            obj.NumberOfAntisymmetricChannels = nChs_(ChannelGroup.LOWER);
            obj.nAllChs = nLevels*(sum(nChs_)-1)+1;
            nDecs = double(obj.decX*obj.decY);
            if nDecs == 1
                obj.nAllCoefs = numel(srcImg)*(...
                    (sum(nChs_)-1)*(double(nLevels)/nDecs) ...
                    + 1/double(obj.decX*obj.decY)^double(nLevels));
            else
                obj.nAllCoefs = numel(srcImg)*(...
                    (sum(nChs_)-1)*(nDecs^double(nLevels)-1)/(nDecs^double(nLevels)*(nDecs-1))  ...
                    + 1/double(obj.decX*obj.decY)^double(nLevels));
            end
            %
            if obj.IsCloneLpPuFb2d
                obj.LpPuFb2d = clone(obj.LpPuFb2d);
            end
            if ~strcmp(get(obj.LpPuFb2d,'OutputMode'),'ParameterMatrixSet')
                warning('OutputMode of OvsdLpPuFb2d is recommended to be ParameterMatrixSet.');
                release(obj.LpPuFb2d);
                set(obj.LpPuFb2d,'OutputMode','ParameterMatrixSet');
            end
        end
        
        function resetImpl(~)
        end     

        function [ coefs, scales ] = stepImpl(obj, srcImg, nLevels)
            obj.paramMtx = step(obj.LpPuFb2d,[],[]);
            %
            nChs_ = ...
                [ obj.NumberOfSymmetricChannels ...
                obj.NumberOfAntisymmetricChannels ];
            iSubband = obj.nAllChs;
            eIdx     = obj.nAllCoefs;
            subbandCoefsL = srcImg;
            scales = zeros(obj.nAllChs,2);
            coefs  = zeros(1,obj.nAllCoefs);
            for iLevel = 1:nLevels
                analyze_(obj,subbandCoefsL);
                for iCh = sum(nChs_):-1:2
                    subbandCoefs = obj.arrayCoefs(iCh,:);                
                    scales(iSubband,:) = [ obj.nRows obj.nCols ];
                    sIdx = eIdx - (obj.nRows*obj.nCols) + 1;
                    coefs(sIdx:eIdx) = subbandCoefs(:).';
                    iSubband = iSubband-1;
                    eIdx = sIdx - 1;
                end
                subbandCoefsL = reshape(obj.arrayCoefs(1,:),obj.nRows,obj.nCols);
            end
            scales(1,:) = [ obj.nRows obj.nCols ];
            coefs(1:eIdx) = subbandCoefsL(:).';
        end

    end
    
    methods (Access = protected, Static = true)
        
        function value = dct2_(x)
            value = dct2(x.data);
        end
        
        function value = permuteDctCoefs_(x)
            coefs = x.data;
            cee = coefs(1:2:end,1:2:end);
            coo = coefs(2:2:end,2:2:end);
            coe = coefs(2:2:end,1:2:end);
            ceo = coefs(1:2:end,2:2:end);
            value = [ cee(:) ; coo(:) ; coe(:) ; ceo(:) ];
            value = reshape(value,size(coefs));
        end
    end
    
end

