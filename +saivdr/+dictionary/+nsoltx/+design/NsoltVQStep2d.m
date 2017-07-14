classdef NsoltVQStep2d <  ...
        saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d %#codegen
    %NSOLTVQSTEP2D 2-D Atom Extender for NSOLT
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015-2017, Shogo MURAMATSU
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
    
    properties(Nontunable)
        PartialDifference = 'off'
    end
    
    properties (Hidden, Transient)
        PartialDifferenceSet = ...
            matlab.system.StringSet({'on','off'});
    end   
    
    methods
        
        % Constructor
        function obj = NsoltVQStep2d(varargin)
            obj = obj@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d(varargin{:});
            setProperties(obj,nargin,varargin{:});
            %
            if strcmp(obj.NsoltType,'Type II')
                error('SaivDr: Type-II is not supported yet.');
            end
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d(obj,s,wasLocked);
        end
        
        function value = getNumInputsImpl(~)
            value = 4;
        end

        function validateInputsImpl(obj, coefs, subScale, pmCoefs, ~)
            validateInputsImpl@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d(obj, coefs, subScale, pmCoefs);
            if size(coefs,1) ~= obj.NumberOfSymmetricChannels ...
                    + obj.NumberOfAntisymmetricChannels
                error('SaivDr: Coefficient array has invalid size.');
            end
        end
        
        function setupImpl(obj, arrayCoefs, subScale, pmCoefs, ~)
            setupImpl@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d(obj,arrayCoefs,subScale,pmCoefs);
        end
        
        function arrayCoefs = stepImpl(obj, arrayCoefs, subScale, pmCoefs, idxPm)
            stepImpl@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d(obj,arrayCoefs,subScale,pmCoefs);
            arrayCoefs = stepVQ_(obj,arrayCoefs,idxPm);
        end
        
    end
    
    methods ( Access = private )
        
        function arrayCoefs = stepVQ_(obj,arrayCoefs,idxPm)
            
            if ~isempty(obj.paramMtxCoefs)
                if idxPm < 3
                    arrayCoefs = initialStep_(obj,arrayCoefs,idxPm);
                elseif mod(idxPm-1,2) == 0
                    arrayCoefs = supportExtPhase1_(obj,arrayCoefs,idxPm);
                else
                    arrayCoefs = supportExtPhase2_(obj,arrayCoefs,idxPm);
                end
            end
            
        end
        
        function arrayCoefs = initialStep_(obj,arrayCoefs,idxPm)
            
            hLenU  = obj.NumberOfSymmetricChannels;
            isPdOn = strcmp(obj.PartialDifference,'on');
            %
            if idxPm == 1
                W0 = getParamMtx_(obj,uint32(1));
                arrayCoefs(1:hLenU,:) = W0*arrayCoefs(1:hLenU,:);
                if isPdOn
                    arrayCoefs(hLenU+1:end,:) ...
                        = zeros(size(arrayCoefs(hLenU+1:end,:)));
                end
            else
                U0 = getParamMtx_(obj,uint32(2));
                arrayCoefs(hLenU+1:end,:) = U0*arrayCoefs(hLenU+1:end,:);
                if isPdOn
                    arrayCoefs(1:hLenU,:) ...
                        = zeros(size(arrayCoefs(1:hLenU,:)));
                end
            end

        end
        
        function arrayCoefs = supportExtPhase1_(obj,arrayCoefs,idxPm)
            hLen = obj.NumberOfSymmetricChannels;
            nCols_ = obj.nCols;
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            isPdOn = strcmp(obj.PartialDifference,'on');            
            %
            Ux1 = getParamMtx_(obj,uint32(idxPm));
            I = eye(size(Ux1));
            Z = zeros(size(Ux1));
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            if isPdOn
                arrayCoefs(1:hLen,:) = zeros(size(arrayCoefs(1:hLen,:)));
            end
            % Lower channel rotation            
            if isPeriodicExt
                arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else
                for iCol = 1:nCols_
                    if iCol == 1 %&& ~isPeriodicExt
                        if isPdOn
                            U = Z;
                        else
                            U = -I;
                        end
                    else
                        U = Ux1;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,iCol,U);
                end
            end
        end
        
        function arrayCoefs = supportExtPhase2_(obj,arrayCoefs,idxPm)
            hLen = obj.NumberOfSymmetricChannels;
            isPdOn = strcmp(obj.PartialDifference,'on');                        
            %
            Ux2 = getParamMtx_(obj,uint32(idxPm));
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPdOn
                arrayCoefs(1:hLen,:) = zeros(size(arrayCoefs(1:hLen,:)));
            end
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
        end
        
        function arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs)
            hLenMn = max([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            nRows_ = obj.nRows;
            %
            lowerCoefsPre = arrayCoefs(hLenMn+1:end,end-nRows_+1:end);
            arrayCoefs(hLenMn+1:end,nRows_+1:end) = ...
                arrayCoefs(hLenMn+1:end,1:end-nRows_);
            arrayCoefs(hLenMn+1:end,1:nRows_) = ...
                lowerCoefsPre;
        end
        
        function arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs)
            hLenMx = min([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            nRows_ = obj.nRows;
            %
            upperCoefsPost = arrayCoefs(1:hLenMx,1:nRows_);
            arrayCoefs(1:hLenMx,1:end-nRows_) = ...
                arrayCoefs(1:hLenMx,nRows_+1:end);
            arrayCoefs(1:hLenMx,end-nRows_+1:end) = ...
                upperCoefsPost;
        end
    end
    
end
