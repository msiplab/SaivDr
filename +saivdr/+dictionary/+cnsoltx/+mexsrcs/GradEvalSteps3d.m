classdef GradEvalSteps3d < matlab.System %#codegen
    %GRADEVALSTEPS3D Gradient Evaluation Steps for 3-D NSOLT
    %
    % SVN identifier:
    % $Id: GradEvalSteps3d.m 868 2015-11-25 02:33:11Z sho $
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: https://www.linkedin.com/in/shogo-muramatsu-627b084b
    %
    
    % Public, tunable properties.
    properties (Nontunable, PositiveInteger)
        NumberOfSymmetricChannels      = 4
        NumberOfAntisymmetricChannels  = 4
    end
    
    properties (Logical)
        IsPeriodicExt = false
    end
    
    properties
        PolyPhaseOrder = [ 0 0 0 ]
    end
    
    properties (Access = private, Nontunable)
        vqStep
        vqStepPd
    end
    
    properties (Access = private)
        omgpd
        paramMtxCoefs
        indexOfParamMtxSzTab
        paramMtxSzTab         
    end
    
    methods
        
        function obj = GradEvalSteps3d(varargin)
            setProperties(obj,nargin,varargin{:});
            %
            ps  = obj.NumberOfSymmetricChannels;
            pa  = obj.NumberOfAntisymmetricChannels;
            
            obj.vqStep = saivdr.dictionary.nsoltx.design.NsoltVQStep3d(...
                'PartialDifference','off',...
                'NumberOfSymmetricChannels',ps,...
                'NumberOfAntisymmetricChannels',pa);
            
            obj.vqStepPd = saivdr.dictionary.nsoltx.design.NsoltVQStep3d(...
                'PartialDifference','on',...
                'NumberOfSymmetricChannels',ps,...
                'NumberOfAntisymmetricChannels',pa);

            obj.omgpd = saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'NumberOfDimensions',ps);            
            
        end
    end
    
    methods (Access = protected)

        function processTunedPropertiesImpl(obj)
            if isChangedProperty(obj,'IsPeriodicExt');
                fpe = obj.IsPeriodicExt;
                set(obj.vqStep,'IsPeriodicExt',fpe);
                set(obj.vqStepPd,'IsPeriodicExt',fpe);
            end
            if isChangedProperty(obj,'PolyPhaseOrder');
                ord = obj.PolyPhaseOrder;
                set(obj.vqStep,'PolyPhaseOrder',ord);
                set(obj.vqStepPd,'PolyPhaseOrder',ord);
                setupParamMtx_(obj);            
            end
        end

        function setupImpl(obj,...
                ~, ~, ~, pmCoefs, ~, ~, ~ )
            ord = obj.PolyPhaseOrder;
            fpe = obj.IsPeriodicExt;
            ps  = obj.NumberOfSymmetricChannels;

            set(obj.vqStep,'IsPeriodicExt',fpe);
            set(obj.vqStep,'PolyPhaseOrder',ord);
            set(obj.vqStepPd,'IsPeriodicExt',fpe);
            set(obj.vqStepPd,'PolyPhaseOrder',ord);

            nAngsPm = ps*(ps-1)/2;
            nMusPm  = ps;
            step(obj.omgpd,zeros(nAngsPm,1),ones(nMusPm,1),uint32(1));
            %
            setupParamMtx_(obj);
            obj.paramMtxCoefs = pmCoefs;
        end
        
        function grad = stepImpl(obj, ...
                arrayCoefsB, arrayCoefsC, scale, pmCoefs, ...
                angs, mus, isnodc )
            %
            ord       = obj.PolyPhaseOrder;
            ps        = obj.NumberOfSymmetricChannels;
            omgpd_    = obj.omgpd;
            nAngs     = numel(angs);
            nAngsPm   = ps*(ps-1)/2;
            nMusPm    = ps;
            vqStep_   = obj.vqStep;
            vqStepPd_ = obj.vqStepPd;
            
            %
            arrayCoefsB = step(vqStep_,arrayCoefsB,scale,pmCoefs,...
                uint32(1));
            arrayCoefsB = step(vqStep_,arrayCoefsB,scale,pmCoefs,...
                uint32(2));
            
            grad  = zeros(nAngs,1);
            permFlagDep = false;
            permFlagHor = false;
            for iAng = 1:nAngs
                state  = getState_(obj,iAng);
                iPdAng = state.iAngPerMtx;
                sIdAng = state.startIdxAng;
                eIdAng = sIdAng + nAngsPm - 1;
                sIdMu  = state.startIdxMu;
                eIdMu  = sIdMu  + nMusPm  - 1;
                angs_  = angs(sIdAng:eIdAng);
                mus_   = mus(sIdMu:eIdMu);
                idxMtx = state.curIdxMtx;
                
                % Steps 3-5
                if state.curOrd == 0 && state.curDir == 0
                    if ~state.isMtxU
                        if isnodc && iAng < ps
                            dW0 = zeros(ps);
                        else
                            dW0 = step(omgpd_,angs_,mus_,iPdAng); % TODO
                        end
                        setParamMtx_(obj,dW0,idxMtx); % TODO
                    else
                        dU0 = step(omgpd_,angs_,mus_,iPdAng); % TODO
                        setParamMtx_(obj,dU0,idxMtx); % TODO
                    end
                    pdCoefs = obj.paramMtxCoefs;
                    arrayCoefsA = step(vqStepPd_,arrayCoefsC,scale,...
                        pdCoefs,idxMtx);
                    %
                    grad(iAng) = -2*(arrayCoefsA(:).'*arrayCoefsB(:));
                    %
                    if state.isMtxU && iPdAng == nAngsPm
                        arrayCoefsC = step(vqStep_,arrayCoefsC,scale,pmCoefs,...
                            uint32(1));
                        arrayCoefsC = step(vqStep_,arrayCoefsC,scale,pmCoefs,...
                            uint32(2));
                    end
                end
                
                % Steps 6-7
                % Depth -> permute
                % -> Horizontal -> permute
                % -> Vertical   -> permute
                if state.curDir > 0
                    %
                    if state.curDir < saivdr.dictionary.utility.Direction.DEPTH && ~permFlagDep
                        [arrayCoefsB,~] = ...
                            saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps3d.permuteCoefs_(arrayCoefsB,scale);
                        [arrayCoefsC,scale] = ...
                            saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps3d.permuteCoefs_(arrayCoefsC,scale);
                        permFlagDep = true;
                    end                    
                    %
                    if state.curDir <saivdr.dictionary.utility. Direction.HORIZONTAL && ~permFlagHor
                        [arrayCoefsB,~] = ...
                            saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps3d.permuteCoefs_(arrayCoefsB,scale);
                        [arrayCoefsC,scale] = ...
                            saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps3d.permuteCoefs_(arrayCoefsC,scale);
                        permFlagHor = true;
                    end                                        
                    %
                    if iPdAng == 1
                        arrayCoefsB = step(vqStep_,arrayCoefsB,scale,...
                            pmCoefs,idxMtx);
                    end
                    dU = step(omgpd_,angs_,mus_,iPdAng); 
                    setParamMtx_(obj,dU,idxMtx);
                    pdCoefs = obj.paramMtxCoefs;
                    arrayCoefsA = step(vqStepPd_,arrayCoefsC,scale,...
                        pdCoefs,idxMtx);
                    %
                    grad(iAng) = -2*(arrayCoefsA(:).'*arrayCoefsB(:));
                    %
                    if iPdAng == nAngsPm
                        arrayCoefsC = step(vqStep_,arrayCoefsC,scale,pmCoefs,...
                            idxMtx);
                    end
                    %
                    if  state.curDir == saivdr.dictionary.utility.Direction.VERTICAL && ...
                            ( (state.curOrd == ord(saivdr.dictionary.utility.Direction.VERTICAL) && iPdAng == nAngsPm) || ...
                                ord(saivdr.dictionary.utility.Direction.VERTICAL) == 0 )
                        [arrayCoefsB,~] = ...
                            saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps3d.permuteCoefs_(arrayCoefsB,scale);
                        [arrayCoefsC,scale] = ...
                            saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps3d.permuteCoefs_(arrayCoefsC,scale);
                    end
                end
            end
        end
        
        function resetImpl(~)
        end
    end
    
    methods (Access = private)
        %%
        
        function state = getState_(obj,iAng)       
            ps          = obj.NumberOfSymmetricChannels;
            nAngsPerMtx = uint32(ps*(ps-1)/2);
            nMusPerMtx  = uint32(ps);
            ord         = obj.PolyPhaseOrder;
            ordZ        = uint32(ord(saivdr.dictionary.utility.Direction.DEPTH));                        
            ordX        = uint32(ord(saivdr.dictionary.utility.Direction.HORIZONTAL));            
            ordY        = uint32(ord(saivdr.dictionary.utility.Direction.VERTICAL));
            
            % Initial -> Depth -> Horizontal -> Vertical
            
            state.curIdxMtx = uint32(ceil(double(iAng)/double(nAngsPerMtx)));
            if iAng <= nAngsPerMtx       % dW0
                state.isMtxU      = false;
                state.curOrd      = uint32(0);
                state.iAngPerMtx  = uint32(iAng);
                state.startIdxAng = uint32(1);
                state.startIdxMu  = uint32(1);
                state.curDir      = 0;
            elseif iAng < 2*nAngsPerMtx  % dU0
                state.isMtxU      = true;
                state.curOrd      = uint32(0);
                state.iAngPerMtx  = uint32(iAng-nAngsPerMtx);
                state.startIdxAng = nAngsPerMtx+1;
                state.startIdxMu  = nMusPerMtx+1;
                state.curDir      = 0;
            elseif iAng == 2*nAngsPerMtx % dU0
                state.isMtxU      = true;
                state.curOrd      = uint32(0);
                state.iAngPerMtx  = uint32(iAng-nAngsPerMtx);
                state.startIdxAng = nAngsPerMtx+1;
                state.startIdxMu  = nMusPerMtx+1;
                state.curDir      = 0;
            elseif iAng < (ordZ+2)*nAngsPerMtx  % dUz
                state.isMtxU      = true;
                state.curOrd      = uint32(ceil(double(iAng-2*nAngsPerMtx)/double(nAngsPerMtx)));
                state.iAngPerMtx  = uint32(iAng-(state.curOrd+1)*nAngsPerMtx);
                state.startIdxAng = (state.curOrd+1)*nAngsPerMtx+1;
                state.startIdxMu  = (state.curOrd+1)*nMusPerMtx+1;
                state.curDir      = saivdr.dictionary.utility.Direction.DEPTH;
            elseif iAng == (ordZ+2)*nAngsPerMtx % dUz
                state.isMtxU      = true;
                state.curOrd      = uint32(ceil(double(iAng-2*nAngsPerMtx)/double(nAngsPerMtx)));
                state.iAngPerMtx  = uint32(iAng-(state.curOrd+1)*nAngsPerMtx);
                state.startIdxAng = (ordZ+1)*nAngsPerMtx+1;
                state.startIdxMu  = (ordZ+1)*nMusPerMtx+1;
                state.curDir      = saivdr.dictionary.utility.Direction.DEPTH;
            elseif iAng < (ordZ+ordX+2)*nAngsPerMtx % dUx
                state.isMtxU      = true;
                state.curOrd      = uint32(ceil(double(iAng-(double(ordZ)+2)*nAngsPerMtx)/double(nAngsPerMtx)));
                state.iAngPerMtx  = uint32(iAng-(ordZ+state.curOrd+1)*nAngsPerMtx);
                state.startIdxAng = (ordZ+state.curOrd+1)*nAngsPerMtx+1;
                state.startIdxMu  = (ordZ+state.curOrd+1)*nMusPerMtx+1;
                state.curDir      = saivdr.dictionary.utility.Direction.HORIZONTAL;
            elseif iAng == (ordZ+ordX+2)*nAngsPerMtx % dUx
                state.isMtxU      = true;
                state.curOrd      = uint32(ceil(double(iAng-(double(ordZ)+2)*nAngsPerMtx)/double(nAngsPerMtx)));
                state.iAngPerMtx  = uint32(iAng-(ordZ+state.curOrd+1)*nAngsPerMtx);
                state.startIdxAng = (ordZ+ordX+1)*nAngsPerMtx+1;
                state.startIdxMu  = (ordZ+ordX+1)*nMusPerMtx+1;
                state.curDir      = saivdr.dictionary.utility.Direction.HORIZONTAL;
            elseif iAng < (ordZ+ordX+ordY+2)*nAngsPerMtx % dUy
                state.isMtxU      = true;
                state.curOrd      = uint32(ceil(double(iAng-(double(ordZ+ordX)+2)*nAngsPerMtx)/double(nAngsPerMtx)));
                state.iAngPerMtx  = uint32(iAng-(ordZ+ordX+state.curOrd+1)*nAngsPerMtx);
                state.startIdxAng = (ordZ+ordX+state.curOrd+1)*nAngsPerMtx+1;
                state.startIdxMu  = (ordZ+ordX+state.curOrd+1)*nMusPerMtx+1;
                state.curDir      = saivdr.dictionary.utility.Direction.VERTICAL;
            else % dUy
                state.isMtxU      = true;
                state.curOrd      = uint32(ceil(double(iAng-(double(ordZ+ordX)+2)*nAngsPerMtx)/double(nAngsPerMtx)));
                state.iAngPerMtx  = uint32(iAng-(ordZ+ordX+state.curOrd+1)*nAngsPerMtx);
                state.startIdxAng = (ordZ+ordX+ordY+1)*nAngsPerMtx+1;
                state.startIdxMu  = (ordZ+ordX+ordY+1)*nMusPerMtx+1;
                state.curDir      = saivdr.dictionary.utility.Direction.VERTICAL;
            end
        end
        
        function setupParamMtx_(obj)
            ord = obj.PolyPhaseOrder;
            ps  = obj.NumberOfSymmetricChannels;
            pa  = obj.NumberOfAntisymmetricChannels;
            %
            paramMtxSzTab_ = zeros(sum(ord)+2, 2);
            for iOrd = 0:sum(ord)/2
                paramMtxSzTab_(2*iOrd+1,:) = [ ps ps ];
                paramMtxSzTab_(2*iOrd+2,:) = [ pa pa ];
            end
            %
            nRowsPm = size(paramMtxSzTab_,1);
            indexOfParamMtxSzTab_ = zeros(nRowsPm,3);
            cidx = 1;
            for iRow = uint32(1):nRowsPm
                indexOfParamMtxSzTab_(iRow,:) = ...
                    [ cidx paramMtxSzTab_(iRow,:)];
                cidx = cidx + prod(paramMtxSzTab_(iRow,:));
            end
            obj.paramMtxSzTab = paramMtxSzTab_;
            obj.indexOfParamMtxSzTab = indexOfParamMtxSzTab_;
        end
        
        function value = getParamMtx_(obj,index)
            startIdx  = obj.indexOfParamMtxSzTab(index,1);
            dimension = obj.indexOfParamMtxSzTab(index,2:3);
            nElements = prod(dimension);
            endIdx = startIdx + nElements - 1;
            value = reshape(... % <- Coder doesn't support this. (R2014a)
                obj.paramMtxCoefs(startIdx:endIdx),...
                dimension);
        end

        function setParamMtx_(obj,pMtx,index)
            startIdx  = obj.indexOfParamMtxSzTab(index,1);
            dimension = obj.indexOfParamMtxSzTab(index,2:3);
            nElements = prod(dimension);
            endIdx = startIdx + nElements - 1;
            obj.paramMtxCoefs(startIdx:endIdx) = pMtx(:);
        end
        
    end
    
    methods (Access = private, Static = true)
        
        function [arrayCoefs,scale] = permuteCoefs_(arrayCoefs,scale)
            nRows_ = scale(1);
            nCols_ = scale(2);
            nLays_ = scale(3);
            nRowsxnCols_ = nRows_*nCols_;
            tmpArray = arrayCoefs;
            for idx = 0:nRowsxnCols_-1
                arrayCoefs(:,idx*nLays_+1:(idx+1)*nLays_) = ...
                    tmpArray(:,idx+1:nRowsxnCols_:end);
            end
            scale(1) = nLays_;
            scale(2) = nRows_;
            scale(3) = nCols_;
        end
        
%         function [arrayCoefs,scale] = ipermuteCoefs_(arrayCoefs,scale)
%             nRows_ = scale(1);
%             nCols_ = scale(2);
%             nLays_ = scale(3);
%             nColsxnLays_ = nCols_*nLays_;
%             tmpArray = arrayCoefs;
%             for idx = 0:nColsxnLays_-1
%                 arrayCoefs(:,idx+1:nColsxnLays_:end) = ...
%                     tmpArray(:,idx*nRows_+1:(idx+1)*nRows_);
%             end
%             scale(1) = nCols_;
%             scale(2) = nLays_;
%             scale(3) = nRows_;
%         end
        
    end
end
