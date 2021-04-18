classdef GradEvalSteps2d < matlab.System %#codegen
    %GRADEVALSTEPS2D Gradient Evaluation Steps for 2-D NSOLT
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015-2021, Shogo MURAMATSU
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
    
    % Public, tunable properties.
    properties (PositiveInteger)
        NumberOfSymmetricChannels     = 2
        NumberOfAntisymmetricChannels = 2
    end
    
    properties (Logical)
        IsPeriodicExt = false
    end
    
    properties
        PolyPhaseOrder = [ 0 0 ]
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
        
        function obj = GradEvalSteps2d(varargin)
            setProperties(obj,nargin,varargin{:});
            
            obj.vqStep = saivdr.dictionary.nsoltx.design.NsoltVQStep2d(...
                'PartialDifference','off');
            
            obj.vqStepPd = saivdr.dictionary.nsoltx.design.NsoltVQStep2d(...
                'PartialDifference','on');

            obj.omgpd = saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');         
            
        end
    end
    
    methods (Access = protected)

        function processTunedPropertiesImpl(obj)
            if isChangedProperty(obj,'IsPeriodicExt')
                fpe = obj.IsPeriodicExt;
                set(obj.vqStep,'IsPeriodicExt',fpe);
                set(obj.vqStepPd,'IsPeriodicExt',fpe);
            end
            propChange = ...
                isChangedProperty(obj,'NumberOfSymmetricChannels') ||...
                isChangedProperty(obj,'NumberOfAntisymmetricChannels') ||...
                isChangedProperty(obj,'PolyPhaseOrder');
            if propChange            
                ord = obj.PolyPhaseOrder;
                ps = obj.NumberOfSymmetricChannels;
                pa = obj.NumberOfAntisymmetricChannels;
                set(obj.vqStep,'PolyPhaseOrder',ord);
                set(obj.vqStepPd,'PolyPhaseOrder',ord);
                set(obj.omgpd,'NumberOfDimensions',ps);
                set(obj.vqStep,'NumberOfSymmetricChannels',ps);
                set(obj.vqStep,'NumberOfAntisymmetricChannels',pa);
                set(obj.vqStepPd,'NumberOfSymmetricChannels',ps);
                set(obj.vqStepPd,'NumberOfAntisymmetricChannels',pa);
                nAngsPm = ps*(ps-1)/2;
                nMusPm  = ps;
                obj.omgpd.reset();
                obj.omgpd.step(zeros(nAngsPm,1),ones(nMusPm,1),uint32(0));
                setupParamMtx_(obj);
                %
                obj.paramMtxCoefs = zeros((ps^2)*(2+sum(ord)),1);                
            end
        end
        
        function setupImpl(obj,...
                ~, ~, ~, ~, ~, ~, ~ )
            ord = obj.PolyPhaseOrder;
            fpe = obj.IsPeriodicExt;
            ps = obj.NumberOfSymmetricChannels;
            pa = obj.NumberOfAntisymmetricChannels;
            
            set(obj.vqStep,'IsPeriodicExt',fpe);
            set(obj.vqStepPd,'IsPeriodicExt',fpe);            
            set(obj.vqStep,'PolyPhaseOrder',ord);
            set(obj.vqStepPd,'PolyPhaseOrder',ord);
            set(obj.omgpd,'NumberOfDimensions',ps);
            set(obj.vqStep,'NumberOfSymmetricChannels',ps);
            set(obj.vqStep,'NumberOfAntisymmetricChannels',pa);
            set(obj.vqStepPd,'NumberOfSymmetricChannels',ps);
            set(obj.vqStepPd,'NumberOfAntisymmetricChannels',pa);
            
            nAngsPm = ps*(ps-1)/2;
            nMusPm  = ps;
            obj.omgpd.reset()
            obj.omgpd.step(zeros(nAngsPm,1),ones(nMusPm,1),uint32(0));
            %
            setupParamMtx_(obj);
            obj.paramMtxCoefs = zeros((ps^2)*(2+sum(ord)),1);
        end
        
        function grad = stepImpl(obj, ...
                arrayCoefsB, arrayCoefsC, scale, pmCoefs, ...
                angs, mus, isnodc )
            %
            ord       = obj.PolyPhaseOrder;
            ps        = obj.NumberOfSymmetricChannels;
            %
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
                %
                if iPdAng == 1
                    omgpd_.reset();
                    omgpd_.step(angs_,mus_,uint32(0));
                end
                
                % Steps 3-5
                if state.curOrd == 0 && state.curDir == 0
                    if ~state.isMtxU
                        if isnodc && iAng < ps
                            omgpd_.step(angs_,mus_,iPdAng);
                            dW0 = zeros(ps);
                        else
                            dW0 = omgpd_.step(angs_,mus_,iPdAng); % TODO
                        end
                        setParamMtx_(obj,dW0,idxMtx); % TODO
                    else
                        dU0 = omgpd_.step(angs_,mus_,iPdAng); % TODO
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
                if state.curDir > 0
                    %
                    if state.curDir == saivdr.dictionary.utility.Direction.VERTICAL && ~permFlagHor
                        [arrayCoefsB,~] = ...
                            saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps2d.permuteCoefs_(arrayCoefsB,scale);
                        [arrayCoefsC,scale] = ...
                            saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps2d.permuteCoefs_(arrayCoefsC,scale);
                        permFlagHor = true;
                    end 
                    
                    if iPdAng == 1
                        arrayCoefsB = step(vqStep_,arrayCoefsB,scale,...
                            pmCoefs,idxMtx);
                    end
                    dUn = omgpd_.step(angs_,mus_,iPdAng); % TODO
                    setParamMtx_(obj,dUn,idxMtx); 
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
                            saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps2d.permuteCoefs_(arrayCoefsB,scale);
                        [arrayCoefsC,scale] = ...
                            saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps2d.permuteCoefs_(arrayCoefsC,scale);
                    end
                end
            end
        end
        
%         function resetImpl(~)
%         end
    end
    
    methods (Access = private)
        
        function state = getState_(obj,iAng)       
            ps          = obj.NumberOfSymmetricChannels;
            nAngsPerMtx = uint32(ps*(ps-1)/2);
            nMusPerMtx  = uint32(ps);
            ord         = obj.PolyPhaseOrder;
            ordY        = uint32(ord(saivdr.dictionary.utility.Direction.VERTICAL));
            ordX        = uint32(ord(saivdr.dictionary.utility.Direction.HORIZONTAL));
            
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
            elseif iAng < (ordX+2)*nAngsPerMtx  % dUx
                state.isMtxU      = true;
                state.curOrd      = uint32(ceil(double(iAng-2*nAngsPerMtx)/double(nAngsPerMtx)));
                state.iAngPerMtx  = uint32(iAng-(state.curOrd+1)*nAngsPerMtx);
                state.startIdxAng = (state.curOrd+1)*nAngsPerMtx+1;
                state.startIdxMu  = (state.curOrd+1)*nMusPerMtx+1;
                state.curDir      = saivdr.dictionary.utility.Direction.HORIZONTAL;
            elseif iAng == (ordX+2)*nAngsPerMtx % dUx
                state.isMtxU      = true;
                state.curOrd      = uint32(ceil(double(iAng-2*nAngsPerMtx)/double(nAngsPerMtx)));
                state.iAngPerMtx  = uint32(iAng-(state.curOrd+1)*nAngsPerMtx);
                state.startIdxAng = (ordX+1)*nAngsPerMtx+1;
                state.startIdxMu  = (ordX+1)*nMusPerMtx+1;
                state.curDir      = saivdr.dictionary.utility.Direction.HORIZONTAL;
            elseif iAng < (ordX+ordY+2)*nAngsPerMtx % dUy
                state.isMtxU      = true;
                state.curOrd      = uint32(ceil(double(iAng-(double(ordX)+2)*nAngsPerMtx)/double(nAngsPerMtx)));
                state.iAngPerMtx  = uint32(iAng-(ordX+state.curOrd+1)*nAngsPerMtx);
                state.startIdxAng = (ordX+state.curOrd+1)*nAngsPerMtx+1;
                state.startIdxMu  = (ordX+state.curOrd+1)*nMusPerMtx+1;
                state.curDir      = saivdr.dictionary.utility.Direction.VERTICAL;
            else                                    % dUy
                state.isMtxU      = true;
                state.curOrd      = uint32(ceil(double(iAng-(double(ordX)+2)*nAngsPerMtx)/double(nAngsPerMtx)));
                state.iAngPerMtx  = uint32(iAng-(ordX+state.curOrd+1)*nAngsPerMtx);
                state.startIdxAng = (ordX+ordY+1)*nAngsPerMtx+1;
                state.startIdxMu  = (ordX+ordY+1)*nMusPerMtx+1;
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
            tmpArray = arrayCoefs;
            for idx = 0:nRows_-1
                arrayCoefs(:,idx*nCols_+1:(idx+1)*nCols_) = ...
                    tmpArray(:,idx+1:nRows_:end);
            end
            scale(1) = nCols_;
            scale(2) = nRows_;
        end
        
        function [arrayCoefs,scale] = ipermuteCoefs_(arrayCoefs,scale)
            nRows_ = scale(1);
            nCols_ = scale(2);
            tmpArray = arrayCoefs;
            for idx = 0:nCols_-1
                arrayCoefs(:,idx+1:nCols_:end) = ...
                    tmpArray(:,idx*nRows_+1:(idx+1)*nRows_);
            end
            scale(1) = nCols_;
            scale(2) = nRows_;
        end
        
    end
end
