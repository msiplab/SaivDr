classdef OvsdLpPuFb2dTypeICostEvaluator < ... %#codegen
        saivdr.dictionary.nsoltx.design.AbstOvsdLpPuFbCostEvaluator
    %OVSDLPPUFB2DTYPEICOSTEVALUATOR Cost evaluator for Type-I NSOLT
    %
    % SVN identifier:
    % $Id: OvsdLpPuFb2dTypeICostEvaluator.m 868 2015-11-25 02:33:11Z sho $
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
    %
    
    properties (Access = protected, Constant = true)
        DATA_DIMENSION = 2
    end
    
    methods
        
        % Constractor
        function obj = OvsdLpPuFb2dTypeICostEvaluator(varargin)
            obj = obj@saivdr.dictionary.nsoltx.design.AbstOvsdLpPuFbCostEvaluator(...
                varargin{:});
        end
        
    end
    
    methods (Access=protected)
        
        function dim = getDataDimension(obj)
            dim = obj.DATA_DIMENSION;
        end
        
%         function s = saveObjectImpl(obj)
%             s = saveObjectImpl@saivdr.dictionary.nsoltx.design.AbstOvsdLpPuFbCostEvaluator(obj);
%         end
%         
%         function loadObjectImpl(obj,s,wasLocked)
%             loadObjectImpl@saivdr.dictionary.nsoltx.design.AbstOvsdLpPuFbCostEvaluator(obj,s,wasLocked);
%         end     
        
        function validatePropertiesImpl(~)
        end
        
        function setupImpl(obj,~,~,scales)
            
            nch = [ obj.NumberOfSymmetricChannels ...
                obj.NumberOfAntisymmetricChannels ];
            nChs = sum(nch);
            ord = uint32(obj.polyPhaseOrder);
            
            % Check nLeves
            nLevels = (size(scales,1)-1)/(nChs-1);
            if nLevels ~= 1
                error('Number of tree levels should be one.');
            end
            
            % Prepare MEX function
            if ~obj.isMexFcn
                import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_atomcnc2d
                [mexFcnAcnc, isMexFcnAcnc] = fcn_autobuild_atomcnc2d(nch);
                %
                import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_gradevalsteps2d
                [mexFcnGrad, isMexFcnGrad] = fcn_autobuild_gradevalsteps2d(nch,ord);
                %
                obj.isMexFcn = isMexFcnAcnc && isMexFcnGrad;
            end
            % Atom concatenator
            if ~isempty(mexFcnAcnc)
                obj.atomCncFcn = @(coefs,scale,pmcoefs,ord,fpe) ...
                    mexFcnAcnc(coefs,scale,pmcoefs,nch,ord,fpe);
            else
                import saivdr.dictionary.nsoltx.mexsrcs.fcn_NsoltAtomConcatenator2d
                clear fcn_NsoltAtomConcatenator2d
                obj.atomCncFcn = @(coefs,scale,pmcoefs,ord,fpe) ...
                    fcn_NsoltAtomConcatenator2d(coefs,scale,pmcoefs,nch,ord,fpe);
            end
            % Gradient evaluator
            if ~isempty(mexFcnGrad)
                obj.gradFcn = @(coefsB,coefsC,scale,pmCoefs,angs,mus,fpe,isnodc) ...
                    mexFcnGrad(coefsB,coefsC,scale,pmCoefs,angs,mus,nch,ord,fpe,isnodc);
            else
                import saivdr.dictionary.nsoltx.mexsrcs.fcn_GradEvalSteps2d
                clear fcn_GradEvalSteps2d
                obj.gradFcn = @(coefsB,coefsC,scale,pmCoefs,angs,mus,fpe,isnodc) ...
                    fcn_GradEvalSteps2d(coefsB,coefsC,scale,pmCoefs,angs,mus,nch,ord,fpe,isnodc);
            end
        end
        
        function [cost,grad] = stepImpl(obj,srcImg,coefs,scales)
            pmMtx = step(obj.LpPuFb,[],[]);
            %
            [recImg,intrCoefs] = synthesize_(obj, coefs, scales, pmMtx);
            difImg = srcImg-recImg;
            cost = sum(difImg(:).^2);
            %
            angs = get(obj.LpPuFb,'Angles');
            mus  = get(obj.LpPuFb,'Mus');
            isnodc = isa(obj.LpPuFb,...
                'saivdr.dictionary.nsoltx.OvsdLpPuFb2dTypeIVm1System');
            grad = gradient_(obj, difImg, intrCoefs, pmMtx, scales, ...
                angs, mus, isnodc);
        end
        
    end
    
    methods (Access = private)
        
        function grad = gradient_(obj, difImg, arrayCoefsB, pmMtx, scales,...
                angs, mus, isnodc)
            import saivdr.dictionary.utility.Direction
            %
            pmCoefs = get(pmMtx,'Coefficients');
            %
            ps  = obj.NumberOfSymmetricChannels;
            pa  = obj.NumberOfAntisymmetricChannels;
            nChs  = ps + pa;
            %
            decY_  = obj.decimationFactor(Direction.VERTICAL);
            decX_  = obj.decimationFactor(Direction.HORIZONTAL);
            %
            nRows_ = uint32(scales(1,1));
            nCols_ = uint32(scales(1,2));
            %
            scale_    = [ nRows_ nCols_];
            blockSize = [ decY_ decX_ ];
            
            % Prepare array
            arrayCoefsC = zeros(nChs,nRows_*nCols_);
            
            % Block DCT for difImg (Step 2)
            if decY_ == 1 && decX_ == 1
                coefs = im2col(difImg,blockSize,'distinct');
                arrayCoefsC(1,:) = coefs(1,:);
            elseif decY_ == 2 && decX_ == 2
                difImg1 = difImg(1:2:end,1:2:end);
                difImg2 = difImg(2:2:end,1:2:end);
                difImg3 = difImg(1:2:end,2:2:end);
                difImg4 = difImg(2:2:end,2:2:end);
                %
                difImg1 = difImg1(:).';
                difImg2 = difImg2(:).';
                difImg3 = difImg3(:).';
                difImg4 = difImg4(:).';
                %
                arrayCoefsC(1,:) = ...
                    (difImg1+difImg2+difImg3+difImg4)/2;
                arrayCoefsC(2,:) = ...
                    (difImg1-difImg2-difImg3+difImg4)/2;
                arrayCoefsC(ps+1,:) = ...
                    (difImg1-difImg2+difImg3-difImg4)/2;
                arrayCoefsC(ps+2,:) = ...
                    (difImg1+difImg2-difImg3-difImg4)/2;
            else
                mc = ceil(decX_*decY_/2);
                mf = floor(decX_*decY_/2);
                dctCoefs = blockproc(difImg,blockSize,...
                    @obj.dct2_);
                dctCoefs = blockproc(dctCoefs,blockSize,...
                    @obj.permuteDctCoefs_);
                coefs = im2col(dctCoefs,blockSize,'distinct');
                arrayCoefsC(1:mc,:) = coefs(1:mc,:);
                arrayCoefsC(ps+1:ps+mf,:) = coefs(mc+1:end,:);
            end
            
            % Gradient calculation steps
            %import saivdr.dictionary.nsoltx.mexsrcs.fcn_GradEvalSteps2d                   
            fpe = strcmp(obj.BoundaryOperation,'Circular');
            grad = obj.gradFcn(...
                arrayCoefsB, arrayCoefsC, scale_, pmCoefs, ...
                angs, mus, fpe, isnodc);
        end
        
        function [recImg,arrayCoefs] = synthesize_(obj,coefs,scales,pmMtx)
            import saivdr.dictionary.utility.Direction
            %
            pmCoefs = get(pmMtx,'Coefficients');
            nChs = obj.NumberOfSymmetricChannels ...
                + obj.NumberOfAntisymmetricChannels;
            %
            iSubband = 1;
            eIdx = prod(scales(iSubband,:));
            %
            height = scales(1,1);
            width  = scales(1,2);
            arrayCoefs = zeros(nChs,height*width);
            arrayCoefs(1,:) = coefs(1:eIdx);
            for iCh = 2:nChs
                iSubband = iSubband + 1;
                sIdx = eIdx + 1;
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                arrayCoefs(iCh,:) = coefs(sIdx:eIdx);
            end
            %
            ps = obj.NumberOfSymmetricChannels;
            nRows_ = uint32(height);
            nCols_ = uint32(width);
            decY_  = obj.decimationFactor(Direction.VERTICAL);
            decX_  = obj.decimationFactor(Direction.HORIZONTAL);
            nDec   = decY_*decX_;
            %
            blockSize = [ decY_ decX_ ];
            %
            if isinteger(arrayCoefs)
                arrayCoefs = double(arrayCoefs);
            end
            
            % Atom concatenation
            subScale  = [ nRows_ nCols_ ];
            ord  = uint32(obj.polyPhaseOrder);
            fpe = strcmp(obj.BoundaryOperation,'Circular');
            arrayCoefs = obj.atomCncFcn(arrayCoefs,subScale,pmCoefs,...
                ord,fpe);
            
            % Block IDCT
            if decY_ == 1 && decX_ == 1
                scale = double(subScale);
                coefs = zeros(nDec,nRows_*nCols_);
                coefs(1,:) = arrayCoefs(1,:);
                recImg = col2im(coefs,blockSize,scale,'distinct');
            elseif decY_ == 2 && decX_ == 2
                recImg = zeros(2*subScale);
                subCoef1 = arrayCoefs(1,:);
                subCoef2 = arrayCoefs(2,:);
                subCoef3 = arrayCoefs(ps+1,:);
                subCoef4 = arrayCoefs(ps+2,:);
                %
                recImg(1:2:end,1:2:end) = ...
                    reshape(subCoef1+subCoef2+subCoef3+subCoef4,subScale);
                recImg(2:2:end,1:2:end)  = ...
                    reshape(subCoef1-subCoef2-subCoef3+subCoef4,subScale);
                recImg(1:2:end,2:2:end)  = ...
                    reshape(subCoef1-subCoef2+subCoef3-subCoef4,subScale);
                recImg(2:2:end,2:2:end)  = ...
                    reshape(subCoef1+subCoef2-subCoef3-subCoef4,subScale);
                %
                recImg = recImg/2;
            else
                mc = ceil(decX_*decY_/2);
                mf = floor(decX_*decY_/2);
                coefs = zeros(nDec,size(arrayCoefs,2));
                coefs(1:mc,:) = arrayCoefs(1:mc,:);
                coefs(mc+1:end,:) = arrayCoefs(ps+1:ps+mf,:);
                scale = double(subScale) .* obj.decimationFactor;
                dctCoefs = col2im(coefs,blockSize,scale,'distinct');
                dctCoefs = blockproc(dctCoefs,blockSize,...
                    @obj.permuteIdctCoefs_);
                recImg = blockproc(dctCoefs,blockSize,...
                    @obj.idct2_);
            end
        end
        
    end
    
    methods (Access = private, Static = true)
        
        function value = idct2_(x)
            value = idct2(x.data);
        end
        
        function value = permuteIdctCoefs_(x)
            coefs = x.data;
            decY_ = x.blockSize(1);
            decX_ = x.blockSize(2);
            nQDecsee = ceil(decY_/2)*ceil(decX_/2);
            nQDecsoo = floor(decY_/2)*floor(decX_/2);
            nQDecsoe = floor(decY_/2)*ceil(decX_/2);
            cee = coefs(         1:  nQDecsee);
            coo = coefs(nQDecsee+1:nQDecsee+nQDecsoo);
            coe = coefs(nQDecsee+nQDecsoo+1:nQDecsee+nQDecsoo+nQDecsoe);
            ceo = coefs(nQDecsee+nQDecsoo+nQDecsoe+1:end);
            value = zeros(decY_,decX_);
            value(1:2:decY_,1:2:decX_) = ...
                reshape(cee,ceil(decY_/2),ceil(decX_/2));
            value(2:2:decY_,2:2:decX_) = ...
                reshape(coo,floor(decY_/2),floor(decX_/2));
            value(2:2:decY_,1:2:decX_) = ...
                reshape(coe,floor(decY_/2),ceil(decX_/2));
            value(1:2:decY_,2:2:decX_) = ...
                reshape(ceo,ceil(decY_/2),floor(decX_/2));
        end
        
        function value = dct2_(x)
            value = dct2(x.data);
        end
        
        function value = permuteDctCoefs_(x)
            coefs = x.data;
            decY_ = x.blockSize(1);
            decX_ = x.blockSize(2);
            cee = coefs(1:2:end,1:2:end);
            coo = coefs(2:2:end,2:2:end);
            coe = coefs(2:2:end,1:2:end);
            ceo = coefs(1:2:end,2:2:end);
            value = [ cee(:) ; coo(:) ; coe(:) ; ceo(:) ];
            value = reshape(value,decY_,decX_);
        end
        
    end
    
end