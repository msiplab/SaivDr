classdef Analysis3dSystem < saivdr.dictionary.AbstAnalysisSystem
    %ANALYSIS3DSYSTEM 3-D analysis system
    %
    % Reference:
    %   Shogo Muramatsu and Hitoshi Kiya,
    %   ''Parallel Processing Techniques for Multidimensional Sampling
    %   Lattice Alteration Based on Overlap-Add and Overlap-Save Methods,''
    %   IEICE Trans. on Fundamentals, Vol.E78-A, No.8, pp.939-943, Aug. 1995
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015-2020, Shogo MURAMATSU
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
    properties (Access = protected, Constant = true)
        DATA_DIMENSION = 3
    end
    
    properties (Nontunable)
        AnalysisFilters
        DecimationFactor = [2 2 2]
        BoundaryOperation = 'Circular'
        FilterDomain = 'Spatial'
    end
    
    properties (Nontunable, PositiveInteger)
        NumberOfLevels
    end
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Circular'});
        FilterDomainSet = ...
            matlab.system.StringSet({'Spatial','Frequency'});
    end
    
    properties (Access = private, Nontunable, PositiveInteger)
        nChs
    end
    
    properties (Access = private, Nontunable)
        nAllCoefs
        nAllChs
    end
    
    properties (Access = private)
        allCoefs
        allScales
        freqRes
    end
    
    methods
        % Constractor
        function obj = Analysis3dSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            obj.nChs = size(obj.AnalysisFilters,4);
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj);
            s.nChs = obj.nChs;
            s.nAllCoefs  = obj.nAllCoefs;
            s.nAllChs    = obj.nAllChs;
            s.allScales  = obj.allScales;
            s.allCoefs   = obj.allCoefs;
            s.AnalysisFilters = obj.AnalysisFilters;
            s.FilterDomain = obj.FilterDomain;
            s.freqRes = obj.freqRes;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.nAllCoefs  = s.nAllCoefs;
            obj.nAllChs    = s.nAllChs;
            obj.allScales   = s.allScales;
            obj.allCoefs    = s.allCoefs;
            obj.nChs = s.nChs;
            obj.AnalysisFilters = s.AnalysisFilters;
            obj.FilterDomain = s.FilterDomain;
            obj.freqRes = s.freqRes;
            loadObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj,s,wasLocked);
        end
        
        function validateInputsImpl(obj,srcImg)
            nLevels = obj.NumberOfLevels;
            if nLevels < 1
                error('SaivDr: #Levels should be positive.');
            end
            
            if strcmp(obj.FilterDomain,'Frequency')
                dim_ = size(srcImg);
                if lcm(dim_,obj.DecimationFactor.^nLevels) ~= dim_
                    error(['SaivDr: Dimensions of input image should be '...
                        'multiples of DecimationFactor.^nLevels.']);
                end
                if any(size(srcImg)./(obj.DecimationFactor.^(nLevels-1)) ...
                        < size(obj.AnalysisFilters(:,:,:,1)) )
                    error('SaivDr: Input image should be larger.');
                end
            end
            
        end
        
        function setupImpl(obj,srcImg)
            nLevels = obj.NumberOfLevels;
            nChs_ = obj.nChs;
            nDec_ = prod(obj.DecimationFactor);
            if nDec_ == 1
                obj.nAllCoefs = uint32(numel(srcImg)*(...
                    (nChs_-1)*(nLevels/nDec_) + 1/nDec_^nLevels));
            else
                obj.nAllCoefs = uint32(numel(srcImg)*(...
                    (nChs_-1)/(nDec_-1)-...
                    (nChs_-nDec_)/((nDec_-1)*nDec_^nLevels)));
            end
            obj.nAllChs = nLevels*(nChs_-1)+1;
            
            % Set data types
            isFrequency_ = strcmp(obj.FilterDomain,'Frequency');
            if obj.UseGpu && isFrequency_
                datatype_ = 'gpuArray';
            else
                datatype_ = 'double';
            end
            obj.allCoefs  = zeros(1,obj.nAllCoefs,datatype_);
            obj.allScales = zeros(obj.nAllChs,obj.DATA_DIMENSION,datatype_);

            % Set up for frequency domain filtering            
            if obj.UseGpu
                datatype_ = 'gpuArray';
            else
                datatype_ = 'double';
            end
            if isFrequency_
                iSubband = obj.nAllChs;
                nRows_ = size(srcImg,1);
                nCols_ = size(srcImg,2);
                nLays_ = size(srcImg,3);
                freqRes_ = ones(nRows_,nCols_,nLays_,obj.nAllChs,datatype_);
                for iLevel = 1:nLevels
                    dec_ = obj.DecimationFactor.^(iLevel-1);
                    for iCh = nChs_:-1:2
                        h    = obj.upsample3_(...
                            obj.AnalysisFilters(:,:,:,iCh),dec_,[0 0 0]);
                        hext = zeros(nRows_,nCols_,nLays_,datatype_);
                        hext(1:size(h,1),1:size(h,2),1:size(h,3)) = h;
                        hext = circshift(hext,...
                            -floor(size(h,1)/(2*dec_(1)))*dec_(1),1);
                        hext = circshift(hext,...
                            -floor(size(h,2)/(2*dec_(2)))*dec_(2),2);
                        hext = circshift(hext,...
                            -floor(size(h,3)/(2*dec_(3)))*dec_(3),3);
                        freqRes_(:,:,:,iSubband) = bsxfun(@times,...
                            freqRes_(:,:,:,1),...
                            fftn(hext,[nRows_,nCols_,nLays_]));
                        iSubband = iSubband - 1;
                    end
                    h    = obj.upsample3_(...
                        obj.AnalysisFilters(:,:,:,1),dec_,[0 0 0]);
                    hext = zeros(nRows_,nCols_,nLays_,datatype_);
                    hext(1:size(h,1),1:size(h,2),1:size(h,3)) = h;
                    hext = circshift(hext,...
                        -floor(size(h,1)/(2*dec_(1)))*dec_(1),1);
                    hext = circshift(hext,...
                        -floor(size(h,2)/(2*dec_(2)))*dec_(2),2);
                    hext = circshift(hext,...
                        -floor(size(h,3)/(2*dec_(3)))*dec_(3),3);
                    freqRes_(:,:,:,1) = bsxfun(@times,...
                        freqRes_(:,:,:,1),...
                        fftn(hext,[nRows_,nCols_,nLays_]));
                end
                obj.freqRes = freqRes_;
            end
            
        end
        
        function [coefs,scales] = stepImpl(obj,srcImg)
            if strcmp(obj.FilterDomain,'Spatial')
                [coefs,scales] = analyzeSpatial_(obj,srcImg);
            elseif obj.UseGpu
                 srcImg = gpuArray(srcImg);
                [coefs,scales] = analyzeFrequency_(obj,srcImg);
                coefs  = gather(coefs);
                scales = gather(scales);
            else
                [coefs,scales] = analyzeFrequencyOrg_(obj,srcImg);
            end
            
        end
        
    end
    
    methods (Access = private)
        
        function [coefs,scales] = analyzeFrequencyOrg_(obj,srcImg)
            nLevels = obj.NumberOfLevels;
            
            % Frequency domain analysis
            
            import saivdr.dictionary.utility.Direction
            %
            nChs_  = obj.nChs;
            decY = obj.DecimationFactor(Direction.VERTICAL);
            decX = obj.DecimationFactor(Direction.HORIZONTAL);
            decZ = obj.DecimationFactor(Direction.DEPTH);
            %
            iSubband = obj.nAllChs;
            eIdx     = obj.nAllCoefs;
            %
            freqSrcImg = fftn(srcImg);
            height = size(srcImg,1);
            width  = size(srcImg,2);
            depth  = size(srcImg,3);
            freqRes_ = obj.freqRes;
            for iLevel = 1:nLevels
                nRows_ = height/(decY^iLevel);
                nCols_ = width/(decX^iLevel);
                nLays_ = depth/(decZ^iLevel);
                for iCh = nChs_:-1:2
                    freqResSub = freqRes_(:,:,:,iSubband);
                    freqSubImg = freqSrcImg.*freqResSub;
                    U = 0;
                    for iPhsZ=1:(decZ^iLevel)
                        sIdxZ = (iPhsZ-1)*nLays_+1;
                        eIdxZ = sIdxZ + nLays_-1;
                        for iPhsX=1:(decX^iLevel)
                            sIdxX = (iPhsX-1)*nCols_+1;
                            eIdxX = sIdxX + nCols_-1;
                            for iPhsY=1:(decY^iLevel)
                                sIdxY = (iPhsY-1)*nRows_+1;
                                eIdxY = sIdxY + nRows_-1;
                                U = U + freqSubImg(...
                                    sIdxY:eIdxY,...
                                    sIdxX:eIdxX,...
                                    sIdxZ:eIdxZ);
                            end
                        end
                    end
                    subbandCoefs = real(ifftn(U))/((decY*decX*decZ)^iLevel);
                    obj.allScales(iSubband,:) = [ nRows_ nCols_ nLays_];
                    sIdx = eIdx - (nRows_*nCols_*nLays_) + 1;
                    obj.allCoefs(sIdx:eIdx) = subbandCoefs(:).';
                    iSubband = iSubband - 1;
                    eIdx = sIdx - 1;
                end
            end
            nRows_ = height/(decY^nLevels);
            nCols_ = width/(decX^nLevels);
            nLays_ = depth/(decZ^nLevels);
            freqRes_ = obj.freqRes(:,:,:,1);
            freqSubImg = freqSrcImg.*freqRes_;
            U = 0;
            for iPhsZ=1:(decZ^nLevels)
                sIdxZ = (iPhsZ-1)*nLays_+1;
                eIdxZ = sIdxZ + nLays_-1;
                for iPhsX=1:(decX^nLevels)
                    sIdxX = (iPhsX-1)*nCols_+1;
                    eIdxX = sIdxX + nCols_-1;
                    for iPhsY=1:(decY^nLevels)
                        sIdxY = (iPhsY-1)*nRows_+1;
                        eIdxY = sIdxY + nRows_-1;
                        U = U + freqSubImg(...
                            sIdxY:eIdxY,...
                            sIdxX:eIdxX,...
                            sIdxZ:eIdxZ);
                    end
                end
            end
            subbandCoefs = real(ifftn(U))/((decY*decX*decZ)^nLevels);
            %
            obj.allScales(1,:) = [ nRows_ nCols_ nLays_];
            obj.allCoefs(1:nRows_*nCols_*nLays_) = subbandCoefs(:).';
            %
            scales = obj.allScales;
            coefs  = obj.allCoefs;
        end
        
        function [coefs,scales] = analyzeFrequency_(obj,srcImg)
            nLevels = obj.NumberOfLevels;
            
            % Frequency domain analysis
            
            import saivdr.dictionary.utility.Direction
            %
            nChs_  = obj.nChs;
            decY = obj.DecimationFactor(Direction.VERTICAL);
            decX = obj.DecimationFactor(Direction.HORIZONTAL);
            decZ = obj.DecimationFactor(Direction.DEPTH);
            %
            eSubband = obj.nAllChs;
            eIdx     = obj.nAllCoefs;
            %
            freqSrcImg = fftn(srcImg);
            height   = size(srcImg,1);
            width    = size(srcImg,2);
            depth    = size(srcImg,3);
            freqRes_ = obj.freqRes;           
            freqSrcImgRep_ = repmat(freqSrcImg,[1 1 1 (nChs_-1)]);
            %
            for iLevel = 1:nLevels
                nRows_ = height/(decY^iLevel);
                nCols_ = width/(decX^iLevel);
                nLays_ = depth/(decZ^iLevel);
                nDecs_ = (decY*decX*decZ)^iLevel;       
                sSubband = eSubband-(nChs_-1)+1;
                % Frequency responses
                freqResSubs_ = freqRes_(:,:,:,sSubband:eSubband);
                % Frequency domain filtering
                freqSubImgs_ = bsxfun(@times,freqSrcImgRep_,freqResSubs_);             
                % Frequency domain downsampling
                foldZ = reshape(freqSubImgs_,height,width,...
                    nLays_,(decZ^iLevel),(nChs_-1));
                faddZ = sum(foldZ,4);
                foldX = reshape(faddZ,height,...
                    nCols_,(decX^iLevel),nLays_,(nChs_-1));
                faddX = sum(foldX,3);
                foldY = reshape(faddX,nRows_,(decY^iLevel),...
                    nCols_,nLays_,(nChs_-1));
                faddY = sum(foldY,2);
                U     = squeeze(faddY);
                subbandCoefs = bsxfun(@times,...
                    real(ifft2(ifft(U,[],3))),1/nDecs_);                
                %
                sIdx = eIdx - (nChs_-1)*(nRows_*nCols_*nLays_) + 1;
                obj.allScales(sSubband:eSubband,:) = ...
                    repmat([ nRows_ nCols_ nLays_ ],[(nChs_-1) 1 1]);                
                obj.allCoefs(sIdx:eIdx) = subbandCoefs(:).';
                %
                eSubband = sSubband - 1;
                eIdx     = sIdx - 1;
            end
            nRows_ = height/(decY^nLevels);
            nCols_ = width/(decX^nLevels);
            nLays_ = depth/(decZ^nLevels);
            nDecs_ = (decY*decX*decZ)^nLevels;
            freqResSub = freqRes_(:,:,:,1);
            freqSubImg = bsxfun(@times,freqSrcImg,freqResSub);
            foldZ = reshape(freqSubImg,height,width,nLays_,(decZ^nLevels));
            faddZ = sum(foldZ,4);
            foldX = reshape(faddZ,height,nCols_,(decX^nLevels),nLays_);
            faddX = sum(foldX,3);
            foldY = reshape(faddX,nRows_,(decY^nLevels),nCols_,nLays_);
            faddY = sum(foldY,2);
            U     = squeeze(faddY);
            subbandCoefs = bsxfun(@times,...
                real(ifft2(ifft(U,[],3))),1/nDecs_);
            %
            obj.allScales(1,:) = [ nRows_ nCols_ nLays_];
            obj.allCoefs(1:nRows_*nCols_*nLays_) = subbandCoefs(:).';
            %
            scales = obj.allScales;
            coefs  = obj.allCoefs;
        end
        
        function [coefs,scales] = analyzeSpatial_(obj,srcImg)
            nLevels = obj.NumberOfLevels;
            
            % Spatial domain analysis
            
            import saivdr.dictionary.utility.Direction
            %
            nChs_  = obj.nChs;
            decY = obj.DecimationFactor(Direction.VERTICAL);
            decX = obj.DecimationFactor(Direction.HORIZONTAL);
            decZ = obj.DecimationFactor(Direction.DEPTH);
            nDecs = obj.DecimationFactor;
            %
            iSubband = obj.nAllChs;
            eIdx     = obj.nAllCoefs;
            %
            subImg = srcImg;
            for iLevel = 1:nLevels
                height = size(subImg,1);
                width  = size(subImg,2);
                depth  = size(subImg,3);
                nRows_ = uint32(height/decY);
                nCols_ = uint32(width/decX);
                nLays_ = uint32(depth/decZ);
                for iCh = nChs_:-1:2
                    h = obj.AnalysisFilters(:,:,:,iCh);
                    % TODO: Polyphase implementation
                    filtImg = imfilter(subImg,h,'circ','conv');
                    subbandCoefs  = obj.downsample3_(filtImg,nDecs);
                    %
                    obj.allScales(iSubband,:) = [nRows_ nCols_ nLays_];
                    sIdx = eIdx - (nRows_*nCols_*nLays_) + 1;
                    obj.allCoefs(sIdx:eIdx) = subbandCoefs(:).';
                    iSubband = iSubband - 1;
                    eIdx = sIdx - 1;
                end
                %
                h = obj.AnalysisFilters(:,:,:,1);
                % TODO: Polyphase implementation
                filtImg = imfilter(subImg,h,'circ','conv');
                subImg  = obj.downsample3_(filtImg,nDecs);
                %
                obj.allScales(1,:) = [nRows_ nCols_ nLays_];
                obj.allCoefs(1:nRows_*nCols_*nLays_) = subImg(:).';
                %
                scales = obj.allScales;
                coefs  = obj.allCoefs;
            end
        end
        
    end
    
    methods (Access = private, Static = true)
        
        
        function y = downsample3_(x,d)
            if size(x,3) > 1
                v = ipermute(downsample(permute(x,...
                    [3,1,2]),d(3)),[3,1,2]);
            else
                v = x;
            end
            if size(v,2) > 1
                v = ipermute(downsample(permute(v,...
                    [2,1,3]),d(2)),[2,1,3]);
            end
            if size(v,1) > 1
                y = downsample(v,d(1));
            else
                y = v;
            end
        end
        
        
        function y = upsample3_(x,d,p)
            if size(x,3) > 1
                v = ipermute(upsample(permute(x,...
                    [3,1,2]),d(3),p(3)),[3,1,2]);
            else
                u = cat(3,x,zeros(size(x,1),size(x,2),d(3)-1));
                v = circshift(u,[0 0 p(3)]);
            end
            if size(v,2) > 1
                v = ipermute(upsample(permute(v,...
                    [2,1,3]),d(2),p(2)),[2,1,3]);
            else
                u = cat(2,v,zeros(size(v,1),d(2)-1,size(v,3)));
                v = circshift(u,[0 p(2) 0]);
            end
            if size(v,1) > 1
                y = upsample(v,d(1),p(1));
            else
                u = cat(1,x,zeros(d(1)-1,size(v,2),size(v,3)));
                y = circshift(u,[p(1) 0 0]);
            end
        end
    end
    
end