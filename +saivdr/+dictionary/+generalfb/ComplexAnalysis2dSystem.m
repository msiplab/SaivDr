classdef ComplexAnalysis2dSystem < saivdr.dictionary.AbstAnalysisSystem
    %COMPLEXANALYSIS2DSYSTEM 2-D analysis system
    %
    % Reference:
    %   Shogo Muramatsu and Hitoshi Kiya,
    %   ''Parallel Processing Techniques for Multidimensional Sampling
    %   Lattice Alteration Based on Overlap-Add and Overlap-Save Methods,''
    %   IEICE Trans. on Fundamentals, Vol.E78-A, No.8, pp.939-943, Aug. 1995
    %
    % SVN identifier:
    % $Id: Analysis2dSystem.m 866 2015-11-24 04:29:42Z sho $
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

    properties (Nontunable)
        AnalysisFilters
        DecimationFactor = [2 2]
        BoundaryOperation = 'Circular'
        FilterDomain = 'Spatial'
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
        function obj = ComplexAnalysis2dSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            obj.nChs = size(obj.AnalysisFilters,3);
        end
    end

    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
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
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end

        function validateInputsImpl(obj,srcImg,nLevels)
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
                        < size(obj.AnalysisFilters(:,:,1)) )
                    error('SaivDr: Input image should be larger.');
                end
            end

        end

        function setupImpl(obj,srcImg,nLevels)
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
            obj.allCoefs  = zeros(1,obj.nAllCoefs);
            obj.allScales = zeros(obj.nAllChs,obj.DATA_DIMENSION);

            % Set up for frequency domain filtering
            if strcmp(obj.FilterDomain,'Frequency')
                iSubband = obj.nAllChs;
                nRows_ = size(srcImg,1);
                nCols_ = size(srcImg,2);
                freqRes_ = ones(nRows_,nCols_,obj.nAllChs);
                for iLevel = 1:nLevels
                    dec_ = obj.DecimationFactor.^(iLevel-1);
                    for iCh = nChs_:-1:2
                        h    = obj.upsample2_(...
                            obj.AnalysisFilters(:,:,iCh),dec_,[0 0]);
                        hext = zeros(nRows_,nCols_);
                        hext(1:size(h,1),1:size(h,2)) = h;
                        hext = circshift(hext,...
                            -floor(size(h)./(2*dec_)).*dec_);
                        freqRes_(:,:,iSubband) = freqRes_(:,:,1) ...
                            .* fft2(hext,nRows_,nCols_);
                        iSubband = iSubband - 1;
                    end
                    h    = obj.upsample2_(...
                        obj.AnalysisFilters(:,:,1),dec_,[0 0]);
                    hext = zeros(nRows_,nCols_);
                    hext(1:size(h,1),1:size(h,2)) = h;
                    hext = circshift(hext,...
                        -floor(size(h)./(2*dec_)).*dec_);
                    freqRes_(:,:,1) = freqRes_(:,:,1) ...
                        .* fft2(hext,nRows_,nCols_);
                end
                obj.freqRes = freqRes_;
            end

        end

        function [coefs,scales] = stepImpl(obj,srcImg,nLevels)
            if strcmp(obj.FilterDomain,'Spatial')
                [coefs,scales] = analyzeSpatial_(obj,srcImg,nLevels);
            else
                [coefs,scales] = analyzeFrequency_(obj,srcImg,nLevels);
            end
        end

    end

    methods (Access = private)

        function [coefs,scales] = analyzeFrequency_(obj,srcImg,nLevels)
            % Frequency domain analysis

            import saivdr.dictionary.utility.Direction
            %
            nChs_  = obj.nChs;
            decY = obj.DecimationFactor(Direction.VERTICAL);
            decX = obj.DecimationFactor(Direction.HORIZONTAL);
            %
            iSubband = obj.nAllChs;
            eIdx     = obj.nAllCoefs;
            %
            freqSrcImg = fft2(srcImg);
            height = size(srcImg,1);
            width  = size(srcImg,2);
            freqRes_ = obj.freqRes;
            for iLevel = 1:nLevels
                nRows_ = height/(decY^iLevel);
                nCols_ = width/(decX^iLevel);
                for iCh = nChs_:-1:2
                    freqRefSub = freqRes_(:,:,iSubband);
                    freqSubImg = freqSrcImg.*freqRefSub;
                    U = 0;
                    for iPhsX=1:(decX^iLevel)
                        sIdxX = (iPhsX-1)*nCols_+1;
                        eIdxX = sIdxX + nCols_-1;
                        for iPhsY=1:(decY^iLevel)
                            sIdxY = (iPhsY-1)*nRows_+1;
                            eIdxY = sIdxY+nRows_-1;
                            U = U + freqSubImg(sIdxY:eIdxY,sIdxX:eIdxX);
                        end
                    end
                    subbandCoefs = ifft2(U)/((decY*decX)^iLevel);
                    obj.allScales(iSubband,:) = [ nRows_ nCols_ ];
                    sIdx = eIdx - (nRows_*nCols_) + 1;
                    obj.allCoefs(sIdx:eIdx) = subbandCoefs(:).';
                    iSubband = iSubband - 1;
                    eIdx = sIdx - 1;
                end
            end
            nRows_ = height/(decY^nLevels);
            nCols_ = width/(decX^nLevels);
            freqRefSub = freqRes_(:,:,1);
            freqSubImg = freqSrcImg.*freqRefSub;
            U = 0;
            for iPhsX=1:(decX^nLevels)
                sIdxX = (iPhsX-1)*nCols_+1;
                eIdxX = sIdxX + nCols_-1;
                for iPhsY=1:(decY^nLevels)
                    sIdxY = (iPhsY-1)*nRows_+1;
                    eIdxY = sIdxY+nRows_-1;
                    U = U + freqSubImg(sIdxY:eIdxY,sIdxX:eIdxX);
                end
            end
            subbandCoefs = ifft2(U)/((decY*decX)^nLevels);
            %
            obj.allScales(1,:) = [ nRows_ nCols_ ];
            obj.allCoefs(1:nRows_*nCols_) = subbandCoefs(:).';
            %
            scales = obj.allScales;
            coefs  = obj.allCoefs;
        end

        function [coefs,scales] = analyzeSpatial_(obj,srcImg,nLevels)
            % Spatial domain analysis

            import saivdr.dictionary.utility.Direction
            %
            nChs_  = obj.nChs;
            decY = obj.DecimationFactor(Direction.VERTICAL);
            decX = obj.DecimationFactor(Direction.HORIZONTAL);
            %
            iSubband = obj.nAllChs;
            eIdx     = obj.nAllCoefs;
            %
            subImg = srcImg;
            for iLevel = 1:nLevels
                height = size(subImg,1);
                width  = size(subImg,2);
                nRows_ = uint32(height/decY);
                nCols_ = uint32(width/decX);
                for iCh = nChs_:-1:2
                    h = obj.AnalysisFilters(:,:,iCh);
                    % TODO: Polyphase implementation
                    filtImg = imfilter(subImg,h,'circ','conv');
                    subbandCoefs = obj.downsample2_(filtImg,[decY decX]);
                    %
                    obj.allScales(iSubband,:) = [ nRows_ nCols_ ];
                    sIdx = eIdx - (nRows_*nCols_) + 1;
                    obj.allCoefs(sIdx:eIdx) = subbandCoefs(:).';
                    iSubband = iSubband - 1;
                    eIdx = sIdx - 1;
                end
                %
                h = obj.AnalysisFilters(:,:,1);
                % TODO: Polyphase implementation
                filtImg = imfilter(subImg,h,'circ','conv');
                subImg = obj.downsample2_(filtImg,[decY decX]);
                %
                obj.allScales(1,:) = [ nRows_ nCols_ ];
                obj.allCoefs(1:nRows_*nCols_) = subImg(:).';
                %
                scales = obj.allScales;
                coefs  = obj.allCoefs;
            end
        end

    end

    methods (Access = private, Static = true)

        function y = downsample2_(x,d)
            y = shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1);
        end

        function y = upsample2_(x,d,p)
            y = shiftdim(upsample(...
                shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1);
        end

    end

end
