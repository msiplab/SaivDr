classdef Synthesis3dSystem < saivdr.dictionary.AbstSynthesisSystem
    %SYNTHESIS3DSYSTEM 3-D synthesis system
    %
    % Reference:
    %   Shogo Muramatsu and Hitoshi Kiya,
    %   ''Parallel Processing Techniques for Multidimensional Sampling
    %   Lattice Alteration Based on Overlap-Add and Overlap-Save Methods,''
    %   IEICE Trans. on Fundamentals, Vol.E78-A, No.8, pp.939-943, Aug. 1995
    %
    % SVN identifier:
    % $Id: Synthesis3dSystem.m 866 2015-11-24 04:29:42Z sho $
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
    properties (Nontunable)
        SynthesisFilters
        DecimationFactor = [2 2 2]
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

    properties (Access = private)
        freqRes
    end

    methods

        % Constractor
        function obj = Synthesis3dSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            obj.nChs = size(obj.SynthesisFilters,4);
        end

        function setFrameBound(obj,frameBound)
            obj.FrameBound = frameBound;
        end

    end

    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.nChs = obj.nChs;
            s.SynthesisFilters = obj.SynthesisFilters;
            s.FilterDomain = obj.FilterDomain;
            s.freqRes = obj.freqRes;
        end

        function loadObjectImpl(obj,s,wasLocked)
            obj.nChs = s.nChs;
            obj.SynthesisFilters = s.SynthesisFilters;
            obj.FilterDomain = s.FilterDomain;
            obj.freqRes = s.freqRes;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end

        function flag = isInactivePropertyImpl(obj,propertyName)
            if strcmp(propertyName,'UseGPU')
                flag = strcmp(obj.FilterDomain,'Spatial');
            else
                flag = false;
            end
        end

        function setupImpl(obj,~,scales)
            import saivdr.dictionary.utility.Direction

            % Set up for frequency domain filtering
            if strcmp(obj.FilterDomain,'Frequency')
                decY = obj.DecimationFactor(Direction.VERTICAL);
                decX = obj.DecimationFactor(Direction.HORIZONTAL);
                decZ = obj.DecimationFactor(Direction.DEPTH);
                nChs_  = obj.nChs;
                nLevels = (size(scales,1)-1)/(nChs_-1);
                nAllChs_ = size(scales,1);
                %
                nRows_ = scales(1,1)*decY^nLevels;
                nCols_ = scales(1,2)*decX^nLevels;
                nLays_ = scales(1,3)*decZ^nLevels;
                freqRes_ = ones(nRows_,nCols_,nLays_,nAllChs_);
                iSubband = nAllChs_;
                for iLevel = 1:nLevels
                    dec_ = obj.DecimationFactor.^(iLevel-1);
                    phase_ = mod(obj.DecimationFactor+1,2);
                    for iCh = nChs_:-1:2
                        f    = obj.upsample3_(...
                            obj.SynthesisFilters(:,:,:,iCh),dec_,[0 0 0]);
                        fext = zeros(nRows_,nCols_,nLays_);
                        fext(1:size(f,1),1:size(f,2),1:size(f,3)) = f;
                        fext = circshift(fext,... % #TODO: Certification
                            (-floor(size(f)./(2*dec_))+phase_).*dec_);
                        freqRes_(:,:,:,iSubband) = freqRes_(:,:,:,1) ...
                            .* fftn(fext,[nRows_,nCols_,nLays_]);
                        iSubband = iSubband - 1;
                    end
                    f    = obj.upsample3_(...
                        obj.SynthesisFilters(:,:,:,1),dec_,[0 0 0]);
                    fext = zeros(nRows_,nCols_,nLays_);
                    fext(1:size(f,1),1:size(f,2),1:size(f,3)) = f;
                    fext = circshift(fext,...  % #TODO: Certification
                        (-floor(size(f)./(2*dec_))+phase_).*dec_);
                    freqRes_(:,:,:,1) = freqRes_(:,:,:,1) ...
                        .* fftn(fext,[nRows_,nCols_,nLays_]);
                end
                obj.freqRes = freqRes_;
            end
        end

        function recImg = stepImpl(obj,coefs,scales)
            if strcmp(obj.FilterDomain,'Spatial')
                recImg = synthesizeSpatial_(obj,coefs,scales);
            else
                recImg = synthesizeFrequency_(obj,coefs,scales);
            end
        end

    end

    methods (Access = private)

        function recImg = synthesizeFrequency_(obj,coefs,scales)
            import saivdr.dictionary.utility.Direction
            %
            decY = obj.DecimationFactor(Direction.VERTICAL);
            decX = obj.DecimationFactor(Direction.HORIZONTAL);
            decZ = obj.DecimationFactor(Direction.DEPTH);
            nChs_  = obj.nChs;
            nLevels = (size(scales,1)-1)/(nChs_-1);
            %
            iSubband = 1;
            eIdx = prod(scales(1,:));
            %
            subImg = reshape(coefs(1:eIdx),scales(1,:));
            freqResSub = obj.freqRes(:,:,:,iSubband);
            updImgFreq = repmat(fftn(subImg),[decY decX decZ].^nLevels);
            recImgFreq = updImgFreq.*freqResSub;
            freqRes_ = obj.freqRes;
            for iLevel = 1:nLevels
                for iCh = 2:nChs_
                    iSubband = iSubband + 1;
                    sIdx = eIdx + 1;
                    eIdx = sIdx + prod(scales(iSubband,:))-1;
                    freqResSub = freqRes_(:,:,:,iSubband);
                    subImg = reshape(coefs(sIdx:eIdx),scales(iSubband,:));
                    updImgFreq = repmat(fftn(subImg),...
                        [decY decX decZ].^(nLevels-iLevel+1));
                    recImgFreq = recImgFreq + updImgFreq.*freqResSub;
                end
            end
            recImg = ifftn(recImgFreq);
        end

        function recImg = synthesizeSpatial_(obj,coefs,scales)
            import saivdr.dictionary.utility.Direction
            %
            decY = obj.DecimationFactor(Direction.VERTICAL);
            decX = obj.DecimationFactor(Direction.HORIZONTAL);
            decZ = obj.DecimationFactor(Direction.DEPTH);
            phaseY = mod(decY+1,2);
            phaseX = mod(decX+1,2);
            phaseZ = mod(decZ+1,2);
            nChs_  = obj.nChs;
            nLevels = (size(scales,1)-1)/(nChs_-1);
            %
            iSubband = 1;
            eIdx = prod(scales(1,:));
            %
            recImg = reshape(coefs(1:eIdx),scales(1,:));
            for iLevel = 1:nLevels
                %
                f = obj.SynthesisFilters(:,:,:,1);
                % TODO: Polyphase implementation
                updImg = obj.upsample3_(recImg,...
                    [decY decX decZ],...
                    [phaseY phaseX phaseZ]);
                recImg = imfilter(updImg,f,'circ','conv');
                for iCh = 2:nChs_
                    iSubband = iSubband + 1;
                    sIdx = eIdx + 1;
                    eIdx = sIdx + prod(scales(iSubband,:))-1;
                    f = obj.SynthesisFilters(:,:,:,iCh);
                    subImg = reshape(coefs(sIdx:eIdx),scales(iSubband,:));
                    % TODO: Polyphase implementation
                    updImg = obj.upsample3_(subImg,...
                        [decY decX decZ],...
                        [phaseY phaseX phaseZ]);
                    recImg = recImg + imfilter(updImg,f,'circ','conv');
                end
            end
        end

    end

    methods (Access = private, Static = true)

        function y = downsample3_(x,d)
            y = shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,...
                d(1)),1),d(2)),1),d(3)),1);
        end

        function y = upsample3_(x,d,p)
            y = shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,...
                d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);
        end
    end

end
