classdef Synthesis2dSystem < saivdr.dictionary.AbstSynthesisSystem
    %SYNTHESIS2DSYSTEM 2-D synthesis system
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
    properties (Nontunable)
        SynthesisFilters
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
    
    properties (Access = private)
        freqRes
    end
    
    methods
        
        % Constractor
        function obj = Synthesis2dSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            obj.nChs = size(obj.SynthesisFilters,3);
        end
        
        function setFrameBound(obj,frameBound)
            obj.FrameBound = frameBound;
        end
        
    end
    
    methods (Access = protected)
        
        function flag = isInactivePropertyImpl(obj,propertyName)
            if strcmp(propertyName,'UseGpu')
                flag = strcmp(obj.FilterDomain,'Frequeny');
            else
                flag = false;
            end
        end        
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj);
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
            loadObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj,s,wasLocked);
        end
        
        function setupImpl(obj,~,scales)
            import saivdr.dictionary.utility.Direction
            if obj.nChs < 2
                error('SaivDr: # of channels should be greater than 1.');
            end
            % Set up for frequency domain filtering
            isFrequency_ = strcmp(obj.FilterDomain,'Frequency');
            if obj.UseGpu && isFrequency_
                datatype_ = 'gpuArray';
            else
                datatype_ = 'double';
            end
            if isFrequency_
                decY = obj.DecimationFactor(Direction.VERTICAL);
                decX = obj.DecimationFactor(Direction.HORIZONTAL);
                nChs_  = obj.nChs;
                nLevels = (size(scales,1)-1)/(nChs_-1);
                nAllChs_ = size(scales,1);
                %
                nRows_ = scales(1,1)*decY^nLevels;
                nCols_ = scales(1,2)*decX^nLevels;
                iSubband = nAllChs_;
                freqRes_ = ones(nRows_,nCols_,nAllChs_,datatype_);
                for iLevel = 1:nLevels
                    dec_ = obj.DecimationFactor.^(iLevel-1);
                    phase_ = mod(obj.DecimationFactor+1,2);
                    for iCh = nChs_:-1:2
                        f    = obj.upsample2_(...
                            obj.SynthesisFilters(:,:,iCh),dec_,[0 0]);
                        fext = zeros(nRows_,nCols_,datatype_);
                        fext(1:size(f,1),1:size(f,2)) = f;
                        fext = circshift(fext,... % #TODO: Certification
                            (-floor(size(f)./(2*dec_))+phase_).*dec_);
                        freqRes_(:,:,iSubband) = ...
                            bsxfun(@times,freqRes_(:,:,1),...
                            fft2(fext,nRows_,nCols_));
                        iSubband = iSubband - 1;
                    end
                    f    = obj.upsample2_(...
                        obj.SynthesisFilters(:,:,1),dec_,[0 0]);
                    fext = zeros(nRows_,nCols_,datatype_);
                    fext(1:size(f,1),1:size(f,2)) = f;
                    fext = circshift(fext,...  % #TODO: Certification
                        (-floor(size(f)./(2*dec_))+phase_).*dec_);
                    freqRes_(:,:,1) = ...
                        bsxfun(@times,freqRes_(:,:,1),...
                        fft2(fext,nRows_,nCols_));
                end
                obj.freqRes = freqRes_;
            end
        end
        
        function recImg = stepImpl(obj,coefs,scales)
            if strcmp(obj.FilterDomain,'Spatial')
                recImg = synthesizeSpatial_(obj,coefs,scales);
            elseif obj.UseGpu
                coefs  = gpuArray(coefs);
                scales = gpuArray(scales);
                recImg = synthesizeFrequency_(obj,coefs,scales);
                recImg = gather(recImg);
            else 
                recImg = synthesizeFrequencyOrg_(obj,coefs,scales);
            end
        end
        
    end
    
    methods (Access = private)
        
        function recImg = synthesizeFrequencyOrg_(obj,coefs,scales)
            import saivdr.dictionary.utility.Direction
            %
            decY = obj.DecimationFactor(Direction.VERTICAL);
            decX = obj.DecimationFactor(Direction.HORIZONTAL);
            nChs_  = obj.nChs;
            nLevels = (size(scales,1)-1)/(nChs_-1);
            %
            iSubband = 1;
            eIdx = prod(scales(1,:));
            %
            subImg = reshape(coefs(1:eIdx),scales(1,:));            
            freqResSub = obj.freqRes(:,:,1);
            updImgFreq = repmat(fft2(subImg),[decY decX].^nLevels);
            recImgFreq = updImgFreq.*freqResSub;
            freqRes_ = obj.freqRes;
            for iLevel = 1:nLevels
                nDecs_ = [decY decX].^(nLevels-iLevel+1);
                for iCh = 2:nChs_
                    iSubband = iSubband + 1;
                    sIdx = eIdx + 1;
                    eIdx = sIdx + prod(scales(iSubband,:))-1;
                    freqResSub = freqRes_(:,:,iSubband);
                    subImg = reshape(coefs(sIdx:eIdx),scales(iSubband,:));
                    updImgFreq = repmat(fft2(subImg),nDecs_);
                    recImgFreq = recImgFreq + updImgFreq.*freqResSub;                    
                end
            end
            recImg = real(ifft2(recImgFreq));
        end
        
        function recImg = synthesizeFrequency_(obj,coefs,scales)
            import saivdr.dictionary.utility.Direction
            %
            decY = obj.DecimationFactor(Direction.VERTICAL);
            decX = obj.DecimationFactor(Direction.HORIZONTAL);
            nChs_  = obj.nChs;
            nLevels = (size(scales,1)-1)/(nChs_-1);
            %
            eSubband = 1;
            eIdx = prod(scales(1,:));
            %
            subImg = reshape(coefs(1:eIdx),scales(1,:));
            updImgFreq = zeros(size(subImg).*[decY decX].^nLevels,'like',subImg);
            updImgFreq(:,:,1) = repmat(fft2(subImg),[decY decX].^nLevels);
            freqRes_ = obj.freqRes;
            for iLevel = 1:nLevels
                nDecs_ = [decY decX].^(nLevels-iLevel+1);
                sSubband = eSubband + 1;                
                eSubband = sSubband + nChs_ - 2;
                sIdx = eIdx + 1;
                eIdx = sIdx + (nChs_-1)*prod(scales(sSubband,:))-1;                
                subImg = reshape(coefs(sIdx:eIdx),...
                    scales(sSubband,1),scales(sSubband,2),(nChs_-1)); 
                % Frequency domain upsampling
                updImgFreq(:,:,sSubband:eSubband) = ...
                    repmat(fft2(subImg),nDecs_(1),nDecs_(2));                
            end
            recImgFreq = bsxfun(@times,updImgFreq,freqRes_);
            recImg = real(ifft2(sum(recImgFreq,3)));
        end
        
        function recImg = synthesizeSpatial_(obj,coefs,scales)
            import saivdr.dictionary.utility.Direction
            %
            decY = obj.DecimationFactor(Direction.VERTICAL);
            decX = obj.DecimationFactor(Direction.HORIZONTAL);
            phaseY = mod(decY+1,2);
            phaseX = mod(decX+1,2);
            nChs_  = obj.nChs;
            nLevels = (size(scales,1)-1)/(nChs_-1);
            %
            iSubband = 1;
            eIdx = prod(scales(1,:));
            %
            recImg = reshape(coefs(1:eIdx),scales(1,:));
            for iLevel = 1:nLevels
                %
                f = obj.SynthesisFilters(:,:,1);
                % TODO: Polyphase implementation
                updImg = obj.upsample2_(recImg,[decY decX],...
                    [phaseY phaseX]);
                recImg = imfilter(updImg,f,'circ','conv');
                for iCh = 2:nChs_
                    iSubband = iSubband + 1;
                    sIdx = eIdx + 1;
                    eIdx = sIdx + prod(scales(iSubband,:))-1;
                    f = obj.SynthesisFilters(:,:,iCh);
                    subImg = reshape(coefs(sIdx:eIdx),scales(iSubband,:));
                    % TODO: Polyphase implementation
                    updImg = obj.upsample2_(subImg,[decY decX],...
                        [phaseY phaseX]);
                    recImg = recImg + imfilter(updImg,f,'circ','conv');
                end
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
