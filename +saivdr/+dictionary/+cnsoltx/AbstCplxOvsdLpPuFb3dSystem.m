classdef AbstCplxOvsdLpPuFb3dSystem < matlab.System %#codegen
    %ABSTOVSDLPPUFBMDSYSTEM Abstract class M-D OLPPUFB
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2014-2016, Kosuke FURUYA and Shogo MURAMATSU
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
        DecimationFactor = [ 2 2 2 ];
        PolyPhaseOrder   = [];
        NumberOfChannels = [];
        OutputMode = 'Coefficients'
    end

    properties (Hidden, Transient)
        OutputModeSet = ...
            matlab.system.StringSet({...
            'Coefficients',...
            'AnalysisFilters',...
            'AnalysisFilterAt',...
            'SynthesisFilters',...
            'SynthesisFilterAt',...
            'ParameterMatrixSet'});
    end

    properties (Hidden)
        Symmetry = 0;
        Angles = 0;
        Mus    = 1;
        ColorMapAtmImShow      = 'cool'
        SliceIntervalAtmImShow = 1/2;
        IsColorBarAtmImShow    = false;
    end

    properties (GetAccess = public, SetAccess = protected)
        ParameterMatrixSet
    end

    properties (Access = protected)
        Coefficients
    end

    properties(Access = protected, Logical)
        mexFlag = false
    end

    methods (Access = protected, Abstract = true)
        value = getAnalysisFilterBank_(obj)
        updateParameterMatrixSet_(obj)
        updateAngles_(obj)
        updateMus_(obj)
    end

    methods
        function obj = AbstCplxOvsdLpPuFb3dSystem(varargin)
            setProperties(obj,nargin,varargin{:});
        end

        function atmimshow(obj,varargin)
            % TODO: ï°ëfëŒâû
            % Show Atomic Images
            updateParameterMatrixSet_(obj);
            obj.mexFlag = false;
            H = getAnalysisFilterBank_(obj);
            H = flip(flip(flip(H,1),2),3);
            cmn = min(H(:));
            cmx = max(H(:));
            H = padarray(H,[2 2 2]);
            ly = size(H,1);
            lx = size(H,2);
            lz = size(H,3);
            %
            [x,y,z] = meshgrid(-(lx-1)/2:(lx-1)/2,-(ly-1)/2:(ly-1)/2,-(lz-1)/2:(lz-1)/2);
            for ib=1:sum(obj.NumberOfChannels)
                %hold off
                %
                subplot(2,obj.NumberOfChannels(1),ib);
                v = H(:,:,:,ib);
                v = 2*(v-cmn)/(cmx-cmn)-1;
                %{
                    ngd = 16;
                    hsg = surf(linspace(-lx/2,lx/2,ngd*lx),...
                        linspace(-ly/2,ly/2,ngd*ly),...
                        zeros(ngd*lx,ngd*ly));
                    rotate(hsg,[-1,1,0],-45)
                    xd = get(hsg,'XData');
                    yd = get(hsg,'YData');
                    zd = get(hsg,'ZData');
                    hsd = slice(x,y,z,v,xd,yd,zd,'nearest');
                    hold on
                    set(hsd,'FaceColor','interp',...
                        'EdgeColor','none',...
                        'DiffuseStrength',.8)
                    xslice = (lx-1)/2;
                    yslice = (ly-1)/2;
                    zslice = (lz-1)/2;
                %}
                sliceInterval = obj.SliceIntervalAtmImShow;
                xslice = -(lx-1)/2:sliceInterval:(lx-1)/2;
                yslice = -(ly-1)/2:sliceInterval:(ly-1)/2;
                zslice = -(lz-1)/2:sliceInterval:(lz-1)/2;
                hslice = slice(x,y,z,v,xslice,yslice,zslice,'nearest');
                set(hslice,'FaceColor','interp',...
                    'EdgeColor','none',...
                    'DiffuseStrength',.8);
                caxis([cmn cmx]);
                axis equal
                axis vis3d
                %grid off
                set(gca,'XTickLabel',[])
                set(gca,'YTickLabel',[])
                set(gca,'ZTickLabel',[])
                set(gca,'TickLength',[0 0])
                %xlabel('x')
                %ylabel('y')
                %zlabel('z')
                colormap(obj.ColorMapAtmImShow)
                if obj.IsColorBarAtmImShow
                    hcb = colorbar('location','southoutside');
                    if isprop(hcb,'YTickLabel')
                        set(hcb,'YTickLabel',[]);
                    end
                    if isprop(hcb,'XTickLabel')
                        set(hcb,'XTickLabel',[]);
                    end
                    if isprop(hcb,'ZTickLabel')
                        set(hcb,'ZTickLabel',[]);
                    end
                end
                for iSlice = 1:length(hslice)
                    map = abs(get(hslice(iSlice),'CData'));
                    set(hslice(iSlice),...
                        'AlphaDataMapping','scaled',...
                        'AlphaData',map.^1.6,...
                        'FaceAlpha','texture',...
                        'FaceColor','texture');
                end
                %{
                    for iz = lz-1:-1:1
                        pause(0.1)
                        zd = get(hslice(3),'ZData') - 1;
                        set(hslice(3),'ZData',zd);
                        set(hslice(3),'CData',v(:,:,iz))
                        drawnow
                    end
                    pause(0.1)
                    set(hslice,'Visible','off')
                %}
            end
        end

    end

    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.ParameterMatrixSet = matlab.System.saveObject(obj.ParameterMatrixSet);
            s.Coefficients = obj.Coefficients;
            s.mexFlag = obj.mexFlag;
        end

        function loadObjectImpl(obj,s,wasLocked)
            import saivdr.dictionary.utility.ParameterMatrixContainer
            obj.mexFlag = s.mexFlag;   
            obj.Coefficients = s.Coefficients;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
            %
            if strcmp(s.ParameterMatrixSet.ClassNameForLoadTimeEval,...
                    'saivdr.dictionary.utility.ParameterMatrixSet')
                s.ParameterMatrixSet.ClassNameForLoadTimeEval = ...
                    'saivdr.dictionary.utility.ParameterMatrixContainer';
            end            
            obj.ParameterMatrixSet = matlab.System.loadObject(s.ParameterMatrixSet);
        end

        function validatePropertiesImpl(obj)
            id = 'SaivDr:IllegalPropertyException';
            if prod(obj.DecimationFactor) > sum(obj.NumberOfChannels)
                error('%s:\n sum(NumberOfChannels) must be greater than or equalto prod(DecimationFactor).',...
                    id);
            end
        end

        function resetImpl(obj)
            obj.mexFlag = false;
        end

        function output = stepImpl(obj,varargin)

            if ~isempty(varargin{1})
                obj.Angles = varargin{1};
                updateAngles_(obj);
            end
            if ~isempty(varargin{2})
                obj.Mus = varargin{2};
                updateMus_(obj);
            end

            updateParameterMatrixSet_(obj);
            updateCoefficients_(obj);

            if strcmp(obj.OutputMode,'Coefficients')
                output = obj.Coefficients;
            elseif strcmp(obj.OutputMode,'ParameterMatrixSet')
                output = clone(obj.ParameterMatrixSet);
            elseif strcmp(obj.OutputMode,'AnalysisFilterAt')
                idx = varargin{3};
                H = getAnalysisFilterBank_(obj);
                output = H(:,:,:,idx);
            elseif strcmp(obj.OutputMode,'AnalysisFilters')
                output = getAnalysisFilterBank_(obj);
            elseif strcmp(obj.OutputMode,'SynthesisFilterAt')
                idx = varargin{3};
                H = getAnalysisFilterBank_(obj);
                H = conj(H);
                output = flip(H(:,:,:,idx),[1 2 3]);
            elseif strcmp(obj.OutputMode,'SynthesisFilters')
                H = getAnalysisFilterBank_(obj);
                H = conj(H);
                H = flip(H,1);
                H = flip(H,2);
                output = flip(H,3);
            else
                output = [];
            end
        end

        function N = getNumInputsImpl(obj)
            if strcmp(obj.OutputMode,'Coefficients') || ...
                    strcmp(obj.OutputMode,'ParameterMatrixSet') || ...
                    strcmp(obj.OutputMode,'AnalysisFilters') || ...
                    strcmp(obj.OutputMode,'SynthesisFilters')
                N = 2;
            else
                N = 3;
            end
        end

        function N = getNumOutputsImpl(~)
            N = 1;
        end

        function updateCoefficients_(obj)
            import saivdr.dictionary.utility.Direction
            decX = obj.DecimationFactor(Direction.HORIZONTAL);
            decY = obj.DecimationFactor(Direction.VERTICAL);
            decZ = obj.DecimationFactor(Direction.DEPTH);
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            ordZ = obj.PolyPhaseOrder(Direction.DEPTH);
            nChs = obj.NumberOfChannels;
            %
            H = getAnalysisFilterBank_(obj);
            nSubbands = sum(nChs);
            coefs = complex(zeros(nSubbands,decY*decX*decZ,ordY+1,ordX+1,ordZ+1));
            for iSubband = 1:nSubbands
                hi = H(:,:,:,iSubband);
                for iOrdZ = 0:ordZ
                    idxZ = iOrdZ*decZ;
                    for iOrdX = 0:ordX
                        idxX = iOrdX*decX;
                        for iOrdY = 0:ordY
                            idxY = iOrdY*decY;
                            b = hi(...
                                idxY+1:idxY+decY,...
                                idxX+1:idxX+decX,...
                                idxZ+1:idxZ+decZ);
                            coefs(iSubband,:,iOrdY+1,iOrdX+1,iOrdZ+1) = b(:).';
                        end
                    end
                end
            end
            obj.Coefficients = coefs;
        end
        
        function updateSymmetry_(obj)
            if isscalar(obj.Symmetry) && obj.Symmetry == 0
                obj.Symmetry = zeros(obj.NumberOfChannels,1);
            end
            if length(obj.Symmetry) ~= obj.NumberOfChannels
                %TODO: ó·äOèàóù
            end
        end

        function value = getMatrixE0_(obj)
            import saivdr.dictionary.utility.Direction
            import saivdr.utility.HermitianSymmetricDFT
            nRows = obj.DecimationFactor(Direction.VERTICAL);
            nCols = obj.DecimationFactor(Direction.HORIZONTAL);
            nDeps = obj.DecimationFactor(Direction.DEPTH);
            nElmBi = nRows*nCols*nDeps;
            coefs = complex(zeros(nElmBi));
            iElm = 1;
            for iRow = 1:nRows
                for iCol = 1:nCols
                    hsdftCoefYX = complex(zeros(nRows,nCols));
                    hsdftCoefYX(iRow,iCol) = 1;
                    basisYX = HermitianSymmetricDFT.ihsdft2(hsdftCoefYX);
                    for iDep = 1:nDeps
                        hsdftCoefZ = zeros(nDeps,1);
                        hsdftCoefZ(iDep) = 1;
                        %basisZ  = permute(idct(hsdftCoefZ),[2 3 1]);
                        basisZ = permute(HermitianSymmetricDFT.ihsdft(hsdftCoefZ),[2 3 1]);
                        basisVd = convn(basisZ,basisYX);
                        coefs(iElm,:) = basisVd(:).';
                        iElm = iElm + 1;
                    end
                end
            end
            %
            value = flip(coefs,2);
        end

        function value = permuteCoefs_(~,arr_,phs_)
            len_ = size(arr_,2)/phs_;
            value = zeros(size(arr_));
            for idx = 0:phs_-1
                value(:,idx*len_+1:(idx+1)*len_) = arr_(:,idx+1:phs_:end);
            end
        end
        
        function [initAngles,propAngles] = splitAngles_(obj)
            nCh = obj.NumberOfChannels;
            nInitAngles = nCh*(nCh-1)/2;
            
            initAngles = obj.Angles(1:nInitAngles);
            
            propAngles = obj.Angles(nInitAngles+1:end);
        end
        
%         function value = hsdftmtx_(~, nDec) %Hermitian-Symmetric DFT matrix
%             w = exp(-2*pi*1i/nDec);
%             value = complex(zeros(nDec));
%             for u = 0:nDec-1
%                 for x =0:nDec-1
%                     value(u+1,x+1) = w^(u*(x+0.5))/sqrt(nDec);
%                 end
%             end
%         end
%         
%         function value = ihsdft_(obj, X)
%             value = hsdftmtx_(obj, length(X))'*X;
%         end
%         
%         function value = ihsdft2_(obj, X)
%             [lenU, lenV] = size(X);
%             ihsdftU = hsdftmtx_(obj, lenU)';
%             ihsdftV = hsdftmtx_(obj, lenV)';
%             value = (ihsdftV*((ihsdftU*X).')).';
%         end

    end

end
