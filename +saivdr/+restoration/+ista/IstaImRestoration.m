classdef IstaImRestoration < matlab.System %~#codegen
    %ISTAIMRESTORATION ISTA-based image restoration
    %
    % SVN identifier:
    % $Id: IstaImRestoration.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
    %

    properties(Nontunable)
        Synthesizer
        AdjOfSynthesizer
        LinearProcess
        NumberOfTreeLevels = 1
    end

    properties(Hidden,Nontunable)
        NumberOfComponents
    end

    properties
        StepMonitor
        Eps0   = 1e-6
        Lambda
    end

    properties (Logical)
        UseParallel = false;
    end

    properties (PositiveInteger)
        MaxIter = 1000
    end

    properties (Access = protected,Nontunable)
        AdjLinProcess
    end

    properties (Access = private)
        nItr
        x
        y
        r
        hu
        hx
        err
        valueL
        scales
        threshold
    end

    methods
        function obj = IstaImRestoration(varargin)
            setProperties(obj,nargin,varargin{:})
        end
    end

    methods(Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            %
            s.Synthesizer = ...
                matlab.System.saveObject(obj.Synthesizer);
            s.AdjOfSynthesizer = ...
                matlab.System.saveObject(obj.AdjOfSynthesizer);
            s.LinearProcess = ...
                matlab.System.saveObject(obj.LinearProcess);
            %
            s.AdjLinProcess = ...
                 matlab.System.saveObject(obj.AdjLinProcess);
%             s.nItr = obj.nItr;
%             s.x    = obj.x;
%             s.y    = obj.y;
%             s.r    = obj.r;
%             s.hu   = obj.hu;
%             s.hx   = obj.hx;
%             s.err  = obj.err;
             s.valueL = obj.valueL;
             s.scales = obj.scales;
%             s.threshold    = obj.threshold;
        end

        function loadObjectImpl(obj, s, wasLocked)
%             obj.nItr = s.nItr;
%             obj.x    = s.x;
%             obj.y    = s.y;
%             obj.r    = s.r;
%             obj.hu   = s.hu;
%             obj.hx   = s.hx;
%             obj.err  = s.err;
             obj.valueL = s.valueL;
             obj.scales = s.scales;
%             obj.threshold    = s.threshold;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
            %
            obj.Synthesizer = ...
                matlab.System.loadObject(s.Synthesizer);
            obj.AdjOfSynthesizer = ...
                matlab.System.loadObject(s.AdjOfSynthesizer);
            obj.LinearProcess = ...
                matlab.System.loadObject(s.LinearProcess);
            %
            obj.AdjLinProcess = ...
                matlab.System.loadObject(s.AdjLinProcess);

        end

        function validatePropertiesImpl(obj)
            if isempty(obj.Synthesizer)
                me = MException('SaivDr:InstantiationException',...
                    'Synthesizer must be given.');
                throw(me)
            end
            if isempty(obj.AdjOfSynthesizer)
                me = MException('SaivDr:InstantiationException',...
                    'AdjOfSynthesizer must be given.');
                throw(me)
            end
            if isempty(obj.LinearProcess)
                me = MException('SaivDr:InstantiationException',...
                    'LinearProcess must be given.');
                throw(me)
            end
            if ~strcmp(get(obj.LinearProcess,'ProcessingMode'),'Normal')
                error('SaivDr: Invalid processing mode')
            end
        end

        function setupImpl(obj,srcImg)
            obj.AdjLinProcess = clone(obj.LinearProcess);
            set(obj.AdjLinProcess,'ProcessingMode','Adjoint');
            obj.NumberOfComponents = size(srcImg,3);
            obj.x = srcImg;
            obj.valueL  = getLipschitzConstant_(obj);
        end

        function resetImpl(~)
            %            obj.valueL  = getLipschitzConstant_(obj);
        end

        function [resImg,coefvec,scales] = stepImpl(obj,srcImg)
            % Initialization
            obj.x = srcImg;
            obj.nItr  = 0;
            % ^u = P.'r = P.'x
            obj.hu = step(obj.AdjLinProcess,obj.x);
            %  y = D.'P.'r =  D.'P.'x = D.'^u
            for iCmp = 1:obj.NumberOfComponents
                [ obj.y(:,iCmp), obj.scales(:,:,iCmp) ] = ...
                    step(obj.AdjOfSynthesizer,...
                    obj.hu(:,:,iCmp),obj.NumberOfTreeLevels);
            end
            %  ^x = P^u = PP.'r = PP.'x
            obj.hx = step(obj.LinearProcess,obj.hu);
            % r = ^x - x;
            obj.r   = obj.hx - obj.x;
            obj.threshold   = obj.Lambda/obj.valueL;
            %
            if ~isempty(obj.StepMonitor)
                reset(obj.StepMonitor)
            end

            % Iterative processing
            obj.err = Inf;
            % ypre = y
            ypre = obj.y;
            while ( obj.err > obj.Eps0 && obj.nItr < obj.MaxIter )
                obj.nItr = obj.nItr + 1;
                % Process per iteration
                procPerIter_(obj);
                % ypst = y
                ypst = obj.y;
                % err = ||ypst-ypre||^2/||ypst||^2
                obj.err = norm(ypst(:) - ypre(:))^2/norm(ypst(:))^2;
                % Update
                ypre = ypst;
                % Monitoring
                if ~isempty(obj.StepMonitor)
                    step(obj.StepMonitor,obj.hu);
                end
            end
            resImg = obj.hu;
            if nargout > 1
                coefvec = obj.y;
                scales = obj.scales;
            end
         end

        function N = getNumInputsImpl(~)
            N = 1;
        end

        function N = getNumOutputsimpl(~)
            N = 1;
            %TODO:
        end

    end

    methods (Access = private)

        function procPerIter_(obj)
            adjSyn_  = obj.AdjOfSynthesizer;
            syn_     = obj.Synthesizer;
            nLevels_ = obj.NumberOfTreeLevels;
            reciprocalL_  = 1/obj.valueL;
            scales_  = obj.scales;
            threshold_ = obj.threshold;

            % Processing per iteration

            % h = P.'r = P.'(^x-x)
            h_ = step(obj.AdjLinProcess,obj.r);
            %
            import saivdr.restoration.ista.IstaImRestoration
            if obj.UseParallel
                y_         = cell(obj.NumberOfComponents);
                hu_        = cell(obj.NumberOfComponents);
                for iCmp = 1:obj.NumberOfComponents
                    y_{iCmp} = obj.y(:,iCmp);
                    hu_{iCmp} = obj.hu(:,:,iCmp);
                end
                parfor iCmp = 1:obj.NumberOfComponents
                    % ^v = D.'h = D.'P.'r = D.'P.'(^x-x)
                    v_ = step(adjSyn_,h_(:,:,iCmp),nLevels_);
                    % y = softshrink(y -(1/L)*D.'P.'(^x-x))
                    y_{iCmp} = IstaImRestoration.softshrink_(...
                        y_{iCmp}-(reciprocalL_)*v_(:),threshold_);
                    % ^u = Dy
                    hu_{iCmp} = ...
                        step(syn_,y_{iCmp},scales_(:,:,iCmp));
                end
                for iCmp = 1:obj.NumberOfComponents
                    obj.y(:,iCmp) = y_{iCmp};
                    obj.hu(:,:,iCmp) = hu_{iCmp};
                end
            else
                for iCmp = 1:obj.NumberOfComponents
                    % ^v = D.'h = D.'P.'r = D.'P.'(^x-x)
                    v_ = ...
                        step(adjSyn_,h_(:,:,iCmp),nLevels_);
                    % y = softshrink(y -(1/L)*D.'P.'(^x-x))
                    obj.y(:,iCmp) = IstaImRestoration.softshrink_(...
                        obj.y(:,iCmp)-(reciprocalL_)*v_(:),threshold_);
                    % ^u = Dy
                    obj.hu(:,:,iCmp) = ...
                        step(syn_,obj.y(:,iCmp),scales_(:,:,iCmp));
                end
            end
            %
            % ^x = P^u = PDy
            obj.hx = step(obj.LinearProcess,obj.hu);
            % r = ^x - x;
            obj.r  = obj.hx - obj.x;
        end

        function value = getLipschitzConstant_(obj)
            B_ = get(obj.Synthesizer,'FrameBound');
            step(obj.LinearProcess,obj.x);
            value =B_*get(obj.LinearProcess,'LambdaMax');
        end

    end

    methods (Static = true, Access = private)
        % Soft shrink
        function outputcf = softshrink_(inputcf,threshold)
            % Soft-thresholding shrinkage
%             cfr = real(inputcf);
%             cfi = imag(inputcf);
%             outputcf = max(1.0-threshold./cfr).*cfr + 1i*max(1.0-threshold./cfi).*cfi;
            outputcf = max(1.0-threshold./abs(inputcf),0).*inputcf;
        end

    end

end
