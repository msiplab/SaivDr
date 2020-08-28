classdef FistaImRestoration2d < saivdr.restoration.ista.AbstIstaImRestoration %~#codegen
    %FistaImRestoration2d ISTA-based image restoration
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2020, Shogo MURAMATSU
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
    
    properties (Access = protected)
        nItr
        y
        w
        r
        hu
        hx
        err
        threshold
        wpre
        tpre
    end
    
    properties(Hidden,Nontunable)
        NumberOfComponents
    end
    
    methods
        function obj = FistaImRestoration2d(varargin)
            obj = obj@saivdr.restoration.ista.AbstIstaImRestoration(...
                varargin{:});
        end
    end
    
    methods(Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.restoration.ista.AbstIstaImRestoration(obj);
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            loadObjectImpl@saivdr.restoration.ista.AbstIstaImRestoration(obj,s,wasLocked);
        end
        
        function setupImpl(obj,srcImg)
            setupImpl@saivdr.restoration.ista.AbstIstaImRestoration(obj,srcImg)
            obj.NumberOfComponents = size(srcImg,3);
        end
        
        function resImg = stepImpl(obj,srcImg)
            % Initialization
            obj.x = srcImg;
            obj.nItr  = 0;
            % ^u = P.'r = P.'x
            obj.hu = step(obj.AdjLinProcess,obj.x);
            %  wpre = D.'P.'r =  D.'P.'x = D.'^u
            for iCmp = 1:obj.NumberOfComponents
                [ obj.wpre(:,iCmp), obj.scales(:,:,iCmp) ] = ...
                    step(obj.AdjOfSynthesizer,...
                    obj.hu(:,:,iCmp));
            end
            obj.y = obj.wpre;
            obj.w = zeros(size(obj.wpre),'like',obj.wpre);
            %  ^x = P^u = PP.'r = PP.'x
            obj.hx = step(obj.LinearProcess,obj.hu);
            % r = ^x - x;
            obj.r  = obj.hx - obj.x;
            obj.threshold   = obj.Lambda/obj.valueL;
            %
            if ~isempty(obj.StepMonitor)
                reset(obj.StepMonitor)
            end
            
            % Iterative processing
            obj.err = Inf;
            obj.tpre = 1;
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
        end
        
    end
    
    methods (Access = private)
        
        function procPerIter_(obj)
            adjSyn_      = obj.AdjOfSynthesizer;
            syn_         = obj.Synthesizer;
            reciprocalL_ = 1/obj.valueL;
            scales_      = obj.scales;
            threshold_   = obj.threshold;
            nComps_      = obj.NumberOfComponents;
            
            % Processing per iteration
            t_ = (1+sqrt(1+4*obj.tpre^2))/2;
            tau_ = ((obj.tpre-1)/t_);
            % h = P.'r = P.'(^x-x)
            h_ = step(obj.AdjLinProcess,obj.r);
            %
            import saivdr.restoration.ista.AbstIstaImRestoration
            if obj.UseParallel
                w_    = cell(nComps_);
                wpre_ = cell(nComps_);
                y_    = cell(nComps_);
                hu_   = cell(nComps_);
                for iCmp = 1:nComps_
                    w_{iCmp}  = obj.w(:,iCmp);
                    y_{iCmp}  = obj.y(:,iCmp);
                    hu_{iCmp} = obj.hu(:,:,iCmp);
                end
                parfor iCmp = 1:nComps_
                    % ^v = D.'h = D.'P.'r = D.'P.'(^x-x)
                    v_ = step(adjSyn_,h_(:,:,iCmp));
                    % y = softshrink(y -(1/L)*D.'P.'(^x-x))
                    w_{iCmp} = AbstIstaImRestoration.softshrink_(...
                        y_{iCmp}-(reciprocalL_)*v_(:),threshold_);
                    %
                    y_{iCmp} = w_{iCmp} + tau_*(w_{iCmp}-wpre_{iCmp});
                    wpre_{iCmp} = w_{iCmp};
                    % ^u = Dy
                    hu_{iCmp} = step(syn_,y_{iCmp},scales_(:,:,iCmp));
                end
                for iCmp = 1:nComps_
                    obj.w(:,iCmp) = w_{iCmp};
                    obj.wpre(:,iCmp) = wpre_{iCmp};
                    obj.y(:,iCmp) = y_{iCmp};
                    obj.hu(:,:,iCmp) = hu_{iCmp};
                end
            else
                for iCmp = 1:nComps_
                    % ^v = D.'h = D.'P.'r = D.'P.'(^x-x)
                    v_ = step(adjSyn_,h_(:,:,iCmp));
                    % y = softshrink(y -(1/L)*D.'P.'(^x-x))
                    obj.w(:,iCmp) = AbstIstaImRestoration.softshrink_(...
                        obj.y(:,iCmp)-(reciprocalL_)*v_(:),threshold_);
                    %
                    obj.y(:,iCmp) = obj.w(:,iCmp) + ...
                        tau_*(obj.w(:,iCmp)-obj.wpre(:,iCmp));
                    obj.wpre = obj.w;
                    % ^u = Dy
                    obj.hu(:,:,iCmp) = ...
                        step(syn_,obj.y(:,iCmp),scales_(:,:,iCmp));
                end
            end
            % ^x = P^u = PDy
            obj.hx = step(obj.LinearProcess,obj.hu);
            % r = ^x - x;
            obj.r  = obj.hx - obj.x;
            %
            obj.tpre = t_;
        end
    end
    
end

