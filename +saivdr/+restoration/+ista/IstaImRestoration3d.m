classdef IstaImRestoration3d < saivdr.restoration.ista.AbstIstaImRestoration %~#codegen
    %ISTAIMRESTORATION2D ISTA-based image restoration
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2017-2020, Shogo MURAMATSU
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
        r
        hu
        hx
        err
        threshold
    end
    
    methods
        function obj = IstaImRestoration3d(varargin)
            obj = obj@saivdr.restoration.ista.AbstIstaImRestoration(...
                varargin{:});
        end
    end
    
    methods(Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@...
                saivdr.restoration.ista.AbstIstaImRestoration(obj);
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            loadObjectImpl@...
                saivdr.restoration.ista.AbstIstaImRestoration(obj,s,wasLocked);
        end
        
        function setupImpl(obj,srcImg)
            setupImpl@saivdr.restoration.ista.AbstIstaImRestoration(obj,srcImg)
        end
        
        function resImg = stepImpl(obj,srcImg)
            % Initialization
            obj.x = srcImg;
            obj.nItr  = 0;
            % ^u = P.'r = P.'x
            obj.hu = step(obj.AdjLinProcess,obj.x);
            %  y = D.'P.'r =  D.'P.'x = D.'^u
            [ obj.y(:,1), obj.scales ] = ...
                step(obj.AdjOfSynthesizer,...
                obj.hu);
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
        end
        
    end
    
    methods (Access = private)
        
        function procPerIter_(obj)
            adjSyn_  = obj.AdjOfSynthesizer;
            syn_     = obj.Synthesizer;
            reciprocalL_  = 1/obj.valueL;
            scales_  = obj.scales;
            threshold_ = obj.threshold;
            
            % Processing per iteration
            
            % h = P.'r = P.'(^x-x)
            h_ = step(obj.AdjLinProcess,obj.r);
            %
            import saivdr.restoration.ista.AbstIstaImRestoration
            
            % ^v = D.'h = D.'P.'r = D.'P.'(^x-x)
            v_ = step(adjSyn_,h_);
            % y = softshrink(y -(1/L)*D.'P.'(^x-x))
            obj.y = AbstIstaImRestoration.softshrink_(...
                obj.y(:)-(reciprocalL_)*v_(:),threshold_);
            % ^u = Dy
            obj.hu = step(syn_,obj.y,scales_);
            
            % ^x = P^u = PDy
            obj.hx = step(obj.LinearProcess,obj.hu);
            % r = ^x - x;
            obj.r  = obj.hx - obj.x;
        end

    end
    
end

