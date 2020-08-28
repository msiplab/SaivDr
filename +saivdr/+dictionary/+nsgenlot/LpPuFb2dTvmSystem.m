classdef LpPuFb2dTvmSystem < saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeISystem
    %LPPUFB2DTVMSYSTEM 2-D LPPUFB with 2-order trend vanishing moments
    %
    % Reference:
    %   Shogo Muramatsu, Dandan Han, Tomoya Kobayashi and Hisakazu Kikuchi,
    %   ''Directional Lapped Orthogonal Transform: Theory and Design,'' 
    %   IEEE Trans. on Image Proc., Vol.21, No.5, pp.2434-2448, 
    %    DOI: 10.1109/TIP.2011.2182055, May 2012
    %
    % SVN identifier:
    % $Id: LpPuFb2dTvmSystem.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
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
    % http://msiplab.eng.niigata-u.ac.jp/    
    %
    properties (Nontunable)
        TvmAngleInDegree = 0;% degree    
    end
    
    properties
        DirectionOfTriangle = 1;
    end
    
    properties (SetAccess = private, GetAccess = public)
        lenx3
        lambdaueq
    end
    
    properties (Access = private)
        omgs_
        ops_
        x3_
        ckTvm_
    end
    
    methods
        
        function obj = LpPuFb2dTvmSystem(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            import saivdr.dictionary.utility.OrthogonalProjectionSystem
            obj = obj@saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeISystem(...
                varargin{:});
            obj.omgs_ = OrthonormalMatrixGenerationSystem();
            obj.ops_  = OrthogonalProjectionSystem();
            
            obj.TvmAngleInDegree = mod(obj.TvmAngleInDegree+45,180)-45;
            
            if sum(obj.PolyPhaseOrder) < 2
                id = 'SaivDr:IllegalArgumentException';
                msg = 'Order must be greater than or equal to 2';
                me = MException(id, msg);
                throw(me);
            end
            
            import saivdr.dictionary.utility.Direction
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            if ~(ordX >= 2 && obj.TvmAngleInDegree <= 45) && ... % Table I
                    ~(ordY >= 2 && ...
                    (obj.TvmAngleInDegree >= 45 || obj.TvmAngleInDegree == -45 ))% Table II
                id = 'SaivDr:IllegalArgumentException';
                msg = 'Unsupported combination of PHI and ORD';
                me = MException(id, msg);
                throw(me);
            end            
            
        end
        
        function [valueC, valueA]= checkTvm(obj)
            valueC = obj.ckTvm_;
            valueA = obj.TvmAngleInDegree;
            if ~(obj.lenx3<=2)
                disp('Warnning: ||x3|| > 2.');
            end
            if ~(obj.lambdaueq<=0)
                disp('Warnning: lambda is violated.');
            end
        end
        
        %{
            if obj.decX ~= obj.decY
                id = 'SaivDr:IllegalArgumentException';
                msg = 'My and Mx must be the same as each other';
                me = MException(id, msg);
                throw(me);
            end
                if nargin < 6
                    obj = setSign4Lambda(obj,1);
                else
                    obj = setSign4Lambda(obj,varargin{6});
                end
            end
            obj = update_(obj);
            obj.isInitialized = true;
        end

        function obj = setSign4Lambda(obj,DirectionOfTriangle)
            obj.DirectionOfTriangle = DirectionOfTriangle;
            if obj.isInitialized
                obj = update_(obj);
            end
        end
        
        %}
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeISystem(obj);
            s.lenx3 = obj.lenx3;
            s.lambdaueq = obj.lambdaueq;
            s.x3_ = obj.x3_;
            s.ckTvm_ = obj.ckTvm_;
            s.omgs_ = matlab.System.saveObject(obj.omgs_);
            s.ops_  = matlab.System.saveObject(obj.ops_);            
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeISystem(obj,s,wasLocked);
            obj.lenx3 = s.lenx3;
            obj.lambdaueq = s.lambdaueq;            
            obj.x3_ = s.x3_;
            obj.ckTvm_ = s.ckTvm_;
            obj.omgs_ = matlab.System.loadObject(s.omgs_);
            obj.ops_  = matlab.System.loadObject(s.ops_);
        end 
        
        function updateParameterMatrixSet_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.utility.Direction
            
            nChs = obj.NumberOfChannels;
            nStages = sum(obj.PolyPhaseOrder) + 1;
            angles  = obj.Angles;
            mus     = obj.Mus;
            omgs = obj.omgs_;
            pmMtxSet = obj.ParameterMatrixSet;
            
            % VM1 condition
            angles(1:nChs(ChannelGroup.LOWER)-1,1) = ...
                zeros(nChs(ChannelGroup.LOWER)-1,1);
            mus(1,1) = 1;
            
            % W0, Unx, Uny
            for iParamMtx=uint32(1):nStages+1
                U = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
                step(pmMtxSet,U,iParamMtx);
            end
            
            % TVM condition
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);            
            if ordX >= 2 && obj.TvmAngleInDegree <= 45 % Table I
                updateParameterMatricesTvmHorizontal_(obj);
            elseif ordY >= 2 && (obj.TvmAngleInDegree >= 45 || ...
                    obj.TvmAngleInDegree == -45 )% Table II
                updateParameterMatricesTvmVertical_(obj);
            else
                error('Invaid combination of PHI and ORD');
            end
            
        end
        
        function updateParameterMatricesTvmHorizontal_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.utility.Direction

            nChs = obj.NumberOfChannels;
            angles  = obj.Angles;
            mus     = obj.Mus;            
            omgs = obj.omgs_;
            pmMtxSet = obj.ParameterMatrixSet;
                        
            % Calculate \bar{x}_3 and the length
            obj.x3_ = getBarX3Horizontal_(obj);
            obj.lenx3 = norm( obj.x3_ );
            
            % Calculate lambda
            lambda1 = real(acos(1-(obj.lenx3^2)/2));
            lambda2 = real(acos(obj.lenx3/2));
            lambda = lambda1 + lambda2;
            
            % Calculate U_{Nx-2} and U_{Nx-1}
            L = nChs(ChannelGroup.LOWER);
            a = zeros(L,1);
            a(1) = 1;

            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            iParamMtx = uint32(ordX); % Nx-2
            %obj.lambdaueq = abs(cos(lambda))...
            %    -abs(prod(cos(obj.angles(2:L-1,iParamMtx)))); % <= 0
            cc = cos(lambda)/(prod(cos(angles(2:L-1,iParamMtx))));
            obj.lambdaueq = abs(cc) -1; % <= 0
            %
            %angles(1,iParamMtx) = obj.DirectionOfTriangle*real(acos(cos(lambda)/...
            %    prod(cos(angles(2:L-1,iParamMtx)))));
            angles(1,iParamMtx) = obj.DirectionOfTriangle*real(acos(cc));
            mus(1,iParamMtx) = 1;
            %
            A = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
            angle_ = step(obj.ops_, obj.x3_);
            P = step(omgs,angle_,1);
            Vb = A*P;
            step(pmMtxSet,Vb,iParamMtx);
            iParamMtx = iParamMtx + 1;
            
            % U_{Nx-1}
            % to be revised
            angles(1:L-1,iParamMtx) = zeros(L-1,1);
            mus(1,iParamMtx) = 1;
            angle_ = step(obj.ops_,-a-Vb*obj.x3_);
            angles(:,iParamMtx) = angles(:,iParamMtx) + angle_;
            Va = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
            step(pmMtxSet,Va,iParamMtx);
            iParamMtx = iParamMtx + 1;
            
            % U_{Nx} = \bar{U} * U_{Nx-2}.' * U_{Nx-1}.'
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            if ordY > 0
                U = step(obj.ParameterMatrixSet,[],iParamMtx) * ...
                    step(obj.ParameterMatrixSet,[],iParamMtx-2).' * ...
                    step(obj.ParameterMatrixSet,[],iParamMtx-1).';
                step(obj.ParameterMatrixSet,U,iParamMtx);
            end
            
            % Check TVM
            vx1 = a;
            vx2 = Va*a;
            vx3 = Va*Vb*obj.x3_;
            obj.ckTvm_ = norm( vx1 + vx2 + vx3 )/sqrt(L);
        end
        
        function value = getBarX3Horizontal_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.utility.Direction
            
            nChs = obj.NumberOfChannels;
            nStages = sum(obj.PolyPhaseOrder)+1;
            I = getMatrixI_(obj);
            Z = getMatrixZ_(obj);
            E0 = obj.matrixE0;
            %
            L = nChs(ChannelGroup.LOWER);
            M = 2*L;
            My = obj.DecimationFactor(Direction.VERTICAL);
            phi = pi * obj.TvmAngleInDegree /180.0;
            %
            dy = -mod(0:M-1,My).';
            dx = -floor((0:M-1)/My).';
            by = 2/sqrt(M)*[Z I]*E0*dy;
            bx = 2/sqrt(M)*[Z I]*E0*dx;
            bphix = (1/sqrt(M))*(tan(phi)*by + bx);
            a = zeros(L,1);
            a(1) = 1;
            
            % u0
            u0 = bphix;
            iParamMtx = uint32(2); % n=0
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            for iFree=1:ordX-2
                U = step(obj.ParameterMatrixSet,[],iParamMtx);
                iParamMtx = iParamMtx + 1;
                u0 = U * u0;
            end
            
            % u1
            u1 = zeros(L,1);
            if ordX > 2
                u1 = a;
                iParamMtx = uint32(3); % nx=1
                for iFree=1:ordX-3
                    U = step(obj.ParameterMatrixSet,[],iParamMtx);
                    u1 = U * u1 + a;
                    iParamMtx = iParamMtx + 1;
                end
            end
            
            % u2
            u2 = zeros(L,1);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            if ordY > 0
                v = a;
                iParamMtx = uint32(nStages); % ny=Ny-1
                for iFree=2:ordY
                    U = step(obj.ParameterMatrixSet,[],iParamMtx);
                    v = U.' * v + a;
                    iParamMtx = iParamMtx - 1;
                end
                iParamMtx = uint32(ordX)+2; % nx=Nx as \bar{U}
                U = step(obj.ParameterMatrixSet,[],iParamMtx);
                u2 = tan(phi) * U.' * v;
            end
            
            % \bar{x}_3
            value = u0 + u1 + u2;
        end
        
        function updateParameterMatricesTvmVertical_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.utility.Direction
            
            nChs = obj.NumberOfChannels;
            nStages = sum(obj.PolyPhaseOrder)+1;
            angles  = obj.Angles;
            mus     = obj.Mus;            
            omgs    = obj.omgs_;
            pmMtxSet = obj.ParameterMatrixSet;
            
            % Calculate \bar{x}_3 and the length
            obj.x3_ = getBarX3Vertical_(obj);
            obj.lenx3 = norm( obj.x3_ );
            
            % Calculate lambda
            lambda1 = real(acos(1-(obj.lenx3^2)/2));
            lambda2 = real(acos(obj.lenx3/2));
            lambda = lambda1 + lambda2;
            
            % Update U_{N-2} and U_{N-1}
            L = nChs(ChannelGroup.LOWER);
            a = zeros(L,1);
            a(1) = 1;
            
            iParamMtx = uint32(nStages-1); % U_{Ny-2}
            %obj.lambdaueq = abs(cos(lambda))...
            %    -abs(prod(cos(obj.angles(2:L-1,iParamMtx)))); % <= 0
            cc = cos(lambda)/(prod(cos(angles(2:L-1,iParamMtx))));
            obj.lambdaueq = abs(cc) -1; % <= 0
            %
            %angles(1,iParamMtx) = obj.DirectionOfTriangle*real(acos(cos(lambda)/...
            %    prod(cos(angles(2:L-1,iParamMtx)))));
            angles(1,iParamMtx) = obj.DirectionOfTriangle*real(acos(cc));
            mus(1,iParamMtx) = 1;
            %
            A = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
            angle_ = step(obj.ops_,obj.x3_);
            P = step(omgs,angle_,1);
            Vb = A*P;
            step(pmMtxSet,Vb,iParamMtx);
            iParamMtx = iParamMtx + 1;
            
            % Uy_{Ny-1}
            % to be revised
            angles(1:L-1,iParamMtx) = zeros(L-1,1);
            mus(1,iParamMtx) = 1;
            angle_ = step(obj.ops_,-a-Vb*obj.x3_);
            angles(:,iParamMtx) = angles(:,iParamMtx) + angle_;
            Va = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
            step(pmMtxSet,Va,iParamMtx);
            
            % Check TVM
            vx1 = a;
            vx2 = Va*a;
            vx3 = Va*Vb*obj.x3_;
            obj.ckTvm_ = norm( vx1 + vx2 + vx3 )/sqrt(L);
        end
        
        function value = getBarX3Vertical_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.utility.Direction
            
            nChs = obj.NumberOfChannels;
            I = getMatrixI_(obj);
            Z = getMatrixZ_(obj);
            E0 = obj.matrixE0;
            %
            L = nChs(ChannelGroup.LOWER);
            M = 2*L;
            My = obj.DecimationFactor(Direction.VERTICAL);
            phi = pi * obj.TvmAngleInDegree /180.0;
            %
            dy = -mod(0:M-1,My).';
            dx = -floor((0:M-1)/My).';
            by = 2/sqrt(M)*[Z I]*E0*dy;
            bx = 2/sqrt(M)*[Z I]*E0*dx;
            bphiy = (1/sqrt(M))*(by + bx*cot(phi));
            a = zeros(L,1);
            a(1) = 1;
            
            % u0
            u0 = bphiy;
            iParamMtx = uint32(2); % U0
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            for iFree=1:ordX+ordY-2
                U = step(obj.ParameterMatrixSet,[],iParamMtx);
                u0 = U * u0;
                iParamMtx = iParamMtx + 1;
            end
            
            % u1
            u1 = zeros(L,1);
            if ordY > 2
                u1 = a;
                iParamMtx = uint32(ordX)+3; % ny=1
                for iFree=1:ordY-3
                    U = step(obj.ParameterMatrixSet,[],iParamMtx);
                    u1 = U * u1 + a;
                    iParamMtx = iParamMtx + 1;
                end
            end
            
            % u2
            u2 = zeros(L,1);
            if ordX > 0
                v = a;
                iParamMtx = uint32(3); %nx=1
                for iFree=1:ordX-1
                    U = step(obj.ParameterMatrixSet,[],iParamMtx);
                    v = U * v + a;
                    iParamMtx = iParamMtx + 1;
                end
                u2 = v;
                iParamMtx = uint32(ordX)+2; % nx=Nx
                for iFree=1:ordY-2
                    U = step(obj.ParameterMatrixSet,[],iParamMtx);
                    u2 = U * u2;
                    iParamMtx = iParamMtx + 1;
                end
                u2 = cot(phi) * u2;
            end
            
            % \bar{x}_3
            value = u0 + u1 + u2;
            
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = [2 2];
        end
    end
    
    methods (Access = private)
        function value = getMatrixZ_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            value = zeros(obj.NumberOfChannels(ChannelGroup.LOWER));
        end
    end                
    
end
