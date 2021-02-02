classdef LpPuFb2dVm2System < saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem
    %LPPUFB2DVM2 2-D LPPUFB with 2-order classical vanishing moments
    %
    % Reference:
    %   Tomoya Kobayashi, Shogo Muramatsu and Hisakazu Kikuchi, 
    %   ''Two-Degree Vanishing Moments on 2-D Non-separable GenLOT,''
    %   IEEE Proc. of ISPACS2009, pp.248-251, Dec. 2009.
    % 
    % SVN identifier:
    % $Id: LpPuFb2dVm2System.m 683 2015-05-29 08:22:13Z sho $
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
    properties
        DirectionOfTriangleX = -1
        DirectionOfTriangleY =  1
    end
    
    properties (SetAccess = private, GetAccess = public)
        lenx3x
        lenx3y
        lambdaxueq
        lambdayueq    
    end
    
    properties (Access = private)
        omgs_
        ops_
        x3x_
        x3y_
        ckVm2_
    end
    
    methods
        
        function obj = LpPuFb2dVm2System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            import saivdr.dictionary.utility.OrthogonalProjectionSystem
            obj = obj@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem(...
                varargin{:});
            obj.omgs_ = OrthonormalMatrixGenerationSystem();
            obj.ops_  = OrthogonalProjectionSystem();
            
            import saivdr.dictionary.utility.Direction
            if obj.PolyPhaseOrder(Direction.HORIZONTAL) < 2 || ...
                    obj.PolyPhaseOrder(Direction.HORIZONTAL) < 2 
                    id = 'SaivDr:IllegalArgumentException';
                    msg = 'Order must be greater than or equal to [ 2 2 ]';
                    me = MException(id, msg);
                    throw(me);                
            end
        end
        
        function value = checkVm2(obj)
            value = obj.ckVm2_;
            if ~(obj.lenx3x<=2)
                disp('Warnning: ||x3x|| > 2.');
            end
            if ~(obj.lenx3y<=2)
                disp('Warnning: ||x3y|| > 2.');
            end
            if ~(obj.lambdaxueq<=0)
                disp('Warnning: lambdax is violated.');
            end
            if ~(obj.lambdayueq<=0)
                disp('Warnning: lambday is violated.');
            end
        end
        
    end
    
    methods (Access = protected)
    
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem(obj);
            s.omgs_ = matlab.System.saveObject(obj.omgs_);
            s.ops_  = matlab.System.saveObject(obj.ops_);            
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem(obj,s,wasLocked);
            obj.omgs_ = matlab.System.loadObject(s.omgs_);
            obj.ops_  = matlab.System.loadObject(s.ops_);
            %
            obj.omgs_.release()
        end 
        
        function updateParameterMatrixSet_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.utility.Direction

            nChs = obj.NumberOfChannels;
            angles = obj.Angles;
            mus    = obj.Mus;
            
            % VM1 condition
            angles(1:nChs(ChannelGroup.LOWER)-1,1) = ...
                zeros(nChs(ChannelGroup.LOWER)-1,1);
            mus(1,1) = 1;
            
            I = getMatrixI_(obj);
            Z = getMatrixZ_(obj);
            E0 = obj.matrixE0;
            
            L = nChs(ChannelGroup.LOWER);
            M = 2*L;
            My = obj.DecimationFactor(Direction.VERTICAL);
            Mx = obj.DecimationFactor(Direction.HORIZONTAL);
            
            dy = -mod(0:M-1,My).';
            dx = -floor((0:M-1)/My).';
            by = 2/(My*sqrt(M))*[Z I]*E0*dy;
            bx = 2/(Mx*sqrt(M))*[Z I]*E0*dx;
            a = zeros(L,1);
            a(1) = 1;
            cx = I;
            Uc = I;
            vx = bx;
            vy = by;
            
            omgs = obj.omgs_;
            pmMtxSet = obj.ParameterMatrixSet;
            % W
            W = step(omgs,angles(:,1),mus(:,1));
            step(pmMtxSet,W,uint32(1));
            iParamMtx = uint32(2);
            
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            nFreeX = ordX-2;
            for iFree=1:nFreeX
                U = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
                step(pmMtxSet,U,iParamMtx);
                iParamMtx = iParamMtx + 1;
                Uc = U * Uc;
                cx = [ I U * cx ];
                vx = [ a ; vx ]; %#ok<AGROW>
            end
            
            % Ux(Nx-2)
            obj.x3x_ =  cx * vx;
            obj.lenx3x = norm( obj.x3x_ );
            lambdax1 = 2*real(asin(obj.lenx3x/2));
            lambdax2 = real(acos(obj.lenx3x/2));
            lambdax = lambdax1 + lambdax2;
            
            % to be revised
            %lambdaxueq = abs(cos(lambdax))...
            %    - abs(prod(cos(obj.angles(2:L-1,iParamMtx)))); % <= 0
            cc = cos(lambdax)/prod(cos(angles(2:L-1,iParamMtx)));
            obj.lambdaxueq = abs(cc) - 1; % <= 0
            %
            %angles(1,iParamMtx) = obj.DirectionOfTriangleX*real(acos(cos(lambdax)/...
            %    prod(cos(angles(2:L-1,iParamMtx)))));
            angles(1,iParamMtx) = obj.DirectionOfTriangleX*real(acos(cc));
            mus(1,iParamMtx) = 1;
            %
            Ax = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
            angle_ = step(obj.ops_,obj.x3x_);
            Px = step(obj.omgs_,angle_,1);
            U = Ax*Px;
            step(pmMtxSet,U,iParamMtx);
            iParamMtx = iParamMtx + 1;
            Uc = U * Uc;
            cx = [ I U * cx ];
            vx = [ a ; vx ];
            
            % Ux(Nx-1)
            % to be revised
            angles(1:L-1,iParamMtx) = zeros(L-1,1);
            mus(1,iParamMtx) = 1;
            angle_ = step(obj.ops_,-a-U*obj.x3x_);
            angles(:,iParamMtx) = angles(:,iParamMtx) + angle_;
            U = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
            step(pmMtxSet,U,iParamMtx);
            iParamMtx = iParamMtx + 1;
            Uc = U * Uc;
            cx = [ I U * cx ];
            cy = Uc;
            vx = [ a ; vx ];
            
            % UxNx ... Uy(Ny-3)
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            nFreeY = ordY-2;
            for iFree=1:nFreeY
                U = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
                step(pmMtxSet,U,iParamMtx);
                iParamMtx = iParamMtx + 1;
                cy = [ I U * cy ];
                vy = [ a ; vy ]; %#ok<AGROW>
            end
            
            % Uy(Ny-2) or UxNx
            obj.x3y_ = cy * vy ;
            obj.lenx3y = norm( obj.x3y_ );
            lambday1 = 2*real(asin(obj.lenx3y/2));
            lambday2 = real(acos(obj.lenx3y/2));
            lambday = lambday1 + lambday2;
            
            % to be revised
            %lambdayueq = abs(cos(lambday))...
            %      -abs(prod(cos(angles(2:L-1,iParamMtx)))); % <= 0
            cc = cos(lambday)/prod(cos(angles(2:L-1,iParamMtx)));
            obj.lambdayueq =  abs(cc) - 1; % <= 0
            %angles(1,iParamMtx) = obj.DirectionOfTriangleY*real(acos(cos(lambday)/...
            %    prod(cos(angles(2:L-1,iParamMtx)))));
            angles(1,iParamMtx) = obj.DirectionOfTriangleY*real(acos(cc));
            mus(1,iParamMtx) = 1;
            %
            Ay = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
            angle_ = step(obj.ops_,obj.x3y_);
            Py = step(obj.omgs_,angle_,1);
            U = Ay*Py;
            step(pmMtxSet,U,iParamMtx);
            iParamMtx = iParamMtx + 1;
            cy = [ I U * cy ];
            vy = [ a ; vy ];
            
            % Uy(Ny-1)
            % to be revised
            angles(1:L-1,iParamMtx) = zeros(L-1,1);
            mus(1,iParamMtx) = 1;
            angle_ = step(obj.ops_,-a-U*obj.x3y_);
            angles(:,iParamMtx) = angles(:,iParamMtx) + angle_;
            U = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
            step(pmMtxSet,U,iParamMtx);
            iParamMtx = iParamMtx + 1;
            cy = [ I U * cy ];
            vy = [ a ; vy ];
            
            % UyNy
            U = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
            step(pmMtxSet,U,iParamMtx);
            
            %
            obj.ckVm2_(1) = norm( cy * vy )/sqrt(L);
            obj.ckVm2_(2) = norm( cx * vx )/sqrt(L);
                    
        end
                        
    end

    methods (Access = protected, Static = true)
    
        function value = getDefaultPolyPhaseOrder_()
            value = [2 2];
        end
        
    end
    
    methods (Access = private)
        function value = getMatrixZ_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            value = zeros(obj.NumberOfChannels(ChannelGroup.LOWER));
        end
        
        function value = getMatrixI_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            value = eye(ceil(obj.NumberOfChannels(ChannelGroup.LOWER)));
        end
        
    end
    
end
