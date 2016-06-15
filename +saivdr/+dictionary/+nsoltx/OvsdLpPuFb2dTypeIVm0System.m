classdef OvsdLpPuFb2dTypeIVm0System < saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem %#codegen
    %OVSDLPPUFBMDTYPEIVM0SYSTEM 2-D Type-I Oversapled LPPUFB without VM
    %
    % SVN identifier:
    % $Id: OvsdLpPuFb2dTypeIVm0System.m 683 2015-05-29 08:22:13Z sho $
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
    
    properties (Access = private)
        initOmgs_
        propOmgs_
    end
    
    methods
        function obj = OvsdLpPuFb2dTypeIVm0System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem(...
                varargin{:});
            obj.initOmgs_ = OrthonormalMatrixGenerationSystem();
            obj.propOmgs_ = OrthonormalMatrixGenerationSystem();
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem(obj);
            s.initOmgs_ = matlab.System.saveObject(obj.initOmgs_);
            s.propOmgs_ = matlab.System.saveObject(obj.propOmgs_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.initOmgs_ = matlab.System.loadObject(s.initOmgs_);
            obj.propOmgs_ = matlab.System.loadObject(s.propOmgs_);
            loadObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem(obj,s,wasLocked);
        end
        
        function obj = updateParameterMatrixSet_(obj)
            nch = sum(obj.NumberOfChannels);
            
            % initial matrix
            nInitMtxAngs = nch*(nch-1)/2;
            mtx = step(obj.initOmgs_,obj.Angles(1:nInitMtxAngs),obj.Mus(:,1));
            step(obj.ParameterMatrixSet,mtx,uint32(1));
            
            angles = reshape(obj.Angles(nInitMtxAngs+1:end),[],obj.nStages-1);
            mus    = obj.Mus(:,2:end);
            
            nParamMtxAngs = nch*(nch-2)/8;
            for iParamMtx = uint32(1):obj.nStages-1
                % W
                mtx = step(obj.propOmgs_,angles(1:nParamMtxAngs,iParamMtx),mus(1:nch/2,iParamMtx));
                step(obj.ParameterMatrixSet,mtx,3*iParamMtx-1);
                
                % U
                mtx = step(obj.propOmgs_,angles(nParamMtxAngs+1:2*nParamMtxAngs,iParamMtx),mus(nch/2+1:end,iParamMtx));
                step(obj.ParameterMatrixSet,mtx,3*iParamMtx);
                
                % angsB
                step(obj.ParameterMatrixSet,angles(2*nParamMtxAngs+1:end,iParamMtx),3*iParamMtx+1);
            end
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = [ 0 0 ];
        end
    end
    
end
