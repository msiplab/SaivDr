classdef CplxOvsdLpPuFb3dTypeIVm0System < saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dTypeISystem %#codegen
    %OVSDLPPUFBMDTYPEIVM0SYSTEM 3-D Type-I Oversapled LPPUFB without VM
    %
    % SVN identifier:
    % $Id: OvsdLpPuFb3dTypeIVm0System.m 683 2015-05-29 08:22:13Z sho $
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
        function obj = CplxOvsdLpPuFb3dTypeIVm0System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dTypeISystem(...
                varargin{:});
            obj.initOmgs_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            obj.propOmgs_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dTypeISystem(obj);
            s.initOmgs_ = matlab.System.saveObject(obj.initOmgs_);
            s.propOmgs_ = matlab.System.saveObject(obj.propOmgs_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.initOmgs_ = matlab.System.loadObject(s.initOmgs_);
            obj.propOmgs_ = matlab.System.loadObject(s.propOmgs_);
            loadObjectImpl@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dTypeISystem(obj,s,wasLocked);
        end
        
        function obj = updateParameterMatrixSet_(obj)
            nCh = obj.NumberOfChannels;
            hCh = nCh/2;
            
            pmMtxSet = obj.ParameterMatrixSet;
            
            [initAngles, propAngles] = splitAngles_(obj);
            
            % V0
            mtx = step(obj.initOmgs_,initAngles,obj.Mus(1:nCh));
            step(pmMtxSet,mtx,uint32(1));
            
            angles = reshape(propAngles,[],obj.nStages-1);
            mus    = reshape(obj.Mus(nCh+1:end),[],obj.nStages-1);
            
            nParamMtxAngs = nCh*(nCh-2)/8;
            for iParamMtx = uint32(1):obj.nStages-1
                % W
                mtx = step(obj.propOmgs_,angles(1:nParamMtxAngs,iParamMtx),mus(1:hCh,iParamMtx));
                step(pmMtxSet,mtx,3*iParamMtx-1);
                
                % U
                mtx = step(obj.propOmgs_,angles(nParamMtxAngs+1:2*nParamMtxAngs,iParamMtx),mus(hCh+1:end,iParamMtx));
                step(pmMtxSet,mtx,3*iParamMtx);
                
                % angsB
                step(pmMtxSet,angles(2*nParamMtxAngs+1:end,iParamMtx),3*iParamMtx+1);
            end
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = [ 0 0 0 ];
        end
    end
    
end
