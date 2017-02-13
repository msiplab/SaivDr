classdef CplxOvsdLpPuFb1dTypeIVm1System < ...
        saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeISystem %#codegen
    %OVSDLPPUFBMDTYPEIVM1SYSTEM 2-D Type-I Oversampled LPPUFB with one VM
    %
    % SVN identifier:
    % $Id: CplxOvsdLpPuFb1dTypeIVm1System.m 653 2015-02-04 05:21:08Z sho $
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2014, Shogo MURAMATSU
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
        propOmfs_
    end
    
    methods
        function obj = CplxOvsdLpPuFb1dTypeIVm1System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            import saivdr.dictionary.utility.OrthonormalMatrixFactorizationSystem
            obj = obj@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeISystem(...
                varargin{:});
            obj.initOmgs_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            obj.propOmgs_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            obj.propOmfs_ = OrthonormalMatrixFactorizationSystem('OrderOfProduction','Ascending');
        end
    end
    
    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeISystem(obj);
            s.initOmgs_ = matlab.System.saveObject(obj.initOmgs_);
            s.propOmgs_ = matlab.System.saveObject(obj.propOmgs_);
            s.propOmfs_ = matlab.System.saveObject(obj.propOmfs_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeISystem(obj,s,wasLocked);
            obj.initOmgs_ = matlab.System.loadObject(s.initOmgs_);
            obj.propOmgs_ = matlab.System.loadObject(s.propOmgs_);
            obj.propOmfs_ = matlab.System.loadObject(s.propOmfs_);
        end        
        
        function updateParameterMatrixSet_(obj)
            nCh = obj.NumberOfChannels;
            hCh = nCh/2;

            [initAngles, propAngles] = splitAngles_(obj);
            
            angles = reshape(propAngles,[],obj.nStages-1);
            mus    = reshape(obj.Mus(nCh+1:end),[],obj.nStages-1);
            
            nParamMtxAngs = nCh*(nCh-2)/8;

            W_ = eye(hCh);
            
            pmMtxSet = obj.ParameterMatrixSet;
            for iParamMtx = uint32(1):obj.nStages-1
                % W
                mtx = step(obj.propOmgs_,angles(1:nParamMtxAngs,iParamMtx),mus(1:hCh,iParamMtx));
                step(pmMtxSet,mtx,3*iParamMtx-1);
                W_ = mtx*W_;
                
                % U
                mtx = step(obj.propOmgs_,angles(nParamMtxAngs+1:2*nParamMtxAngs,iParamMtx),mus(hCh+1:end,iParamMtx));
                step(pmMtxSet,mtx,3*iParamMtx);
                
                % angsB
                step(pmMtxSet,angles(2*nParamMtxAngs+1:end,iParamMtx),3*iParamMtx+1);
            end
            
            % Initial matrix V0 with No-DC-leakage condition
            [angles_,~] = step(obj.propOmfs_,W_.');
            initAngles(1:hCh-1) = angles_(1:hCh-1);
            initAngles(hCh:nCh-1) = zeros(hCh,1);
            mtx = step(obj.initOmgs_,initAngles,obj.Mus(1:nCh));
            step(pmMtxSet,mtx,uint32(1));
            
            obj.Angles = [initAngles ; angles(:)];
            obj.Mus(nCh+1:end) = mus(:);
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = 0;
        end
    end
    
end
