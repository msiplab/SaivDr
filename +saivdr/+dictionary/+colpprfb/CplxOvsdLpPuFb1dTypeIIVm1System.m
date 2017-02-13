classdef CplxOvsdLpPuFb1dTypeIIVm1System < ...
        saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeIISystem %#codegen
    %OVSDLPPUFB1dTYPEIIVM1SYSTEM 2-D Type-II Oversampled LPPUFB with one VM
    %
    % SVN identifier:
    % $Id: CplxOvsdLpPuFb1dTypeIIVm1System.m 653 2015-02-04 05:21:08Z sho $
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
   
    properties (Access = private, Nontunable)
        initOmgs_
        propOmgs1st_
        propOmgs2nd_
        omfs_
    end
      
    methods        
        function obj = CplxOvsdLpPuFb1dTypeIIVm1System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixFactorizationSystem
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeIISystem(...
                varargin{:});
            obj.omfs_  = OrthonormalMatrixFactorizationSystem('OrderOfProduction','Ascending');
            obj.initOmgs_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            obj.propOmgs1st_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            obj.propOmgs2nd_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
        end
    end
    
    methods (Access = protected)
            
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeIISystem(obj);
            s.omfs_  = matlab.System.saveObject(obj.omfs_);
            s.initOmgs_ = matlab.System.saveObject(obj.initOmgs_);
            s.propOmgs1st_ = matlab.System.saveObject(obj.propOmgs1st_);
            s.propOmgs2nd_ = matlab.System.saveObject(obj.propOmgs2nd_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeIISystem(obj,s,wasLocked);
            obj.omfs_ = matlab.System.loadObject(s.omfs_);
            obj.initOmgs_ = matlab.System.loadObject(s.initOmgs_);
            obj.propOmgs1st_ = matlab.System.loadObject(s.propOmgs1st_);
            obj.propOmgs2nd_ = matlab.System.loadObject(s.propOmgs2nd_);
        end
        
        function updateParameterMatrixSet_(obj)
            nCh = obj.NumberOfChannels;
            hCh = floor(nCh/2);
            pmMtxSt_ = obj.ParameterMatrixSet;
            
            [initAngles, propAngles] = splitAngles_(obj);
            
            angles = reshape(propAngles,[],obj.nStages-1);
            mus    = reshape(obj.Mus(nCh+1:end),[],obj.nStages-1);            
            nAngs1st = hCh*(hCh-1)/2;
            nAngs2nd = (hCh+1)*hCh/2;
            nAngsB = floor(nCh/4);
            nMus1st = hCh;
            nMus2nd = hCh+1;
            
            V_ = eye(nCh);
            
            for iParamMtx = uint32(1):obj.nStages-1
                % W
                mtx = step(obj.propOmgs1st_,angles(1:nAngs1st,iParamMtx),...
                    mus(1:nMus1st,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx-4);
                V_(1:hCh,:) = mtx*V_(1:hCh,:);
                
                % U
                mtx = step(obj.propOmgs1st_,angles(nAngs1st+1:2*nAngs1st,iParamMtx),...
                        mus(nMus1st+1:2*nMus1st,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx-3);
                V_(hCh+1:end-1,:) = mtx*V_(hCh+1:end-1,:);
                
                % angsB1
                step(pmMtxSt_,angles(2*nAngs1st+1:2*nAngs1st+nAngsB,iParamMtx),6*iParamMtx-2);
                
                % HU
                mtx = step(obj.propOmgs2nd_,angles(2*nAngs1st+nAngsB+nAngs2nd+1:2*nAngs1st+nAngsB+2*nAngs2nd,iParamMtx),...
                        mus(2*nMus1st+nMus2nd+1:end,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx);
                V_(hCh+1:end,:) = mtx*V_(hCh+1:end,:);
                
                % HW
                mtx = step(obj.propOmgs2nd_,angles(2*nAngs1st+nAngsB+1:2*nAngs1st+nAngsB+nAngs2nd,iParamMtx),...
                    mus(2*nMus1st+1:2*nMus1st+nMus2nd,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx-1);
                V_(1:hCh+1,:) = mtx*V_(1:hCh+1,:);

                % angsB2
                step(pmMtxSt_,angles(2*nAngs1st+1:2*nAngs1st+nAngsB,iParamMtx),6*iParamMtx+1);
            end
            
            % Initial matrix V0 with No-DC-leakage condition
            [angles_,~] = step(obj.omfs_,V_.');
            initAngles(1:nCh-1) = angles_(1:nCh-1);
            mtx = step(obj.initOmgs_,initAngles,obj.Mus(1:nCh));
            step(pmMtxSt_,mtx,uint32(1));
            
            obj.Angles = [initAngles ; angles(:)];
            obj.Mus(nCh+1:end)    = mus(:);
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = 0;
        end
    end
end
