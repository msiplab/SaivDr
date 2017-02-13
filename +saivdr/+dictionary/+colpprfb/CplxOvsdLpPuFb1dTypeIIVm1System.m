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
        omgsV_
        omgsWU_
        omgsHWHU_
        omfs_
    end
      
    methods        
        function obj = CplxOvsdLpPuFb1dTypeIIVm1System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixFactorizationSystem
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeIISystem(...
                varargin{:});
            obj.omfs_  = OrthonormalMatrixFactorizationSystem('OrderOfProduction','Ascending');
            obj.omgsV_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            obj.omgsWU_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            obj.omgsHWHU_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
        end
    end
    
    methods (Access = protected)
            
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeIISystem(obj);
            s.omfs_  = matlab.System.saveObject(obj.omfs_);
            s.omgsV_ = matlab.System.saveObject(obj.omgsV_);
            s.omgsWU_ = matlab.System.saveObject(obj.omgsWU_);
            s.omgsHWHU_ = matlab.System.saveObject(obj.omgsHWHU_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeIISystem(obj,s,wasLocked);
            obj.omfs_ = matlab.System.loadObject(s.omfs_);
            obj.omgsV_ = matlab.System.loadObject(s.omgsV_);
            obj.omgsWU_ = matlab.System.loadObject(s.omgsWU_);
            obj.omgsHWHU_ = matlab.System.loadObject(s.omgsHWHU_);
        end
        
        function updateParameterMatrixSet_(obj)
            import saivdr.dictionary.cnsoltx.ChannelGroup
            nCh = obj.NumberOfChannels;
            
            [initAngles, propAngles] = splitAngles_(obj);
            
            pmMtxSt_ = obj.ParameterMatrixSet;
            
            mtx = step(obj.omgsV_,initAngles,obj.Mus(1:nCh));
            step(pmMtxSt_,mtx,uint32(1));
            
            angles = reshape(propAngles,[],obj.nStages-1);
            mus    = reshape(obj.Mus(nCh+1:end),[],obj.nStages-1);            
            nAngsW = floor(nCh/2)*(floor(nCh/2)-1)/2;
            nAngsHW = ceil(nCh/2)*(ceil(nCh/2)-1)/2;
            nAngsB = floor(nCh/4);
            nMusW = floor(nCh/2);
            nMusHW = ceil(nCh/2);
            omgsWU = obj.omgsWU_;
            omgsHWHU = obj.omgsHWHU_;
            
            V_ = eye(nCh);
            
            for iParamMtx = uint32(1):obj.nStages-1
                % W
                mtx = step(omgsWU,angles(1:nAngsW,iParamMtx),...
                    mus(1:nMusW,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx-4);
                V_(1:floor(nCh/2),:) = mtx*V_(1:floor(nCh/2),:);
                % U
                mtx = step(omgsWU,angles(nAngsW+1:2*nAngsW,iParamMtx),...
                        mus(nMusW+1:2*nMusW,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx-3);
                V_(floor(nCh/2)+1:end-1,:) = mtx*V_(floor(nCh/2)+1:end-1,:);
                
                % angsB1
                step(pmMtxSt_,angles(2*nAngsW+1:2*nAngsW+nAngsB,iParamMtx),6*iParamMtx-2);
                
                % HU
                mtx = step(omgsHWHU,angles(2*nAngsW+nAngsB+nAngsHW+1:2*nAngsW+nAngsB+2*nAngsHW,iParamMtx),...
                        mus(2*nMusW+nMusHW+1:end,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx);
                V_(ceil(nCh/2):end,:) = mtx*V_(ceil(nCh/2):end,:);
                
                % HW
                mtx = step(omgsHWHU,angles(2*nAngsW+nAngsB+1:2*nAngsW+nAngsB+nAngsHW,iParamMtx),...
                    mus(2*nMusW+1:2*nMusW+nMusHW,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx-1);
                V_(1:ceil(nCh/2),:) = mtx*V_(1:ceil(nCh/2),:);

                % angsB2
                step(pmMtxSt_,angles(2*nAngsW+1:2*nAngsW+nAngsB,iParamMtx),6*iParamMtx+1);
            end
            
            % Initial matrix with No-DC-leakage condition
            [angles_,~] = step(obj.omfs_,V_.');
            initAngles(1:nCh-1) = angles_(1:nCh-1);
            mtx = step(obj.omgsV_,initAngles,obj.Mus(1:nCh));
            step(pmMtxSt_,mtx,uint32(1));
            
            angles = [initAngles angles(:).'];
            
            obj.Angles = angles;
            obj.Mus(nCh+1:end)    = mus(:).';
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = 0;
        end
    end
end
