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
        omgsV0_
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
            obj.omfs_  = OrthonormalMatrixFactorizationSystem();
            obj.omgsV0_ = OrthonormalMatrixGenerationSystem();
            obj.omgsWU_ = OrthonormalMatrixGenerationSystem();
            obj.omgsHWHU_ = OrthonormalMatrixGenerationSystem();
        end
    end
    
    methods (Access = protected)
            
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeIISystem(obj);
            s.omfs_  = matlab.System.saveObject(obj.omfs_);
            s.omgsV0_ = matlab.System.saveObject(obj.omgsV0_);
            s.omgsWU_ = matlab.System.saveObject(obj.omgsWU_);
            s.omgsHWHU_ = matlab.System.saveObject(obj.omgsHWHU_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeIISystem(obj,s,wasLocked);
            obj.omfs_ = matlab.System.loadObject(s.omfs_);
            obj.omgsV0_ = matlab.System.loadObject(s.omgsV0_);
            obj.omgsWU_ = matlab.System.loadObject(s.omgsWU_);
            obj.omgsHWHU_ = matlab.System.loadObject(s.omgsHWHU_);
        end
        
        function updateParameterMatrixSet_(obj)
            import saivdr.dictionary.cnsoltx.ChannelGroup
            nChs = obj.NumberOfChannels;
            nSts = obj.nStages;
            pmMtxSet_ = obj.ParameterMatrixSet;
            % V0
            mtx = step(obj.omgsV0_,obj.Angles(1:nChs*(nChs-1)/2),...
                obj.Mus(1:nChs));
            step(pmMtxSet_,mtx,uint32(1));
            
            angles = reshape(obj.Angles(nChs*(nChs-1)/2+1:end),[],obj.nStages-1);
            mus    = reshape(obj.Mus(nChs+1:end),[],obj.nStages-1);            
            nAngsW = floor(nChs/2)*(floor(nChs/2)-1)/2;
            nAngsHW = ceil(nChs/2)*(ceil(nChs/2)-1)/2;
            nAngsB = floor(nChs/4);
            nMusW = floor(nChs/2);
            nMusHW = ceil(nChs/2);
            omgsWU = obj.omgsWU_;
            omgsHWHU = obj.omgsHWHU_;
            
            % TODO: CNSOLTにおけるNo-DC-Leakage conditionを正しく定義する．
            % No-DC-Leakage condition
            %W_ = eye(ceil(nChs/2)); 
            for iParamMtx = uint32(1):nSts-1
                % W
                mtx = step(omgsWU,angles(1:nAngsW,iParamMtx),...
                    mus(1:nMusW,iParamMtx));
                step(pmMtxSet_,mtx,6*iParamMtx-4);
                % U
                mtx = step(omgsWU,angles(nAngsW+1:2*nAngsW,iParamMtx),...
                        mus(nMusW+1:2*nMusW,iParamMtx));
                step(pmMtxSet_,mtx,6*iParamMtx-3);
                
                % angsB1
                step(pmMtxSet_,angles(2*nAngsW+1:2*nAngsW+nAngsB,iParamMtx),6*iParamMtx-2);
                
                % HW
                mtx = step(omgsHWHU,angles(2*nAngsW+nAngsB+1:2*nAngsW+nAngsB+nAngsHW,iParamMtx),...
                    mus(2*nMusW+1:2*nMusW+nMusHW,iParamMtx));
                step(pmMtxSet_,mtx,6*iParamMtx-1);
                % HU
                mtx = step(omgsHWHU,angles(2*nAngsW+nAngsB+nAngsHW+1:2*nAngsW+nAngsB+2*nAngsHW,iParamMtx),...
                        mus(2*nMusW+nMusHW+1:end,iParamMtx));
                step(pmMtxSet_,mtx,6*iParamMtx);
                
                % angsB2
                step(pmMtxSet_,angles(2*nAngsW+1:2*nAngsW+nAngsB,iParamMtx),6*iParamMtx+1);

                %W_ = step(pmMtxSet,[],2*iParamMtx-1)*W_;
            end
            %TODO: 以下のソースを修正
%             [angles_,mus_] = step(obj.omfs_,W_.');
%             angles(1:ceil(nChs/2)-1,nSts) = ...
%                 angles_(1:ceil(nChs/2)-1);
%             mus(1,nSts) = mus_(1);
%             % W
%             mtx = step(omgsWU,angles(1:nAngsW,nSts),...
%                 mus(1:nMusW,nSts));            
%             step(pmMtxSet,mtx,2*nSts-1); 
%             % U
%             mtx = step(omgsWU,angles(nAngsW+1:end,nSts),...
%                 mus(nMusW+1:end,nSts));
%             step(pmMtxSet,mtx,2*nSts);
%             %
%             obj.Angles = angles;
%             obj.Mus    = mus;
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = 0;
        end
    end
end
