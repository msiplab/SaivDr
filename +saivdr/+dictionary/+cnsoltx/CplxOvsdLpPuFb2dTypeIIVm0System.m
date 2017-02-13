classdef CplxOvsdLpPuFb2dTypeIIVm0System < saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dTypeIISystem
    %OVSDLPPUFBMDTYPEIIVM0SYSTEM M-D Type-II Oversampled LPPUFB without VM
    %
    % SVN identifier:
    % $Id: OvsdLpPuFb2dTypeIIVm0System.m 683 2015-05-29 08:22:13Z sho $
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
        propOmgs1st_
        propOmgs2nd_
    end
    
    methods
        function obj = CplxOvsdLpPuFb2dTypeIIVm0System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dTypeIISystem(...
                varargin{:});
            obj.initOmgs_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            obj.propOmgs1st_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            obj.propOmgs2nd_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dTypeIISystem(obj);
            s.initOmgs_ = matlab.System.saveObject(obj.initOmgs_);
            s.propOmgs1st_ = matlab.System.saveObject(obj.propOmgs1st_);
            s.propOmgs2nd_ = matlab.System.saveObject(obj.propOmgs2nd_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.initOmgs_ = matlab.System.loadObject(s.initOmgs_);
            obj.propOmgs1st_ = matlab.System.loadObject(s.propOmgs1st_);
            obj.propOmgs2nd_ = matlab.System.loadObject(s.propOmgs2nd_);
            loadObjectImpl@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dTypeIISystem(obj,s,wasLocked);
        end                
        
        function updateParameterMatrixSet_(obj)
            nCh = obj.NumberOfChannels;
            hCh = floor(nCh/2);
            pmMtxSt_ = obj.ParameterMatrixSet;
            
            [initAngles, propAngles] = splitAngles_(obj);
            % V0
            mtx = step(obj.initOmgs_,initAngles,obj.Mus(1:nCh));
            step(pmMtxSt_,mtx,uint32(1));
            
            angles = reshape(propAngles,[],obj.nStages-1);
            mus    = reshape(obj.Mus(nCh+1:end),[],obj.nStages-1);            
            nAngs1st = hCh*(hCh-1)/2;
            nAngs2nd = (hCh+1)*hCh/2;
            nAngsB = floor(nCh/4);
            nMus1st = hCh;
            nMus2nd = hCh+1;
            
            for iParamMtx = uint32(1):obj.nStages-1
                % W
                mtx = step(obj.propOmgs1st_,angles(1:nAngs1st,iParamMtx),...
                    mus(1:nMus1st,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx-4);
                
                % U
                mtx = step(obj.propOmgs1st_,angles(nAngs1st+1:2*nAngs1st,iParamMtx),...
                        mus(nMus1st+1:2*nMus1st,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx-3);
                
                % angsB1
                step(pmMtxSt_,angles(2*nAngs1st+1:2*nAngs1st+nAngsB,iParamMtx),6*iParamMtx-2);
                
                % HU
                mtx = step(obj.propOmgs2nd_,angles(2*nAngs1st+nAngsB+nAngs2nd+1:2*nAngs1st+nAngsB+2*nAngs2nd,iParamMtx),...
                        mus(2*nMus1st+nMus2nd+1:end,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx);
                
                % HW
                mtx = step(obj.propOmgs2nd_,angles(2*nAngs1st+nAngsB+1:2*nAngs1st+nAngsB+nAngs2nd,iParamMtx),...
                    mus(2*nMus1st+1:2*nMus1st+nMus2nd,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx-1);

                % angsB2
                step(pmMtxSt_,angles(2*nAngs1st+1:2*nAngs1st+nAngsB,iParamMtx),6*iParamMtx+1);
            end
        end

    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = [ 0 0 ];
        end
    end
    
end
