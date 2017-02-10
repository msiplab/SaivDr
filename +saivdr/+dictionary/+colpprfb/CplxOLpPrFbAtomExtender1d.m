classdef CplxOLpPrFbAtomExtender1d <  ...
        saivdr.dictionary.colpprfb.AbstCplxOLpPrFbCoefManipulator1d %#codegen
    %OLPPRFBATOMEXTENDER1D 1-D Atom Extender for OLPPRFB
    %
    % SVN identifier:
    % $Id: CplxOLpPrFbAtomExtender1d.m 657 2015-03-17 00:45:15Z sho $
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
    
    methods
        
        % Constructor
        function obj = CplxOLpPrFbAtomExtender1d(varargin)
            obj = obj@saivdr.dictionary.colpprfb.AbstCplxOLpPrFbCoefManipulator1d(varargin{:});
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOLpPrFbCoefManipulator1d(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOLpPrFbCoefManipulator1d(obj,s,wasLocked);
        end
        
        function arrayCoefs = stepImpl(obj, arrayCoefs, subScale, pmCoefs)
            stepImpl@saivdr.dictionary.colpprfb.AbstCplxOLpPrFbCoefManipulator1d(obj,arrayCoefs,subScale,pmCoefs);        
            %
            arrayCoefs = initialStep_(obj,arrayCoefs);
            %
            if strcmp(obj.OLpPrFbType,'Type I')
                arrayCoefs = fullAtomExtTypeI_(obj,arrayCoefs);
            else
                arrayCoefs = fullAtomExtTypeII_(obj,arrayCoefs);
            end
            
        end
        
    end
    
    methods ( Access = private )
        
        function arrayCoefs = initialStep_(obj,arrayCoefs)

            %
            if ~isempty(obj.paramMtxCoefs)
                V0 = getParamMtx_(obj,uint32(1));
                arrayCoefs = V0*arrayCoefs;
            end

        end
        
        function arrayCoefs = fullAtomExtTypeI_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            %
            ord = obj.PolyPhaseOrder;
    
            % Order extension
            hOrd = uint32(ord/2);
            if hOrd > 0
                for iOrd = uint32(1):hOrd
                    paramMtxW1 = getParamMtx_(obj,6*iOrd-4);
                    paramMtxU1 = getParamMtx_(obj,6*iOrd-3);
                    paramAngB1 = getParamMtx_(obj,6*iOrd-2);
                    paramMtxW2 = getParamMtx_(obj,6*iOrd-1);
                    paramMtxU2 = getParamMtx_(obj,6*iOrd+0);
                    paramAngB2 = getParamMtx_(obj,6*iOrd+1);
                    %
                    arrayCoefs = supportExtTypeI_(obj,arrayCoefs,...
                        paramMtxW1,paramMtxU1,paramAngB1,...
                        paramMtxW2,paramMtxU2,paramAngB2,...
                        isPeriodicExt);
                end
            end
            
        end
        
        function arrayCoefs = fullAtomExtTypeII_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            %
            ord = obj.PolyPhaseOrder;
            
            % Order extension
            for iOrd = uint32(1):uint32(ord/2)
                paramMtx1 = getParamMtx_(obj,6*iOrd-4); % W
                paramMtx2 = getParamMtx_(obj,6*iOrd-3); % U
                paramMtx3 = getParamMtx_(obj,6*iOrd-2); % angB1
                paramMtx4 = getParamMtx_(obj,6*iOrd-1); % hW
                paramMtx5 = getParamMtx_(obj,6*iOrd+0); % hU
                paramMtx6 = getParamMtx_(obj,6*iOrd+1); % angB2
                %
                arrayCoefs = supportExtTypeII_(...
                    obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,...
                    paramMtx4,paramMtx5,paramMtx6,isPeriodicExt);
            end

        end
        
        function arrayCoefs = supportExtTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            hLen = obj.NumberOfAntisymmetricChannels;

            % Phase 1
            Wx1 = paramMtx1;
            Ux1 = paramMtx2;
            B1 = butterflyMtx_(obj,paramMtx3);
            cB1 = conj(B1);
            %TODO: �����̌�����������
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,angB1);
            arrayCoefs = cB1'*arrayCoefs;
            arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = cB1*arrayCoefs;
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,angB1);
            %arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else % TODO:�����g���łȂ��ꍇ�̕ϊ����@�̑Ó������m�F����
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                
                arrayCoefs(hLen+1:end,1:end-1) = Ux1*arrayCoefs(hLen+1:end,1:end-1);
                arrayCoefs(hLen+1:end,end) = -arrayCoefs(hLen+1:end,end);
            end
            
            % Phase 2
            Wx2 = paramMtx4;
            Ux2 = paramMtx5;
            B2 = butterflyMtx_(obj,paramMtx6);
            cB2 = conj(B2);
            %TODO:
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,angB2);
            arrayCoefs = cB2'*arrayCoefs;
            arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = cB2*arrayCoefs;
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,angB2);
            %arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
            arrayCoefs(1:hLen,:) = Wx2*arrayCoefs(1:hLen,:);
        end
        
        function arrayCoefs = supportExtTypeII_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            hLen = obj.NumberOfAntisymmetricChannels;
            
            % Phase 1
            Wx = paramMtx1;
            Ux = paramMtx2;
            B = butterflyMtx_(obj,paramMtx3);
            cB = conj(B);
            %arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            arrayCoefs(1:end-1,:) = cB'*arrayCoefs(1:end-1,:);
            arrayCoefs(1:end-1,:) = leftShiftLowerCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = cB*arrayCoefs(1:end-1,:);
            %arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            %arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end-1,:) = Ux*arrayCoefs(hLen+1:end-1,:);
            else
                % TODO:�@�A���S���Y���̐��������m���߂�
                arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
                
                arrayCoefs(hLen+1:end-1,1:end-1) = Ux*arrayCoefs(hLen+1:end-1,1:end-1);
                arrayCoefs(hLen+1:end-1,end) = -arrayCoefs(hLen+1:end-1,end);
            end
            
            % Phase 2
            Wx = paramMtx4;
            Ux = paramMtx5;
            B = butterflyMtx_(obj,paramMtx6);
            cB = conj(B);
            %arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            arrayCoefs(1:end-1,:) = cB'*arrayCoefs(1:end-1,:);
            arrayCoefs(1:end-1,:) = rightShiftUpperCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = cB*arrayCoefs(1:end-1,:);
            %arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            %arrayCoefs = arrayCoefs/2.0;
            % Upper channel rotation
            arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
            arrayCoefs(1:hLen+1,:) = Wx*arrayCoefs(1:hLen+1,:);
        end
        
        function arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs)
            hLenMn = min([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            %
            lowerCoefsPost = arrayCoefs(hLenMn+1:end,1);
            arrayCoefs(hLenMn+1:end,1:end-1) = arrayCoefs(hLenMn+1:end,2:end);
            arrayCoefs(hLenMn+1:end,end) = lowerCoefsPost;
        end
        
        function arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs)
            hLenMn = min([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            %
            upperCoefsPre = arrayCoefs(1:hLenMn,end);
            arrayCoefs(1:hLenMn,2:end) = arrayCoefs(1:hLenMn,1:end-1);
            arrayCoefs(1:hLenMn,1) = upperCoefsPre;
        end
        
        function hB = butterflyMtx_(obj, angles)%TODO: ����̊֐���AbstBuildingBlock.m�Ŏ�������Ă���̂ň�ӏ��ɂ܂Ƃ߂�D
            hchs = obj.NumberOfAntisymmetricChannels;
            
            hC = complex(eye(hchs));
            hS = complex(eye(hchs));
            for p = 1:floor(hchs/2)
                tp = angles(p);
                
                hC(2*p-1:2*p, 2*p-1:2*p) = [ -1i*cos(tp), -1i*sin(tp);
                    cos(tp) , -sin(tp)]; %c^
                hS(2*p-1:2*p, 2*p-1:2*p) = [ -1i*sin(tp), -1i*cos(tp);
                    sin(tp) , -cos(tp)]; %s^
            end
            
            hB = [hC, conj(hC); 1i*hS, -1i*conj(hS)]/sqrt(2);
        end
    end
    
end