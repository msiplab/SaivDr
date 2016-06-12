classdef OLpPrFbAtomExtender1d <  ...
        saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d %#codegen
    %OLPPRFBATOMEXTENDER1D 1-D Atom Extender for OLPPRFB
    %
    % SVN identifier:
    % $Id: OLpPrFbAtomExtender1d.m 657 2015-03-17 00:45:15Z sho $
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
        function obj = OLpPrFbAtomExtender1d(varargin)
            obj = obj@saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d(varargin{:});
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d(obj,s,wasLocked);
        end
        
        function arrayCoefs = stepImpl(obj, arrayCoefs, subScale, pmCoefs)
            stepImpl@saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d(obj,arrayCoefs,subScale,pmCoefs);        
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
            
            %hLenU = obj.NumberOfSymmetricChannels;            
            %
            if ~isempty(obj.paramMtxCoefs)
                %W0 = getParamMtx_(obj,uint32(1));
                %U0 = getParamMtx_(obj,uint32(2));
                V0 = getParamMtx_(obj,uint32(1));
                %arrayCoefs(1:hLenU,:)     = W0*arrayCoefs(1:hLenU,:);
                %arrayCoefs(hLenU+1:end,:) = U0*arrayCoefs(hLenU+1:end,:);
                %nChs = obj.NumberOfSymmetricChannels+obj.NumberOfAntisymmetricChannels;
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
            isPsGtPa      = obj.IsPsGreaterThanPa;            
            %
            ord = obj.PolyPhaseOrder;
            
            % Order extension
            for iOrd = uint32(1):uint32(ord/2)
                paramMtx1 = getParamMtx_(obj,6*iOrd-4); % W
                paramMtx2 = getParamMtx_(obj,6*iOrd-3); % U
                paramMtx3 = getParamMtx_(obj,6*iOrd-2); % angB1
                paramMtx4 = getParamMtx_(obj,6*iOrd-1); % HW
                paramMtx5 = getParamMtx_(obj,6*iOrd+0); % HU
                paramMtx6 = getParamMtx_(obj,6*iOrd+1); % angB2
                %
                %if isPsGtPa
                    arrayCoefs = supportExtTypeII_(obj,arrayCoefs,...
                        paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,...
                        isPeriodicExt);
                %else
                %    arrayCoefs = supportExtTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,[],isPeriodicExt);
                %end
            end

        end
        
        function arrayCoefs = supportExtTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;

            % Phase 1
            Wx = paramMtx1;
            Ux = paramMtx2;
            B = butterflyMtx_(obj,paramMtx3);
            cB = conj(B);
            %TODO:
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,angB1);
            arrayCoefs = cB'*arrayCoefs;
            arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = cB*arrayCoefs;
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,angB1);
            %arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
            else % TODO:周期拡張でない場合の変換方法の妥当性を確認する
                arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
                
                arrayCoefs(hLen+1:end,1:end-1) = Ux*arrayCoefs(hLen+1:end,1:end-1);
                arrayCoefs(hLen+1:end,end) = -arrayCoefs(hLen+1:end,end);
            end
            
            % Phase 2
            Wx = paramMtx4;
            Ux = paramMtx5;
            B = butterflyMtx_(obj,paramMtx6);
            cB = conj(B);
            %B = eye(obj.NumberOfSymmetricChannels+obj.NumberOfAntisymmetricChannels);
            %TODO:
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,angB2);
            arrayCoefs = cB'*arrayCoefs;
            arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = cB*arrayCoefs;
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,angB2);
            %arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
            arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
            %arrayCoefs = conj(arrayCoefs);
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
                % TODO:　アルゴリズムの正当性を確かめる
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
        
        function hB = butterflyMtx_(obj, angles)%TODO: 同一の関数がAbstBuildingBlock.mで実装されているので一箇所にまとめる．
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
