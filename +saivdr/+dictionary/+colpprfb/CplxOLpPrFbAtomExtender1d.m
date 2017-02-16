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
                arrayCoefs = V0(1:obj.NumberOfChannels,:)*arrayCoefs;
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
            hLen = obj.NumberOfHalfChannels;
            %import saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock

            % Phase 1
            Wx1 = paramMtx1;
            Ux1 = paramMtx2;
            B1 = saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock.butterflyMtx(hLen,paramMtx3);
            %cB1 = conj(B1);
            %TODO: 実装の効率化を検討
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,angB1);
            arrayCoefs = B1'*arrayCoefs;
            arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = B1*arrayCoefs;
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,angB1);
            %arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else % TODO:周期拡張でない場合の変換方法の妥当性を確認する
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                
                arrayCoefs(hLen+1:end,1:end-1) = Ux1*arrayCoefs(hLen+1:end,1:end-1);
                arrayCoefs(hLen+1:end,end) = Ux1*arrayCoefs(hLen+1:end,end);
            end
            
            % Phase 2
            Wx2 = paramMtx4;
            Ux2 = paramMtx5;
            B2 = saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock.butterflyMtx(hLen,paramMtx6);
            %cB2 = conj(B2);
            %TODO:
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,angB2);
            arrayCoefs = B2'*arrayCoefs;
            arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = B2*arrayCoefs;
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,angB2);
            %arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
            arrayCoefs(1:hLen,:) = Wx2*arrayCoefs(1:hLen,:);
        end
        
        function arrayCoefs = supportExtTypeII_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            hLen = obj.NumberOfHalfChannels;
            %import saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock
            
            % Phase 1
            Wx = paramMtx1;
            Ux = paramMtx2;
            B = saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock.butterflyMtx(hLen,paramMtx3);
            %cB = conj(B);
            %arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            arrayCoefs(1:end-1,:) = B'*arrayCoefs(1:end-1,:);
            arrayCoefs(1:end-1,:) = rightShiftLowerCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = B*arrayCoefs(1:end-1,:);
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
                arrayCoefs(hLen+1:end-1,end) = Ux*arrayCoefs(hLen+1:end-1,end);
            end
            
            % Phase 2
            Wx = paramMtx4;
            Ux = paramMtx5;
            B = saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock.butterflyMtx(hLen,paramMtx6);
            %cB = conj(B);
            %arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            arrayCoefs(1:end-1,:) = B'*arrayCoefs(1:end-1,:);
            arrayCoefs(1:end-1,:) = leftShiftUpperCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = B*arrayCoefs(1:end-1,:);
            %arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            %arrayCoefs = arrayCoefs/2.0;
            % Upper channel rotation
            arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
            arrayCoefs(1:hLen+1,:) = Wx*arrayCoefs(1:hLen+1,:);
        end
        
        function arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs)
            hLenMn = obj.NumberOfHalfChannels;
            %
            lowerCoefsPre = arrayCoefs(hLenMn+1:end,end);
            arrayCoefs(hLenMn+1:end,2:end) = arrayCoefs(hLenMn+1:end,1:end-1);
            arrayCoefs(hLenMn+1:end,1) = lowerCoefsPre;
        end
        
        function arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs)
            hLenMn = obj.NumberOfHalfChannels;
            %
            upperCoefsPost = arrayCoefs(1:hLenMn,1);
            arrayCoefs(1:hLenMn,1:end-1) = arrayCoefs(1:hLenMn,2:end);
            arrayCoefs(1:hLenMn,end) = upperCoefsPost;
        end
    end
    
end
