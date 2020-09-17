classdef nsoltSubbandDeserialization3dLayerTestCase < matlab.unittest.TestCase
    %NSOLTCHANNELSEPARATION3DLAYERTESTCASE
    %
    %   １コンポーネント入力(SSSCB):
    %      nElements x 1 x 1 x 1 x nSamples
    %
    %   複数コンポーネント出力 (SSSCB):（ツリーレベル数）
    %      nRowsLv1 x nColsLv1 x nLaysLv1 x (nChsTotal-1) x nSamples
    %      nRowsLv2 x nColsLv2 x nLaysLv2 x (nChsTotal-1) x nSamples
    %       :
    %      nRowsLvN x nColsLvN x nLaysLvN x (nChsTotal-1) x nSamples
    %      nRowsLvN x nColsLvN x nLaysLvN x 1 x nSamples     
    %
    % Requirements: MATLAB R2020a
    %
    % Copyright (c) 2020, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    
    properties (TestParameter)
        nlevels = { 1, 2, 3 };
        stride = { [2 2 2], [1 2 2], [2 1 2], [2 2 1] };
        nchs = { [5 5], [6 6] };
        datatype = { 'single', 'double' };
        nrows = struct('small', 4,'medium', 8, 'large', 16);
        ncols = struct('small', 4,'medium', 8, 'large', 16);
        nlays = struct('small', 4,'medium', 8, 'large', 16);        
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltSubbandDeserialization3dLayer(...
                'OriginalDimension',[16 16 16],...
                'NumberOfChannels',[5 5],...
                'DecimationFactor',[2 2 2],...
                'NumberOfLevels',3);
            fprintf("\n --- Check layer for 3-D images ---\n");
            checkLayer(layer,[5264 1 1 1],'ObservationDimension',5)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase,...
                nrows,ncols,nlays,nchs,stride,nlevels)
            
            % Expected values
            height = nrows*(stride(1)^nlevels);
            width = ncols*(stride(2)^nlevels);
            depth = nlays*(stride(3)^nlevels);            
            expctdName = 'Sb_Dsz';
            expctdDescription = "Subband deserialization " ...
                + "(h,w,d) = (" ...
                + height + "," + width + "," + depth + "), "  ...
                + "lv = " ...
                + nlevels + ", " ...
                + "(ps,pa) = (" ...
                + nchs(1) + "," + nchs(2) + "), "  ...
                + "(mv,mh,md) = (" ...
                + stride(1) + "," + stride(2) + "," + stride(3) + ")";
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltSubbandDeserialization3dLayer(...
                'Name',expctdName,...
                'OriginalDimension',[height width depth],...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'NumberOfLevels',nlevels);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        function testSetOriginalDimension(testCase,...
                nrows,ncols,nlays,nchs,stride,nlevels)
            
            % Expected values
            height = nrows*(stride(1)^nlevels);
            width = ncols*(stride(2)^nlevels);
            depth = nlays*(stride(3)^nlevels);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltSubbandDeserialization3dLayer(...
                'Name','Sb_Dsz',...
                'OriginalDimension',[height width depth],...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'NumberOfLevels',nlevels);
            expctdOriginalDimension = 2*[height width depth];
            expctdInputSize = [2^3 1 1 1].*layer.InputSize;
            
            % Actual values
            layer = layer.setOriginalDimension(expctdOriginalDimension);
            actualOriginalDimension = layer.OriginalDimension;
            actualInputSize = layer.InputSize;
            
            % Evaluation
            testCase.verifyEqual(actualOriginalDimension,expctdOriginalDimension);
            testCase.verifyEqual(actualInputSize,expctdInputSize);
            
        end
        
        function testPredict(testCase,...
                nchs,nrows,ncols,nlays,stride,nlevels,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            height = nrows*(stride(1)^nlevels);
            width = ncols*(stride(2)^nlevels);
            depth = nlays*(stride(3)^nlevels);            
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            
            % Expected values
            %expctdZ = cell(nlevels,1);
            expctdZ = cell(nlevels+1,1);
            for iLv = 1:nlevels %-1
                subHeight = nrows * stride(1)^(nlevels-iLv);
                subWidth = ncols * stride(2)^(nlevels-iLv);
                subDepth = nlays * stride(3)^(nlevels-iLv);                                
                expctdZ{iLv} = randn(subHeight,subWidth,subDepth,...
                    nChsTotal-1,...
                    nSamples,datatype);
            end
            expctdZ{nlevels+1} = randn(nrows,ncols,nlays,1,...
                nSamples,datatype); %
            
            %expctdScales = zeros(nlevels,4);
            expctdScales = zeros(nlevels+1,4);
            %expctdScales(1,:) = [nrows ncols nlays nChsTotal];
            expctdScales(1,:) = [nrows ncols nlays 1];
            for iRevLv = 1:nlevels %2:nlevels
                %expctdScales(iRevLv,:) = ...
                expctdScales(iRevLv+1,:) = ...
                    [nrows*stride(1)^(iRevLv-1) ncols*stride(2)^(iRevLv-1) nlays*stride(3)^(iRevLv-1) nChsTotal-1];                
            end
            expctdInputSize = [sum(prod(expctdScales,2)) 1 1 1 ];
            
            % Input
            nElements = sum(prod(expctdScales,2));
            X = zeros(nElements,1,1,1,nSamples,datatype);
            for iSample = 1:nSamples
                x = zeros(nElements,1,datatype);
                sidx = 0;
                nSubElements = prod(expctdScales(1,:));
                a = expctdZ{nlevels+1}(:,:,:,:,iSample);
                x(1:nSubElements) = a(:);
                sidx = sidx+nSubElements;
                for iRevLv = 1:nlevels
                    %nSubElements = prod(expctdScales(iRevLv,:));
                    nSubElements = prod(expctdScales(iRevLv+1,:));
                    a = expctdZ{nlevels-iRevLv+1}(:,:,:,:,iSample);
                    x(sidx+1:sidx+nSubElements) = a(:);
                    sidx = sidx+nSubElements;
                end
                X(:,1,1,1,iSample) = x;
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltSubbandDeserialization3dLayer(...
                'Name','Sb_Dsz',...
                'OriginalDimension',[height width depth],...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'NumberOfLevels',nlevels);
            
            % Actual values
            %[actualZ{1:nlevels}] = layer.predict(X);
            [actualZ{1:nlevels+1}] = layer.predict(X);
            actualScales = layer.Scales;
            actualInputSize = layer.InputSize;
            
            % Evaluation
            for iLv = 1:nlevels+1 %1:nlevels
                testCase.verifyInstanceOf(actualZ{iLv},datatype);
                testCase.verifyThat(actualZ{iLv},...
                    IsEqualTo(expctdZ{iLv},'Within',tolObj));
            end
            testCase.verifyThat(actualScales,...
                IsEqualTo(expctdScales,'Within',tolObj));
            testCase.verifyThat(actualInputSize,...
                IsEqualTo(expctdInputSize,'Within',tolObj));
                        
        end
        
        function testBackward(testCase,...
                nchs,nrows,ncols,nlays,stride,nlevels,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            height = nrows*(stride(1)^nlevels);
            width = ncols*(stride(2)^nlevels);
            depth = nlays*(stride(3)^nlevels);            
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            %dLdZ = cell(nlevels,1);
            dLdZ = cell(nlevels+1,1);
            for iLv = 1:nlevels %-1
                subHeight = nrows * stride(1)^(nlevels-iLv);
                subWidth = ncols * stride(2)^(nlevels-iLv);
                subDepth = nlays * stride(3)^(nlevels-iLv);                                
                dLdZ{iLv} = randn(subHeight,subWidth,subDepth,...
                    nChsTotal-1,...
                    nSamples,datatype);
            end
            dLdZ{nlevels+1} = randn(nrows,ncols,nlays,1,...nChsTotal,...
                nSamples,datatype); %
            
            % Expected values
            %expctdScales = zeros(nlevels,4);
            expctdScales = zeros(nlevels+1,4);
            %expctdScales(1,:) = [nrows ncols nlays nChsTotal];
            expctdScales(1,:) = [nrows ncols nlays 1];
            for iRevLv = 1:nlevels %2:nlevels
                expctdScales(iRevLv+1,:) = ...
                    [nrows*stride(1)^(iRevLv-1) ncols*stride(2)^(iRevLv-1) nlays*stride(3)^(iRevLv-1) nChsTotal-1];
            end
            nElements = sum(prod(expctdScales,2));
            expctddLdX = zeros(nElements,1,1,1,nSamples,datatype);
            for iSample = 1:nSamples
                x = zeros(nElements,1,datatype);
                sidx = 0;
                nSubElements = prod(expctdScales(1,:));
                a = dLdZ{nlevels+1}(:,:,:,:,iSample);
                x(1:nSubElements) = a(:);
                sidx = sidx+nSubElements;
                for iRevLv = 1:nlevels
                    nSubElements = prod(expctdScales(iRevLv+1,:));
                    a = dLdZ{nlevels-iRevLv+1}(:,:,:,:,iSample);
                    x(sidx+1:sidx+nSubElements) = a(:);
                    sidx = sidx+nSubElements;
                end
                expctddLdX(:,1,1,1,iSample) = x;
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltSubbandDeserialization3dLayer(...
                'Name','Sb_Srz',...
                'OriginalDimension',[height width depth],...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'NumberOfLevels',nlevels);
            
            % Actual values
            %args = cell(1,1+nlevels+nlevels+1);
            args = cell(1,1+nlevels+1+nlevels+1+1);
            for iLv = 1:nlevels+1 % 1:nlevels
                %args{1+nlevels+iLv} = dLdZ{iLv};
                args{1+nlevels+1+iLv} = dLdZ{iLv};
            end
            actualdLdX = layer.backward(args{:});
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end
    end
    
end

