classdef nsoltSubbandSerialization3dLayerTestCase < matlab.unittest.TestCase
    %NSOLTSUBBANDSERIALIZATION3DLAYERTESTCASE
    %
    %   複数コンポーネント入力 (SSSCB):（ツリーレベル数）
    %      nRowsLv1 x nColsLv1 x nLaysLv1 x (nChsTotal-1) x nSamples
    %      nRowsLv2 x nColsLv2 x nLaysLv2 x (nChsTotal-1) x nSamples
    %       :
    %      nRowsLvN x nColsLvN x nLaysLvN x (nChsTotal-1) x nSamples
    %      nRowsLvN x nColsLvN x nLaysLvN x 1 x nSamples    
    %
    %   １コンポーネント出力(SSSCB):
    %      nElements x 1 x 1 x 1 x nSamples
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
            layer = nsoltSubbandSerialization3dLayer(...
                'OriginalDimension',[16 16 16],...
                'NumberOfChannels',[5 5],...
                'DecimationFactor',[2 2 2],...
                'NumberOfLevels',3);
            fprintf("\n --- Check layer for 3-D images ---\n");
            checkLayer(layer,{[8 8 8 9], [4 4 4 9], [2 2 2 9], [2 2 2 1]},'ObservationDimension',5)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase,...
                nrows,ncols,nlays,nchs,stride,nlevels)
            
            % Expected values
            height = nrows*(stride(1)^nlevels);
            width = ncols*(stride(2)^nlevels);
            depth = nlays*(stride(3)^nlevels);
            expctdName = 'Sb_Srz';
            expctdDescription = "Subband serialization " ...
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
            layer = nsoltSubbandSerialization3dLayer(...
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
            layer = nsoltSubbandSerialization3dLayer(...
                'Name','Sb_Srz',...
                'OriginalDimension',[height width depth],...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'NumberOfLevels',nlevels);
            expctdOriginalDimension = 2*[height width depth];
            expctdScales = [2 2 2 1].*layer.Scales;
            
            % Actual values
            layer = layer.setOriginalDimension(expctdOriginalDimension);
            actualOriginalDimension = layer.OriginalDimension;
            actualScales = layer.Scales;
            
            % Evaluation
            testCase.verifyEqual(actualOriginalDimension,expctdOriginalDimension);
            testCase.verifyEqual(actualScales,expctdScales);
            
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
            X = cell(nlevels,1);
            for iLv = 1:nlevels %-1
                subHeight = nrows * stride(1)^(nlevels-iLv);
                subWidth = ncols * stride(2)^(nlevels-iLv);
                subDepth = nlays * stride(3)^(nlevels-iLv);                
                X{iLv} = randn(subHeight,subWidth,subDepth,...
                    nChsTotal-1,...
                    nSamples,datatype);
            end
            X{nlevels+1} = randn(nrows,ncols,nlays,1,...nChsTotal,...
                nSamples,datatype);
            
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
            expctdZ = zeros(nElements,1,1,1,nSamples,datatype);
            for iSample = 1:nSamples
                x = zeros(nElements,1,datatype);
                sidx = 0;
                nSubElements = prod(expctdScales(1,:));
                a = X{nlevels+1}(:,:,:,:,iSample);
                x(1:nSubElements) = a(:);
                sidx = sidx+nSubElements;
                for iRevLv = 1:nlevels
                    nSubElements = prod(expctdScales(iRevLv+1,:));
                    a = X{nlevels-iRevLv+1}(:,:,:,:,iSample);
                    x(sidx+1:sidx+nSubElements) = a(:);
                    sidx = sidx+nSubElements;
                end
                expctdZ(:,1,1,1,iSample) = x;
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltSubbandSerialization3dLayer(...
                'Name','Sb_Srz',...
                'OriginalDimension',[height width depth],...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'NumberOfLevels',nlevels);
            
            % Actual values
            actualZ = layer.predict(X{:});
            actualScales = layer.Scales;
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            testCase.verifyThat(actualScales,...
                IsEqualTo(expctdScales,'Within',tolObj));
            
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
            
            % Expected values
            expctddLdX = cell(nlevels+1,1);
            for iLv = 1:nlevels %-1
                subHeight = nrows * stride(1)^(nlevels-iLv);
                subWidth = ncols * stride(2)^(nlevels-iLv);
                subDepth = nlays * stride(3)^(nlevels-iLv);                
                expctddLdX{iLv} = randn(subHeight,subWidth,subDepth,...
                    nChsTotal-1,...
                    nSamples,datatype);
            end
            expctddLdX{nlevels+1} = randn(nrows,ncols,nlays,1,...nChsTotal,...
                nSamples,datatype);
            
            %expctdScales = zeros(nlevels,4);
            expctdScales = zeros(nlevels+1,4);
            %expctdScales(1,:) = [nrows ncols nlays nChsTotal];
            expctdScales(1,:) = [nrows ncols nlays 1];
            for iRevLv = 1:nlevels %2:nlevels
                expctdScales(iRevLv+1,:) = ...
                    [nrows*stride(1)^(iRevLv-1) ncols*stride(2)^(iRevLv-1) nlays*stride(3)^(iRevLv-1) nChsTotal-1];
            end
            
            % Input
            nElements = sum(prod(expctdScales,2));
            dLdZ = zeros(nElements,1,1,1,nSamples,datatype);
            for iSample = 1:nSamples
                x = zeros(nElements,1,datatype);
                sidx = 0;
                nSubElements = prod(expctdScales(1,:));
                a = expctddLdX{nlevels+1}(:,:,:,:,iSample);
                x(1:nSubElements) = a(:);
                sidx = sidx+nSubElements;
                for iRevLv = 1:nlevels
                    %nSubElements = prod(expctdScales(iRevLv,:));
                    nSubElements = prod(expctdScales(iRevLv+1,:));
                    a = expctddLdX{nlevels-iRevLv+1}(:,:,:,:,iSample);
                    x(sidx+1:sidx+nSubElements) = a(:);
                    sidx = sidx+nSubElements;
                end
                dLdZ(:,1,1,1,iSample) = x;
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltSubbandSerialization3dLayer(...
                'Name','Sb_Srz',...
                'OriginalDimension',[height width depth],...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'NumberOfLevels',nlevels);
            
            % Actual values
            %args = cell(1,nlevels+1+1+1);
            args = cell(1,nlevels+2+1+1);
            %args{nlevels+2} = dLdZ;
            args{nlevels+3} = dLdZ;
            %[actualdLdX{1:nlevels}] = layer.backward(args{:});
            [actualdLdX{1:nlevels+1}] = layer.backward(args{:});
            
            % Evaluation
            for iLv = 1:nlevels+1 %1:nlevels
                testCase.verifyInstanceOf(actualdLdX{iLv},datatype);
                testCase.verifyThat(actualdLdX{iLv},...
                    IsEqualTo(expctddLdX{iLv},'Within',tolObj));
            end
            
        end
    end
    
end

