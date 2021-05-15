classdef nsoltComponentSeparationLayerTestCase < matlab.unittest.TestCase
    %NSOLTCOMPONENTSEPARATIONLAYERTESTCASE
    %
    %   １コンポーネント入力:
    %       nRows x nCols x (N x nChsTotal) x nSamples
    %
    %   Nコンポーネントト出力
    %      nRows x nCols x ｎChsTotal x nSamples
    %      nRows x nCols x ｎChsTotal x nSamples
    %          :
    %      nRows x nCols x ｎChsTotal x nSamples
    %
    % Requirements: MATLAB R2020b
    %
    % Copyright (c) 2021, Shogo MURAMATSU
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
        ncmps = { 1, 3, 5 };
        nchs = { [3 3], [4 4] };
        datatype = { 'single', 'double' };
        nrows = struct('small', 1,'medium', 4, 'large', 16);
        ncols = struct('small', 1,'medium', 4, 'large', 16);
        batch = { 1, 8 };
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            nComponents = 3;
            layer = nsoltComponentSeparation2dLayer(nComponents);
            fprintf("\n --- Check layer for 2-D images ---\n");
            checkLayer(layer,[32 32 nComponents],...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase,ncmps)
            
            % Expected values
            expctdName = 'Cs';
            expctdOutputNames = cell(1,ncmps);
            for icmp = 1:ncmps
                expctdOutputNames{icmp} = [ 'out' num2str(icmp) ];
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltComponentSeparation2dLayer(ncmps,'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualOutputNames = layer.OutputNames;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);    
            testCase.verifyEqual(actualOutputNames,expctdOutputNames);                
        end
        
        function testPredict(testCase,ncmps,nchs,nrows,ncols,batch,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = batch;
            nChsTotal = sum(nchs);
            
            % Expected values            
            expctdX = cell(1,ncmps);
            for icmp = 1:ncmps
                expctdX{icmp} = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltComponentSeparation2dLayer(ncmps,'Name','Cs');
            
            % Actual values
            X = cat(3,expctdX{:});
            [actualX{1:ncmps}] = layer.predict(X);
            % Evaluation
            for icmp = 1:ncmps
                testCase.verifyInstanceOf(actualX{icmp},datatype);
                testCase.verifySize(actualX{icmp},size(expctdX{icmp}));
                testCase.verifyThat(actualX{icmp},...
                    IsEqualTo(expctdX{icmp},'Within',tolObj));
            end
            
        end

    end
    
end

