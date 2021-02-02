classdef ParameterMatrixContainerTestCase < matlab.unittest.TestCase
    %PARAMETERMATRIXCONTAINERTESTCASE Test case for ParameterMatrixContainer
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2016, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/    
    %
    
    properties
        pms
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.pms)
        end
    end
    
    methods (Test)
        
        % Test for default construction
        function testConstructor(testCase)

            % Preparation
            mstab = [2 2];
            
            % Expected values
            coefsExpctd = zeros(4,1);
            npmtxExpctd = size(mstab,1);
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.pms = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            
            % Actual values
            coefsActual = get(testCase.pms,'Coefficients');
            npmtxActual = get(testCase.pms,'NumberOfParameterMatrices');
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd)
            testCase.verifyEqual(npmtxActual,npmtxExpctd)
            
        end
        
        % Test for default construction
        function testConstructor2(testCase)

            % Preparation
            mstab = [
                2 2 ;
                2 2 ];

            % Expected values
            coefsExpctd = zeros(8,1);
            npmtxExpctd = size(mstab,1);
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.pms = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            
            % Actual values
            coefsActual = get(testCase.pms,'Coefficients');
            npmtxActual = get(testCase.pms,'NumberOfParameterMatrices');

            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd)
            testCase.verifyEqual(npmtxActual,npmtxExpctd)
            
        end        
        
        % Test for default construction
        function testConstructor4(testCase)

            % Preparation
            mstab = [
                2 2 ;
                4 4 ;
                2 2 ;
                4 4 ];

            % Expected values
            coefsExpctd = zeros(40,1);
            npmtxExpctd = size(mstab,1);
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.pms = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            
            % Actual values
            coefsActual = get(testCase.pms,'Coefficients');
            npmtxActual = get(testCase.pms,'NumberOfParameterMatrices');
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd)
            testCase.verifyEqual(npmtxActual,npmtxExpctd)
            
        end                
        
        % Test for default construction
        function testSetCoefficients(testCase)

            % Preparation
            mstab = [
                2 2 ;
                4 4 ;
                2 2 ;
                4 4 ];
            mtx = ones(2);
            idx = 1;

            % Expected values
            coefsExpctd = zeros(40,1);
            coefsExpctd(1:4) = 1;
                       
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.pms = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            
            % Actual values
            step(testCase.pms,mtx,idx);
            coefsActual = get(testCase.pms,'Coefficients');
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd)
            
        end                
        
        % Test for default construction
        function testSetCoefficients2(testCase)

            % Preparation
            mstab = [
                2 2 ;
                4 4 ;
                2 2 ;
                4 4 ];
            mtx1 = ones(2);
            idx1 = 1;
            mtx2 = 2*ones(4);
            idx2 = 2;

            % Expected values
            coefsExpctd = zeros(40,1);
            coefsExpctd(1:4) = 1;
            coefsExpctd(5:20) = 2;
                       
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.pms = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            
            % Actual values
            step(testCase.pms,mtx1,idx1);
            step(testCase.pms,mtx2,idx2);
            coefsActual = get(testCase.pms,'Coefficients');
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd)
            
        end                
        
        % Test for default construction
        function testSetCoefficients4(testCase)

            % Preparation
            mstab = [
                2 2 ;
                4 4 ;
                2 2 ;
                4 4 ];
            mtx1 = ones(2);
            idx1 = 1;
            mtx2 = 2*ones(4);
            idx2 = 2;
            mtx3 = 3*ones(2);
            idx3 = 3;
            mtx4 = 4*ones(4);
            idx4 = 4;            

            % Expected values
            coefsExpctd = zeros(40,1);
            coefsExpctd(1:4) = 1;
            coefsExpctd(5:20) = 2;
            coefsExpctd(21:24) = 3;
            coefsExpctd(25:40) = 4;            
                       
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.pms = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            
            % Actual values
            step(testCase.pms,mtx1,idx1);
            step(testCase.pms,mtx2,idx2);
            step(testCase.pms,mtx3,idx3);
            step(testCase.pms,mtx4,idx4);            
            coefsActual = get(testCase.pms,'Coefficients');
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd)
            
        end                
        
        % Test for default construction
        function testGetCoefficients(testCase)

            % Preparation
            mstab = [
                2 2 ;
                4 4 ;
                2 2 ;
                4 4 ];
            idx = 1;
            coefs = zeros(40,1);
            coefs(1:4) = 1;
            
            % Expected values
            mtxExpctd = ones(2);
                       
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.pms = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            set(testCase.pms,'Coefficients',coefs);
            
            % Actual values
            mtxActual = step(testCase.pms,[],idx);
            
            % Evaluation
            testCase.verifySize(mtxActual,size(mtxExpctd));
            testCase.verifyEqual(mtxActual,mtxExpctd)
            
        end   
        
        % Test for default construction
        function testGetCoefficients2(testCase)

            % Preparation
            mstab = [
                2 2 ;
                4 4 ;
                2 2 ;
                4 4 ];
            coefs = zeros(40,1);
            coefs(1:4) = 1;
            coefs(5:20) = 2;            
            
            % Expected values
            idx1 = 1;
            mtx1Expctd = ones(2);
            idx2 = 2;
            mtx2Expctd = 2*ones(4);
                       
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.pms = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            set(testCase.pms,'Coefficients',coefs);
            
            % Actual values
            mtx1Actual = step(testCase.pms,[],idx1);
            mtx2Actual = step(testCase.pms,[],idx2);
            
            % Evaluation
            testCase.verifySize(mtx1Actual,size(mtx1Expctd));
            testCase.verifyEqual(mtx1Actual,mtx1Expctd)
            testCase.verifySize(mtx2Actual,size(mtx2Expctd));
            testCase.verifyEqual(mtx2Actual,mtx2Expctd)            
            
        end  
        
        % Test for default construction
        function testGetCoefficients4(testCase)

            % Preparation
            mstab = [
                2 2 ;
                4 4 ;
                2 2 ;
                4 4 ];
            coefs = zeros(40,1);
            coefs(1:4) = 1;
            coefs(5:20) = 2;            
            coefs(21:24) = 3;            
            coefs(25:40) = 4;            
            
            % Expected values
            idx1 = 1;
            mtx1Expctd = ones(2);
            idx2 = 2;
            mtx2Expctd = 2*ones(4);
            idx3 = 3;
            mtx3Expctd = 3*ones(2);
            idx4 = 4;
            mtx4Expctd = 4*ones(4);            
                       
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.pms = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            set(testCase.pms,'Coefficients',coefs);
            
            % Actual values
            mtx1Actual = step(testCase.pms,[],idx1);
            mtx2Actual = step(testCase.pms,[],idx2);
            mtx3Actual = step(testCase.pms,[],idx3);
            mtx4Actual = step(testCase.pms,[],idx4);            
            
            % Evaluation
            testCase.verifySize(mtx1Actual,size(mtx1Expctd));
            testCase.verifyEqual(mtx1Actual,mtx1Expctd)
            testCase.verifySize(mtx2Actual,size(mtx2Expctd));
            testCase.verifyEqual(mtx2Actual,mtx2Expctd)            
            testCase.verifySize(mtx3Actual,size(mtx3Expctd));
            testCase.verifyEqual(mtx3Actual,mtx3Expctd)
            testCase.verifySize(mtx4Actual,size(mtx4Expctd));
            testCase.verifyEqual(mtx4Actual,mtx4Expctd)            
            
        end  
        
        % Test for default construction
        function testSetGetCoefficients4(testCase)

            % Preparation
            mstab = [
                2 2 ;
                4 4 ;
                2 2 ;
                4 4 ];
            
            % Expected values
            idx1 = 1;
            mtx1Expctd = ones(2);
            idx2 = 2;
            mtx2Expctd = 2*ones(4);
            idx3 = 3;
            mtx3Expctd = 3*ones(2);
            idx4 = 4;
            mtx4Expctd = 4*ones(4);            
                       
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.pms = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            
            % Actual values
            step(testCase.pms,mtx1Expctd,idx1);
            step(testCase.pms,mtx2Expctd,idx2);
            step(testCase.pms,mtx3Expctd,idx3);
            step(testCase.pms,mtx4Expctd,idx4);            
            mtx1Actual = step(testCase.pms,[],idx1);
            mtx2Actual = step(testCase.pms,[],idx2);
            mtx3Actual = step(testCase.pms,[],idx3);
            mtx4Actual = step(testCase.pms,[],idx4);            
            
            % Evaluation
            testCase.verifySize(mtx1Actual,size(mtx1Expctd));
            testCase.verifyEqual(mtx1Actual,mtx1Expctd)
            testCase.verifySize(mtx2Actual,size(mtx2Expctd));
            testCase.verifyEqual(mtx2Actual,mtx2Expctd)            
            testCase.verifySize(mtx3Actual,size(mtx3Expctd));
            testCase.verifyEqual(mtx3Actual,mtx3Expctd)
            testCase.verifySize(mtx4Actual,size(mtx4Expctd));
            testCase.verifyEqual(mtx4Actual,mtx4Expctd)            
            
        end          
        
    end
end
