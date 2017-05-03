classdef PassBandErrorStopBandEnergyTestCase < matlab.unittest.TestCase
    %PASSBANDERRORSTOPBANDENERGYTESTCASE Test case for PassBandErrrorStopBandEnergy
    %
    % SVN identifier:
    % $Id: PassBandErrorStopBandEnergyTestCase.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
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
    % http://msiplab.eng.niigata-u.ac.jp/    
    %
    
    properties
        psbe
    end
    
    methods (TestMethodTeardown)
        function deteleObject(testCase)
            delete(testCase.psbe);
        end
    end
    
    methods (Test)
        
        % Test for default construction
        function testDefaultRand16x16(testCase)
            
            % Parameters
            nPoints = [16 16];
            nChs = 5;
            
            % Synthesis filters for test
            filtImp = zeros(nPoints(1),nPoints(2),nChs);
            specBand = zeros(nPoints(1),nPoints(2),nChs);
            for idx = 1:nChs
                ros = round(rand(nPoints));
                filtImp(:,:,idx) = ifft2(ifftshift(ros));
                specBand(:,:,idx) = 2*ros-1;
            end
            
            % Expected values
            energyExpctd = 0.0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.design.*
            testCase.psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Actual values
            energyActual = step(testCase.psbe,filtImp);
            
            % Evaluation
            testCase.verifyEqual(energyActual,energyExpctd,'AbsTol',1e-15);
            
        end
        
        function testDefaultZeros16x16(testCase)
            
            % Parameters
            nPoints = [16 16];
            nChs = 5;
            
            % Synthesis filters for test
            filtImp = zeros(nPoints(1),nPoints(2),nChs);
            specBand = zeros(nPoints(1),nPoints(2),nChs);
            for idx = 1:nChs
                ros = zeros(nPoints);
                filtImp(:,:,idx) = ifft2(ifftshift(ones(nPoints)));
                specBand(:,:,idx) = 2*ros-1;
            end
            
            % Expected values
            energyExpctd = nChs*prod(nPoints);
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.design.*
            testCase.psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Actual values
            energyActual = step(testCase.psbe,filtImp);
            
            % Evaluation
            testCase.verifyEqual(energyActual,energyExpctd,'AbsTol',1e-15);
            
        end
        
        % Test for default construction
        function testDefaoult64x64(testCase)
            
            % Parameters
            nPoints = [64 64];
            nChs = 5;
            
            % Synthesis filters for test
            filtImp = zeros(nPoints(1),nPoints(2),nChs);            
            specBand = zeros(nPoints(1),nPoints(2),nChs);
            for idx = 1:nChs
                ros = round(rand(nPoints));
                filtImp(:,:,idx) = ifft2(ifftshift(ros));
                specBand(:,:,idx) = 2*ros-1;
            end
            
            % Expected values
            energyExpctd = 0.0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.design.*
            testCase.psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Actual values
            energyActual = step(testCase.psbe,filtImp);
            
            % Evaluation
            testCase.verifyEqual(energyActual,energyExpctd,'AbsTol',1e-15);
            
        end
        
        % Test for default construction
        function testGetCostAt(testCase)
            
            % Parameters
            nPoints = [16 16];
            nChs = 5;
            
            % Synthesis filters for test
            filtImp = cell(nChs,1);
            specBand = zeros(nPoints(1),nPoints(2),nChs);
            for idx = 1:nChs
                ros = round(rand(nPoints));
                filtImp{idx} = ifft2(ifftshift(ros));
                specBand(:,:,idx) = 2*ros-1;
            end
                        
            % Expected values
            energyExpctd = 0.0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.design.*
            testCase.psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs',specBand,...
                'EvaluationMode','Individual');
            import matlab.unittest.constraints.IsGreaterThan;
            for idxA = 1:nChs
                for idxB = 1:nChs
                    energyActual = step(testCase.psbe,filtImp{idxA},idxB,[]);
                    % Evaluation
                    if idxA==idxB
                        testCase.verifyEqual(energyActual,energyExpctd,'AbsTol',1e-15);
                    else
                        testCase.verifyThat(energyActual,IsGreaterThan(1e-15));
                    end
                end
            end
        end
        
        % Test for default construction
        function testGetCostAtWithNullAssignment(testCase)
            
            % Parameters
            nPoints = [16 16];
            
            % Synthesis filters for test
            filtImp{1} = randn(nPoints);
            
            % PassBand specification
            specBand = zeros(nPoints);
            
            % Expected values
            energyExpctd = 0.0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.design.*
            testCase.psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs',specBand,...
                'EvaluationMode','Individual');
            
            % Actual values
            energyActual = step(testCase.psbe,filtImp,1,[]);
            
            % Evaluation

            testCase.verifyEqual(energyActual,energyExpctd,'AbsTol',1e-15);
            
        end

    end
end
