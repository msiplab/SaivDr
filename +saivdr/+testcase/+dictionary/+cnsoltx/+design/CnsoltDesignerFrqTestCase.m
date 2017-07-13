classdef CnsoltDesignerFrqTestCase < matlab.unittest.TestCase
    %NSOLTDESIGNERFRQTESTCASE Test case for AprxErrorWithSparseRep
    %
    % SVN identifier:
    % $Id: CnsoltDesignerFrqTestCase.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2013a
    %
    % Copyright (c) 2013-2015, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    properties
        dsgnfrq
        display = 'off'
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.dsgnfrq);
        end
    end
    
    methods (Test)
        
        % Test for default construction
        function testDesignFrqDec22Ch7Ord44Vm1(testCase)
            
            % Parameters
            nPoints  = [ 16 16 ];
            nDecs = [ 2 2 ];
            nChs  = 7;
            nOrds = [ 4 4 ];
            nVm = 1;

            % Preperation
            import saivdr.dictionary.cnsoltx.design.*
            specBand = zeros(nPoints(1),nPoints(2),nChs);
            for idx = 1:nChs
                ros = round(rand(nPoints));
                specBand(:,:,idx) = 2*ros-1;
            end            
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            import saivdr.dictionary.utility.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            
            % Instantiation of target
            testCase.dsgnfrq = CnsoltDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            options = optimoptions('fminunc');
            options = optimoptions(options,'Algorithm','quasi-newton');
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'MaxIter',2);
            %
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd = step(psbe,lppufb);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
  
        % Test for default construction
        function testDesignFrqDec22Ch8Ord44Vm1Ga(testCase)
            
            [isGaAvailable, ~] = license('checkout','gads_toolbox');
            if ~isGaAvailable ...
                    || exist('ga','file') ~= 2
                testCase.fail(testCase,'Skipped\n\tGA is not available. ... ');
                return
            end            
            
            % Parameters
            nPoints  = [ 16 16 ];
            nDecs = [ 2 2 ];
            nChs  = 8;
            nOrds = [ 4 4 ];
            nVm = 1;

            % Preperation
            import saivdr.dictionary.cnsoltx.design.*
            specBand = zeros(nPoints(1),nPoints(2),nChs);
            for idx = 1:nChs
                ros = round(rand(nPoints));
                specBand(:,:,idx) = 2*ros-1;
            end            
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            import saivdr.dictionary.utility.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            
            % Instantiation of target
            testCase.dsgnfrq = CnsoltDesignerFrq(...
                'AmplitudeSpecs',specBand,...
                'OptimizationFunction',@ga,...
                'MaxIterOfHybridFmin',2);
            
            % Before optimization
            costPre = step(psbe,lppufb);
            angles = get(lppufb,'Angles');
            
            % Optimization
            options = gaoptimset('ga');
            options = gaoptimset(options,'Display',testCase.display);
            options = gaoptimset(options,'UseParallel','always');
            options = gaoptimset(options,'PopulationSize',4);
            options = gaoptimset(options,'EliteCount',2);
            popInitRange = [angles(:)-pi angles(:)+pi].';
            options = gaoptimset(options,'PopInitRange',popInitRange);
            options = gaoptimset(options,'Generations',2);
            options = gaoptimset(options,'StallGenLimit',4);
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd = step(psbe,lppufb);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));            
     
        end
 
        % Test for default construction
        function testDesignFrqDec22Ch8Ord44Vm1GaOptMus(testCase)
            
            [isGaAvailable, ~] = license('checkout','gads_toolbox');
            if ~isGaAvailable ...
                    || exist('ga','file') ~= 2
                testCase.fail(testCase,'Skipped\n\tGA is not available. ... ');
                return
            end
            
            % Parameters
            nPoints  = [ 16 16 ];
            nDecs = [ 2 2 ];
            nChs  = 8;
            nOrds = [ 4 4 ];
            nVm = 1;

            % Preperation
            import saivdr.dictionary.cnsoltx.design.*
            specBand = zeros(nPoints(1),nPoints(2),nChs);
            for idx = 1:nChs
                ros = round(rand(nPoints));
                specBand(:,:,idx) = 2*ros-1;
            end            
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            import saivdr.dictionary.utility.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            
            % Instantiation of target
            testCase.dsgnfrq = CnsoltDesignerFrq(...
                'AmplitudeSpecs',specBand,...
                'OptimizationFunction',@ga,...
                'MaxIterOfHybridFmin',2,...
                'IsOptimizationOfMus',true);
            
            % Before optimization
            costPre = step(psbe,lppufb);
            angles = get(lppufb,'Angles');
            
            % Optimization
            options = gaoptimset('ga');
            options = gaoptimset(options,'Display',testCase.display);
            options = gaoptimset(options,'UseParallel','always');
            options = gaoptimset(options,'PopulationSize',4);
            options = gaoptimset(options,'EliteCount',2);
            popInitRange = [angles(:)-pi angles(:)+pi].';
            options = gaoptimset(options,'PopInitRange',popInitRange);
            options = gaoptimset(options,'Generations',2);
            options = gaoptimset(options,'StallGenLimit',4);
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd = step(psbe,lppufb);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));            
     
        end        
    end
end
