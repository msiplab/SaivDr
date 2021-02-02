classdef NsGenLotDesignerFrqTestCase < matlab.unittest.TestCase
    %NSGENLOTDESIGNEFRQTESTCASE Test case for NsGenLotDesignerFrq
    %
    % SVN identifier:
    % $Id: NsGenLotDesignerFrqTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        dsgnfrq
        display = 'off'
    end
    
    methods (TestMethodTeardown)
        
        function testCase = tearDown(testCase)
            delete(testCase.dsgnfrq);
        end
        
    end
    
    methods (Test)
        
        % Test
        function testCase = testDesignFrqDec22Ord22(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 2 2 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            optfcn = 'fminunc';
            options = optimoptions(optfcn);
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'Algorithm','quasi-newton');
            %
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %             figure(1)
            %             lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %             dispBasisImages(lppufb);
            %             drawnow
            %
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqDec22Ord33(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 3 3 ];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            options = optimoptions('fminunc');
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'Algorithm','quasi-newton');
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %            figure(2)
            %            lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %            dispBasisImages(lppufb);
            %            drawnow
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqDec22Ord44(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 4 4 ];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            options = optimoptions('fminunc');
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'Algorithm','quasi-newton');
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %                         figure(2)
            %                         lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %                         dispBasisImages(lppufb);
            %                         drawnow
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqDec22Ord44Vm2(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 4 4 ];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            transition = 0.2;
            nVm = 2;
            isConstrainedExpctd = true;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            options = optimoptions('fmincon');
            options = optimoptions(options,'Display','off');
            options = optimoptions(options,'Algorithm','active-set');
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %                         figure(2)
            %                         lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %                         dispBasisImages(lppufb);
            %                         drawnow
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqDec22Ord22Tvm(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 2 2 ];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            transition = 0.2;
            nVm = 2;
            phi = atan(alpha)*180/pi;
            isConstrainedExpctd = true;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'TvmAngleInDegree',phi,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            options = optimoptions('fmincon');
            options = optimoptions(options,'Display','off');
            options = optimoptions(options,'Algorithm','active-set');
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %             figure(2)
            %             lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %             dispBasisImages(lppufb);
            %             drawnow
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqOrd22Ga(testCase)
            
            [isGaAvailable, ~] = license('checkout','gads_toolbox');
            if ~isGaAvailable ...
                    || exist('ga','file') ~= 2
                testCase.verifyFail('Skipped\n\tGA is not available. ... ');
                return
            end
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 2 2 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand,...
                'OptimizationFunction',@ga,...
                'MaxIterOfHybridFmin',2);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            angles = get(lppufb,'Angles');
            
            % Optimization
            options = gaoptimset('ga');
            options = gaoptimset(options,'Display',testCase.display);
            options = gaoptimset(options,'UseParallel','never');
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
            testCase.verifyEqual(...
                isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqOrd22GaPct(testCase)
            
            [isGaAvailable,~] = license('checkout','gads_toolbox');
            if ~isGaAvailable ...
                    || exist('ga','file') ~= 2
                testCase.verifyFail('Skipped\n\tGA is not available. ... ');
                return
            end
            [isPctAvailable,~] = license('checkout','distrib_computing_toolbox');
            if ~isPctAvailable || ...
                   ( exist('matlabpool','file') ~= 2 && ...
                    exist('parpool','file') ~= 2 )
                testCase.verifyFail('Skipped\n\t MATLABPOOL is not available. ...');
                return
            end
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 2 2 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand,...
                'OptimizationFunction',@ga,...
                'MaxIterOfHybridFmin',2);
            
            % Cost before optimization
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
            testCase.verifyEqual(...
                isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqOrd22Vm2GaPct(testCase)
            
            [isGaAvailable,~] = license('checkout','gads_toolbox');
            if ~isGaAvailable ...
                    || exist('ga','file') ~= 2
                testCase.verifyFail('Skipped\n\tGA is not available. ... ');
                return
            end
            [isPctAvailable,~] = license('checkout','distrib_computing_toolbox');
            if ~isPctAvailable || ...
                   ( exist('matlabpool','file') ~= 2 && ...
                   exist('parpool','file') ~= 2 )
                testCase.verifyFail('Skipped\n\t MATLABPOOL is not available. ...');
                return
            end
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 2 2 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 2;
            isConstrainedExpctd = true;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand,...
                'OptimizationFunction',@ga,...
                'MaxIterOfHybridFmin',2);
            
            % Cost before optimization
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
            testCase.verifyEqual(...
                isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        
        % Test
        function testCase = testDesignFrqDec22Ord01(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 0 1 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            optfcn = 'fminunc';
            options = optimoptions(optfcn);
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'Algorithm','quasi-newton');
            %
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %             figure(1)
            %             lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %             dispBasisImages(lppufb);
            %             drawnow
            %
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqDec22Ord10(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 1 0 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            optfcn = 'fminunc';
            options = optimoptions(optfcn);
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'Algorithm','quasi-newton');
            %
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %             figure(1)
            %             lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %             dispBasisImages(lppufb);
            %             drawnow
            %
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqDecOrd11(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 1 1 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            optfcn = 'fminunc';
            options = optimoptions(optfcn);
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'Algorithm','quasi-newton');
            %
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %             figure(1)
            %             lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %             dispBasisImages(lppufb);
            %             drawnow
            %
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqDec22Ord02(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 0 2 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            optfcn = 'fminunc';
            options = optimoptions(optfcn);
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'Algorithm','quasi-newton');
            %
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %             figure(1)
            %             lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %             dispBasisImages(lppufb);
            %             drawnow
            %
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqDec22Ord20(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 2 0 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            optfcn = 'fminunc';
            options = optimoptions(optfcn);
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'Algorithm','quasi-newton');
            %
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %             figure(1)
            %             lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %             dispBasisImages(lppufb);
            %             drawnow
            %
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqDec22Ord04(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 0 4 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            optfcn = 'fminunc';
            options = optimoptions(optfcn);
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'Algorithm','quasi-newton');
            %
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %             figure(1)
            %             lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %             dispBasisImages(lppufb);
            %             drawnow
            %
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqDec22Ord40(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 4 0 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            optfcn = 'fminunc';
            options = optimoptions(optfcn);
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'Algorithm','quasi-newton');
            %
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %             figure(1)
            %             lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %             dispBasisImages(lppufb);
            %             drawnow
            %
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqDec22Ord24(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 2 4 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            optfcn = 'fminunc';
            options = optimoptions(optfcn);
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'Algorithm','quasi-newton');
            %
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %             figure(1)
            %             lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %             dispBasisImages(lppufb);
            %             drawnow
            %
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = costPre - costPst;
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            diff = abs(costExpctd - costPst)./abs(costExpctd);
            testCase.verifyEqual(costPst,costExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testCase = testDesignFrqDec22Ord42(testCase)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            nDecs = [ 2 2 ];
            nOrds = [ 4 2 ];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.2;
            nVm = 1;
            isConstrainedExpctd = false;
            
            % Stopband specification
            import saivdr.dictionary.nsgenlotx.design.*
            import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
            import saivdr.dictionary.utility.Subband
            spp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(spp,'Transition',transition);
            rosLyLx = step(spp,nPoints,Subband.LyLx);
            rosHyLx = step(spp,nPoints,Subband.HyLx);
            rosLyHx = step(spp,nPoints,Subband.LyHx);
            rosHyHx = step(spp,nPoints,Subband.HyHx);
            specBand(:,:,Subband.LyLx) = rosLyLx;
            specBand(:,:,Subband.HyLx) = rosHyLx;
            specBand(:,:,Subband.LyHx) = rosLyHx;
            specBand(:,:,Subband.HyHx) = rosHyHx;
            psbe = PassBandErrorStopBandEnergy(...
                'AmplitudeSpecs', specBand);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','AnalysisFilters');
            testCase.dsgnfrq = NsGenLotDesignerFrq(...
                'AmplitudeSpecs',specBand);
            
            % Cost before optimization
            costPre = step(psbe,lppufb);
            
            % Optimization
            testCase.verifyEqual(isConstrained(testCase.dsgnfrq,lppufb),...
                isConstrainedExpctd);
            optfcn = 'fminunc';
            options = optimoptions(optfcn);
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'Algorithm','quasi-newton');
            %
            [lppufb, costPst, ~] = step(testCase.dsgnfrq,lppufb,options);
            costExpctd= step(psbe,lppufb);
            
            % Show basis images
            %             figure(1)
            %             lppufb = getLpPuFb2d(testCase.dsgnfrq);
            %             dispBasisImages(lppufb);
            %             drawnow
            %
            
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
