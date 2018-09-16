classdef IterativeSparseApproximationTestCase < matlab.unittest.TestCase
    % ITERATIVESPARSERESTORATIONTESTCASE Test case for IterativeSparseApproximation
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2018, Shogo MURAMATSU
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
        target
    end
        
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.target);
        end
    end
    
    methods (Test)
        
        function testConstruction(testCase)
            
            % Expected Value
            algorithmExpctd = [];
            
            % Instantiation
            import saivdr.sparserep.IterativeSparseApproximater
            testCase.target = IterativeSparseApproximater();
            
            % Actual Value
            algorithmActual = testCase.target.Algorithm;
            
            % Evaluation
            testCase.verifyEqual(algorithmActual,algorithmExpctd);
        end
        
        function testStepIsta(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            height = 32;
            width  = 32;
            wSigma = 1e-3;
            srcImg = rand(height,width);
            obsImg = srcImg + wSigma*randn(height,width);
            
            % Expected values
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            lppufb.Angles = angs;
            
            % Instantiation of target class
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');            
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            itermethod = IstaSystem();
            testCase.target = IterativeSparseApproximater(...
                'Algorithm',itermethod,...
                'Dictionary', {synthesizer,analyzer});
            
            % Actual values
            resActual = testCase.target.step(obsImg);
            
            % Evaluation
            %import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(resActual, size(srcImg));
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
            %}
        end

    end
    
end
