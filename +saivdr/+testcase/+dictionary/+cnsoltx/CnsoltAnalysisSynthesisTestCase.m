classdef CnsoltAnalysisSynthesisTestCase < matlab.unittest.TestCase
    %CnsoltAnalysis2dSystemTESTCASE Test case for CnsoltAnalysis2dSystem
    %
    % SVN identifier:
    % $Id: CnsoltAnalysisSynthesisTestCase.m 868 2015-11-25 02:33:11Z sho $
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
    %
    properties
        analyzer
        synthesizer
    end
    
    methods (TestMethodTeardown)
        function deteleObject(testCase)
            delete(testCase.analyzer);
            delete(testCase.synthesizer);
        end
    end
    
    methods (Test)

        % Test
        function testDec22Ch22Ord00Level1(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = 4;
            nOrds = [ 0 0 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width) + 1i*rand(height,width);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            symmetry = 2*pi*rand(1,sum(nChs));
            set(lppufb,'Symmetry',symmetry);
            
            testCase.synthesizer = CnsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CnsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');

            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec22Ch22Ord22Level1(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = 4;
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width) + 1i*rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.cnsoltx.CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            symmetry = 2*pi*rand(1,sum(nChs));
            set(lppufb,'Symmetry',symmetry);
            
            testCase.synthesizer = saivdr.dictionary.cnsoltx.CnsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.cnsoltx.CnsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec22Ch22Ord22Level2(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = 4;
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width) + 1i*rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.cnsoltx.CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            symmetry = 2*pi*rand(1,sum(nChs));
            set(lppufb,'Symmetry',symmetry);
            
            testCase.synthesizer = saivdr.dictionary.cnsoltx.CnsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.cnsoltx.CnsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end    
        
        % Test
        function testDec22Ch44Ord44Level3(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = 8;
            nOrds = [ 4 4 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width) + 1i*rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.cnsoltx.CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            symmetry = 2*pi*rand(1,sum(nChs));
            set(lppufb,'Symmetry',symmetry);
            
            testCase.synthesizer = saivdr.dictionary.cnsoltx.CnsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.cnsoltx.CnsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-8,...
                sprintf('diff = %g',diff));
        end            
        
        % Test
        function testDec22Ch32Ord00Level1(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = 5;
            nOrds = [ 0 0 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width) + 1i*rand(height,width);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            symmetry = 2*pi*rand(1,sum(nChs));
            set(lppufb,'Symmetry',symmetry);
            
            testCase.synthesizer = CnsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CnsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec22Ch32Ord22Level1(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = 5;
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width) + 1i*rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.cnsoltx.CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            symmetry = 2*pi*rand(1,sum(nChs));
            set(lppufb,'Symmetry',symmetry);
            
            testCase.synthesizer = saivdr.dictionary.cnsoltx.CnsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.cnsoltx.CnsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec22Ch32Ord22Level2(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = 5;
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width) + 1i*rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.cnsoltx.CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            symmetry = 2*pi*rand(1,sum(nChs));
            set(lppufb,'Symmetry',symmetry);
            
            testCase.synthesizer = saivdr.dictionary.cnsoltx.CnsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.cnsoltx.CnsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec22Ch54Ord44Level3(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = 9;
            nOrds = [ 4 4 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width) + 1i*rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.cnsoltx.CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            symmetry = 2*pi*rand(1,sum(nChs));
            set(lppufb,'Symmetry',symmetry);
            
            testCase.synthesizer = saivdr.dictionary.cnsoltx.CnsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.cnsoltx.CnsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch44Ord000Level1(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = 8;
            nOrds = [ 0 0 0 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            testCase.synthesizer = CnsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CnsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-9,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch44Ord222Level1(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = 8;
            nOrds = [ 2 2 2 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            testCase.synthesizer = CnsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CnsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-8,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch44Ord222Level2(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = 8;
            nOrds = [ 2 2 2 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 2;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            testCase.synthesizer = CnsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CnsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-9,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch66Ord444Level3(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = 12;
            nOrds = [ 4 4 4 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            testCase.synthesizer = CnsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CnsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-8,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch54Ord000Level1(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = 9;
            nOrds = [ 0 0 0 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            testCase.synthesizer = CnsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CnsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-9,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch54Ord222Level1(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = 9;
            nOrds = [ 2 2 2 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            testCase.synthesizer = CnsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CnsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-9,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch54Ord222Level2(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = 9;
            nOrds = [ 2 2 2 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 2;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            testCase.synthesizer = CnsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CnsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-8,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec12Ch22Ord22Level3(testCase)
            
            nDecs = [ 1 2 ];
            nChs  = 4;
            nOrds = [ 2 2 ];
            height = 16;
            width  = 32;
            nLevels = 3;
            srcImg = rand(height,width) + 1i*rand(height,width);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            testCase.synthesizer = CnsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CnsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-7,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec112Ch22Ord222Level3(testCase)
            
            nDecs = [ 1 1 2 ];
            nChs  = 4;
            nOrds = [ 2 2 2 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            angs = get(lppufb,'Angles');
            angs = 2*pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            testCase.synthesizer = CnsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CnsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg,nLevels);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-7,...
                sprintf('diff = %g',diff));
        end
    end
    
end
