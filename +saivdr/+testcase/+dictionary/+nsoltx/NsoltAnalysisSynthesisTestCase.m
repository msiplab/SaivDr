classdef NsoltAnalysisSynthesisTestCase < matlab.unittest.TestCase
    %NsoltAnalysis2dSystemTESTCASE Test case for NsoltAnalysis2dSystem
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2020, Shogo MURAMATSU
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
            nChs  = [ 2 2 ];
            nOrds = [ 0 0 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);

            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.nsoltx.NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.nsoltx.NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.nsoltx.NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.nsoltx.NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.nsoltx.NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.nsoltx.NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 3 2 ];
            nOrds = [ 0 0 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 3 2 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.nsoltx.NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.nsoltx.NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 3 2 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.nsoltx.NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.nsoltx.NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec22Ch54Ord44Level3(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 4 4 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.nsoltx.NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.nsoltx.NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 4 4 ];
            nOrds = [ 0 0 0 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 4 4 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 4 4 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 2;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 6 6 ];
            nOrds = [ 4 4 4 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 5 4 ];
            nOrds = [ 0 0 0 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 5 4 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 5 4 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 2;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
        function testDec222Ch64Ord444Level3(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 4 ];
            nOrds = [ 4 4 4 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
        function testDec22Ch23Ord00Level1(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 2 3 ];
            nOrds = [ 0 0 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
        function testDec22Ch23Ord22Level1(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 2 3 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.nsoltx.NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.nsoltx.NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
        function testDec22Ch23Ord22Level2(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 2 3 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.nsoltx.NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.nsoltx.NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec22Ch45Ord44Level3(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 4 5 ];
            nOrds = [ 4 4 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.nsoltx.NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.nsoltx.NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width ]);
            diff = max(abs(recImg(:)-srcImg(:))./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'RelTol',1e-9,...
                sprintf('diff = %g',diff));
        end
                
       
        % Test
        function testDec222Ch45Ord000Level1(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 4 5 ];
            nOrds = [ 0 0 0 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
        function testDec222Ch45Ord222Level1(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 4 5 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
        function testDec222Ch45Ord222Level2(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 4 5 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 2;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
        function testDec222Ch46Ord444Level3(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 4 6 ];
            nOrds = [ 4 4 4 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));%./abs(srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-7,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec12Ch22Ord22Level3(testCase)
            
            nDecs = [ 1 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            height = 16;
            width  = 32;
            nLevels = 3;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width  = 32;
            depth  = 64;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
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

