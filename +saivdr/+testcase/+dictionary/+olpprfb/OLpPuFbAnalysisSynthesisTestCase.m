classdef OLpPuFbAnalysisSynthesisTestCase < matlab.unittest.TestCase
    %OLpPuFbAnalysis1dSystemTESTCASE Test case for OLpPuFbAnalysis1dSystem
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015, Shogo MURAMATSU
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
        function testDec2Ch11Ord0Level1(testCase)
            
            nDecs = 2;
            nChs  = [ 1 1 ];
            nOrds = 0;
            nLen = 32;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = OLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = OLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);

            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq);
            recSeq = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcSeq(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recSeq,[ 1 nLen ]);
            diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
            testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end

        % Test
        function testDec4Ch22Ord2Level1(testCase)
            
            nDecs = 4;
            nChs  = [ 2 2 ];
            nOrds = 2;
            nLen = 32;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.olpprfb.OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.olpprfb.OLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.olpprfb.OLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq);
            recSeq = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcSeq(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recSeq,[ 1 nLen ]);
            diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
            testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec2Ch22Ord2Level2(testCase)
            
            nDecs = 2;
            nChs  = [ 2 2 ];
            nOrds = 2;
            nLen = 32;
            nLevels = 2;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.olpprfb.OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.olpprfb.OLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.olpprfb.OLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq);
            recSeq = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcSeq(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recSeq,[ 1 nLen ]);
            diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
            testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end    
        
        % Test
        function testDec2Ch44Ord4Level3(testCase)
            
            nDecs = 2;
            nChs  = [ 4 4 ];
            nOrds = 4;
            nLen = 32;
            nLevels = 2;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.olpprfb.OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.olpprfb.OLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.olpprfb.OLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq);
            recSeq = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcSeq(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recSeq,[ 1 nLen ]);
            diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
            testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end            
        
        % Test
        function testDec2Ch32Ord0Level1(testCase)
            
            nDecs = 2;
            nChs  = [ 3 2 ];
            nOrds = 0;
            nLen = 32;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = OLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = OLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq);
            recSeq = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcSeq(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recSeq,[ 1 nLen ]);
            diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
            testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec2Ch32Ord2Level1(testCase)
            
            nDecs = 2;
            nChs  = [ 3 2 ];
            nOrds = 2;
            nLen = 32;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.olpprfb.OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.olpprfb.OLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.olpprfb.OLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq);
            recSeq = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcSeq(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recSeq,[ 1 nLen ]);
            diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
            testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec2Ch32Ord2Level2(testCase)
            
            nDecs = 2;
            nChs  = [ 3 2 ];
            nOrds = 2;
            nLen = 32;
            nLevels = 2;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.olpprfb.OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.olpprfb.OLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.olpprfb.OLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq);
            recSeq = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcSeq(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recSeq,[ 1 nLen ]);
            diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
            testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec2Ch54Ord4Level3(testCase)
            
            nDecs = 2;
            nChs  = [ 5 4 ];
            nOrds = 4;
            nLen = 32;
            nLevels = 2;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.olpprfb.OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.olpprfb.OLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.olpprfb.OLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq);
            recSeq = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcSeq(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recSeq,[ 1 nLen ]);
            diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
            testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec2Ch23Ord0Level1(testCase)
            
            nDecs = 2;
            nChs  = [ 2 3 ];
            nOrds = 0;
            nLen = 32;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = OLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = OLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);            
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq);
            recSeq = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcSeq(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recSeq,[ 1 nLen ]);
            diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
            testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec2Ch23Ord2Level1(testCase)
            
            nDecs = 2;
            nChs  = [ 2 3 ];
            nOrds = 2;
            nLen = 32;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.olpprfb.OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.olpprfb.OLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.olpprfb.OLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq);
            recSeq = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcSeq(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recSeq,[ 1 nLen ]);
            diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
            testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec2Ch23Ord2Level2(testCase)
            
            nDecs = 2;
            nChs  = [ 2 3 ];
            nOrds = 2;
            nLen = 32;
            nLevels = 2;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.olpprfb.OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.olpprfb.OLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.olpprfb.OLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq);
            recSeq = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcSeq(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recSeq,[ 1 nLen ]);
            diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
            testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec2Ch45Ord4Level3(testCase)
            
            nDecs = 2;
            nChs  = [ 4 5 ];
            nOrds = 4;
            nLen = 32;
            nLevels = 2;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            lppufb = saivdr.dictionary.olpprfb.OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.olpprfb.OLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.olpprfb.OLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq);
            recSeq = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcSeq(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recSeq,[ 1 nLen ]);
            diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
            testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
                sprintf('diff = %g',diff));
        end
                
    end
    
end
