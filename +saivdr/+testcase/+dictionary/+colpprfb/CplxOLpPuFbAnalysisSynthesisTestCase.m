classdef CplxOLpPuFbAnalysisSynthesisTestCase < matlab.unittest.TestCase
    %CplxOLpPuFbAnalysis1dSystemTESTCASE Test case for CplxOLpPuFbAnalysis1dSystem
    %
    % SVN identifier:
    % $Id: CplxOLpPuFbAnalysisSynthesisTestCase.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2013b
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
        function testDec2Ch11Ord0Level1(testCase)
            
            nDecs = 2;
            nChs  = [ 1 1 ];
            nOrds = 0;
            nLen = 32;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.colpprfb.*
            lppufb = CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = CplxOLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CplxOLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination');

            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq,nLevels);
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
            lppufb = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq,nLevels);
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
            lppufb = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq,nLevels);
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
            lppufb = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq,nLevels);
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
            import saivdr.dictionary.colpprfb.*
            lppufb = CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = CplxOLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CplxOLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq,nLevels);
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
            lppufb = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq,nLevels);
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
            lppufb = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq,nLevels);
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
            lppufb = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            
            testCase.synthesizer = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq,nLevels);
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
            import saivdr.dictionary.colpprfb.*
            lppufb = CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            testCase.synthesizer = CplxOLpPrFbFactory.createSynthesis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.analyzer    = CplxOLpPrFbFactory.createAnalysis1dSystem(...
                lppufb,'BoundaryOperation','Termination');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcSeq,nLevels);
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
        
%         % Test
%         function testDec2Ch23Ord2Level1(testCase)
%             
%             nDecs = 2;
%             nChs  = [ 2 3 ];
%             nOrds = 2;
%             nLen = 32;
%             nLevels = 1;
%             srcSeq = rand(1,nLen);
%             
%             % Preparation
%             vm = 1;
%             lppufb = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
%                 'DecimationFactor', nDecs,...
%                 'NumberOfChannels', nChs,...
%                 'PolyPhaseOrder', nOrds,...
%                 'NumberOfVanishingMoments', vm,...
%                 'OutputMode', 'ParameterMatrixSet');
%             
%             testCase.synthesizer = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createSynthesis1dSystem(...
%                 lppufb,'BoundaryOperation','Termination');
%             testCase.analyzer    = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createAnalysis1dSystem(...
%                 lppufb,'BoundaryOperation','Termination');
%             
%             % Step
%             [ coefs, scales ] = step(testCase.analyzer,srcSeq,nLevels);
%             recSeq = step(testCase.synthesizer,coefs,scales);
%             
%             % Evaluation
%             diff = abs(norm(coefs(:))-norm(srcSeq(:)));
%             testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
%                 'AbsTol',1e-10,sprintf('diff = %g',diff));
%             testCase.verifySize(recSeq,[ 1 nLen ]);
%             diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
%             testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
%                 sprintf('diff = %g',diff));
%         end
%         
%         % Test
%         function testDec2Ch23Ord2Level2(testCase)
%             
%             nDecs = 2;
%             nChs  = [ 2 3 ];
%             nOrds = 2;
%             nLen = 32;
%             nLevels = 2;
%             srcSeq = rand(1,nLen);
%             
%             % Preparation
%             vm = 1;
%             lppufb = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
%                 'DecimationFactor', nDecs,...
%                 'NumberOfChannels', nChs,...
%                 'PolyPhaseOrder', nOrds,...
%                 'NumberOfVanishingMoments', vm,...
%                 'OutputMode', 'ParameterMatrixSet');
%             
%             testCase.synthesizer = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createSynthesis1dSystem(...
%                 lppufb,'BoundaryOperation','Termination');
%             testCase.analyzer    = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createAnalysis1dSystem(...
%                 lppufb,'BoundaryOperation','Termination');
%             
%             % Step
%             [ coefs, scales ] = step(testCase.analyzer,srcSeq,nLevels);
%             recSeq = step(testCase.synthesizer,coefs,scales);
%             
%             % Evaluation
%             diff = abs(norm(coefs(:))-norm(srcSeq(:)));
%             testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
%                 'AbsTol',1e-10,sprintf('diff = %g',diff));
%             testCase.verifySize(recSeq,[ 1 nLen ]);
%             diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
%             testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
%                 sprintf('diff = %g',diff));
%         end
%         
%         % Test
%         function testDec2Ch45Ord4Level3(testCase)
%             
%             nDecs = 2;
%             nChs  = [ 4 5 ];
%             nOrds = 4;
%             nLen = 32;
%             nLevels = 2;
%             srcSeq = rand(1,nLen);
%             
%             % Preparation
%             vm = 1;
%             lppufb = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
%                 'DecimationFactor', nDecs,...
%                 'NumberOfChannels', nChs,...
%                 'PolyPhaseOrder', nOrds,...
%                 'NumberOfVanishingMoments', vm,...
%                 'OutputMode', 'ParameterMatrixSet');
%             
%             testCase.synthesizer = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createSynthesis1dSystem(...
%                 lppufb,'BoundaryOperation','Termination');
%             testCase.analyzer    = saivdr.dictionary.colpprfb.CplxOLpPrFbFactory.createAnalysis1dSystem(...
%                 lppufb,'BoundaryOperation','Termination');
%             
%             % Step
%             [ coefs, scales ] = step(testCase.analyzer,srcSeq,nLevels);
%             recSeq = step(testCase.synthesizer,coefs,scales);
%             
%             % Evaluation
%             diff = abs(norm(coefs(:))-norm(srcSeq(:)));
%             testCase.verifyEqual(norm(coefs(:)),norm(srcSeq(:)),...
%                 'AbsTol',1e-10,sprintf('diff = %g',diff));
%             testCase.verifySize(recSeq,[ 1 nLen ]);
%             diff = max(abs(recSeq(:)-srcSeq(:))./abs(srcSeq(:)));
%             testCase.verifyEqual(recSeq,srcSeq,'RelTol',1e-10,...
%                 sprintf('diff = %g',diff));
%         end
                
    end
    
end
