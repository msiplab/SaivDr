classdef AnalysisSynthesisTestCase < matlab.unittest.TestCase
    %ANALYSISSYNTHESISTESTCASE Test case for AnalysisSynthesis
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015-2021, Shogo MURAMATSU
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
    
     properties (TestParameter)       
        nlevels = { 1, 3, 5 };
    end

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

            % Parameters
            nDecs = [ 2 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 0 0 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
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
        function testDec22Ch22Ord22Level1(testCase)

            % Parameters            
            nDecs = [ 2 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
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
        function testDec22Ch22Ord22Level2(testCase)
            
            % Parameters            
            nDecs = [ 2 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
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
        function testDec22Ch44Ord44Level3(testCase)
            
            % Parameters            
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            height = 32;
            width  = 64;
            nLevels = 3;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
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
        function testDec22Ch32Ord00Level1(testCase)
            
            % Parameters            
            nDecs = [ 2 2 ];
            nChs  = [ 3 2 ];
            nOrds = [ 0 0 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
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
        function testDec22Ch32Ord22Level1(testCase)
        
            % Parameters            
            nDecs = [ 2 2 ];
            nChs  = [ 3 2 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
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
        function testDec22Ch32Ord22Level2(testCase)
            
           % Parameters            
            nDecs = [ 2 2 ];
            nChs  = [ 3 2 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
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
            
            % Parameters
            nDecs = [ 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 4 4 ];
            height = 32;
            width  = 64;
            nLevels = 3;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
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
        function testSynthesizerDec22Ch54Ord44Level3(testCase)
            
            % Parameters
            nDecs = [ 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 4 4 ];
            height = 32;
            width  = 64;
            nLevels = 3;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'ParameterMatrixSet');            
            testCase.analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);

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
        function testAnalyzerDec22Ch54Ord44Level3(testCase)
            
            % Parameters
            nDecs = [ 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 4 4 ];
            height = 32;
            width  = 64;
            nLevels = 3;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'ParameterMatrixSet');            
            testCase.synthesizer   = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Circular');
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
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
        function testDec222Ch44Ord000Level1(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 0 0 0 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end

        % Test
        function testDec222Ch44Ord222Level1(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 2 2 2 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch44Ord222Level2(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 2 2 2 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 2;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch66Ord444Level3(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 6 ];
            nOrds = [ 2 2 2 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch54Ord000Level1(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 0 0 0 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch54Ord222Level1(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 2 2 2 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch54Ord222Level2(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 2 2 2 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 2;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch64Ord444Level3(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 4 ];
            nOrds = [ 4 4 4 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testSynthesizerDec222Ch64Ord444Level3(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 4 ];
            nOrds = [ 4 4 4 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'ParameterMatrixSet');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testAnalyzerDec222Ch64Ord444Level3(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 4 ];
            nOrds = [ 4 4 4 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'ParameterMatrixSet');             
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Circular');
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end

        % Test
        function testDec22Ch22Ord00Level1Freq(testCase)

            % Parameters
            nDecs = [ 2 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 0 0 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testDec22Ch22Ord22Level1Freq(testCase)

            % Parameters            
            nDecs = [ 2 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testDec22Ch22Ord22Level2Freq(testCase)
            
            % Parameters            
            nDecs = [ 2 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testDec22Ch44Ord44Level3Freq(testCase)
            
            % Parameters            
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            height = 80;
            width  = 96;
            nLevels = 3;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testDec22Ch32Ord00Level1Freq(testCase)
            
            % Parameters            
            nDecs = [ 2 2 ];
            nChs  = [ 3 2 ];
            nOrds = [ 0 0 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testDec22Ch32Ord22Level1Freq(testCase)
        
            % Parameters            
            nDecs = [ 2 2 ];
            nChs  = [ 3 2 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testDec22Ch32Ord22Level2Freq(testCase)
            
           % Parameters            
            nDecs = [ 2 2 ];
            nChs  = [ 3 2 ];
            nOrds = [ 2 2 ];
            height = 32;
            width  = 64;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testDec22Ch54Ord44Level3Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 4 4 ];
            height = 80;
            width  = 96;
            nLevels = 3;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testSynthesizerDec22Ch54Ord44Level3Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 4 4 ];
            height = 32;
            width  = 64;
            nLevels = 3;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'ParameterMatrixSet');            
            testCase.analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');

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
        function testAnalyzerDec22Ch54Ord44Level3Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 4 4 ];
            height = 80;
            width  = 96;
            nLevels = 3;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'ParameterMatrixSet');            
            testCase.synthesizer   = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Circular');
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testDec222Ch44Ord000Level1Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 0 0 0 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end

        % Test
        function testDec222Ch44Ord222Level1Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 2 2 2 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch44Ord222Level2Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 2 2 2 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 2;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch66Ord444Level3Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 6 ];
            nOrds = [ 2 2 2 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch54Ord000Level1Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 0 0 0 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');    
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch54Ord222Level1Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 2 2 2 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch54Ord222Level2Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 2 2 2 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 2;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec222Ch64Ord444Level3Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 4 ];
            nOrds = [ 4 4 4 ];
            height = 80;
            width  = 96;
            depth  = 104;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testSynthesizerDec222Ch64Ord444Level3Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 4 ];
            nOrds = [ 4 4 4 ];
            height = 32;
            width  = 64;
            depth  = 48;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'ParameterMatrixSet');
            testCase.analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testAnalyzerDec222Ch64Ord444Level3Freq(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 4 ];
            nOrds = [ 4 4 4 ];
            height = 80;
            width  = 96;
            depth  = 104;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'ParameterMatrixSet');             
            testCase.synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Circular');
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testClone2d(testCase)
            
            % Parameters
            nDecs = [ 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 4 4 ];
            height = 80;
            width  = 96;
            nLevels = 3;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);

            % Clone
            cloneSynthesizer = clone(testCase.synthesizer);
            cloneAnalyzer = clone(testCase.analyzer);
            
            % Step
            [ coefs, scales ] = step(cloneAnalyzer,srcImg);
            recImg = step(cloneSynthesizer,coefs,scales);
            
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
        function testClone3d(testCase)
            
            % Parameters
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 4 ];
            nOrds = [ 4 4 4 ];
            height = 80;
            width  = 96;
            depth  = 104;
            nLevels = 3;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Clone
            cloneSynthesizer = testCase.synthesizer;
            cloneAnalyzer    = testCase.analyzer;
            
            % Step
            [ coefs, scales ] = step(cloneAnalyzer,srcImg);
            recImg = step(cloneSynthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec44Ch1212Ord22Level1(testCase)

            % Parameters
            nDecs = [ 4 4 ];
            nChs  = [ 12 12 ];
            nOrds = [ 2 2 ];
            height = 12*4;
            width  = 16*4;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters  = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Spatial');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Spatial',...
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
        function testDec44Ch1212Ord22Level1Freq(testCase)

            % Parameters
            nDecs = [ 4 4 ];
            nChs  = [ 12 12 ];
            nOrds = [ 2 2 ];
            height = 12*4;
            width  = 16*4;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters  = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testDec33Ch55Ord22Level1(testCase)

            % Parameters
            nDecs = [ 3 3 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 ];
            height = 12*3;
            width  = 16*3;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters  = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Spatial');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Spatial',...
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
        function testDec33Ch55Ord22Level1Freq(testCase)

            % Parameters
            nDecs = [ 3 3 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 ];
            height = 12*3;
            width  = 16*3;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters  = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testDec44Ch1212Ord22Level2(testCase)

            % Parameters
            nDecs = [ 4 4 ];
            nChs  = [ 12 12 ];
            nOrds = [ 2 2 ];
            height = 12*4^2;
            width  = 16*4^2;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters  = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Spatial');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Spatial',...
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
        function testDec44Ch1212Ord22Level2Freq(testCase)

            % Parameters
            nDecs = [ 4 4 ];
            nChs  = [ 12 12 ];
            nOrds = [ 2 2 ];
            height = 12*4^2;
            width  = 16*4^2;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters  = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testDec33Ch55Ord22Level2(testCase)

            % Parameters
            nDecs = [ 3 3 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 ];
            height = 12*3^2;
            width  = 16*3^2;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters  = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Spatial');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Spatial',...
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
        function testDec33Ch55Ord22Level2Freq(testCase)

            % Parameters
            nDecs = [ 3 3 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 ];
            height = 12*3^2;
            width  = 16*3^2;
            nLevels = 2;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters  = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
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
        function testDec33Ch55Ord22Level2FreqUseGpuFalse(testCase)

            % Parameters
            nDecs = [ 3 3 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 ];
            height = 12*3^2;
            width  = 16*3^2;
            nLevels = 2;
            useGpu = false;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)            
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters  = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*                            
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency',...
                'UseGpu',useGpu);
            testCase.analyzer    = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'UseGpu',useGpu,...
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
        function testDec333Ch1414Ord222Level1(testCase)
            
            % Parameters
            nDecs = [ 3 3 3 ];
            nChs  = [ 14 14 ];
            nOrds = [ 2 2 2 ];
            height = 8*3;
            width  = 12*3;
            depth  = 16*3;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec333Ch1414Ord222Level1Freq(testCase)
            
            % Parameters
            nDecs = [ 3 3 3 ];
            nChs  = [ 14 14 ];
            nOrds = [ 2 2 2 ];
            height = 8*3;
            width  = 12*3;
            depth  = 16*3;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec333Ch1414Ord222Level2(testCase)
            
            % Parameters
            nDecs = [ 3 3 3 ];
            nChs  = [ 14 14 ];
            nOrds = [ 2 2 2 ];
            height = 8*3^2;
            width  = 12*3^2;
            depth  = 16*3^2;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec333Ch1414Ord222Level2Freq(testCase)
            
            % Parameters
            nDecs = [ 3 3 3 ];
            nChs  = [ 14 14 ];
            nOrds = [ 2 2 2 ];
            height = 8*3^2;
            width  = 12*3^2;
            depth  = 16*3^2;
            nLevels = 2;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
                
        % Test
        function testDec234Ch1414Ord222Level2(testCase)
            
            % Parameters
            nDecs = [ 2 3 4 ];
            nChs  = [ 14 14 ];
            nOrds = [ 2 2 2 ];
            height = 8*2^2;
            width  = 12*3^2;
            depth  = 16*4^2;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
        % Test
        function testDec234Ch1414Ord222Level2Freq(testCase)
            
            % Parameters
            nDecs = [ 2 3 4 ];
            nChs  = [ 14 14 ];
            nOrds = [ 2 2 2 ];
            height = 8*2^2;
            width  = 12*3^2;
            depth  = 16*4^2;
            nLevels = 2;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            vm = 1;
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm);
            release(lppufb)
            set(lppufb,'OutputMode', 'SynthesisFilters');
            synthesisFilters = step(lppufb,[],[]);
            release(lppufb)
            set(lppufb,'OutputMode', 'AnalysisFilters');
            analysisFilters = step(lppufb,[],[]);
            
            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            testCase.analyzer    = Analysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
                
        % Test
        function test2plus1dSystem(testCase,nlevels)
            
            nsubrows_ = 32;
            nsubcols_ = 16;
            nsublays_ = 1;
            ndecsY = 2;
            ndecsX = 2;
            ndecsZ = 4;
            
             % Parameters
            import saivdr.dictionary.utility.Direction
            nLevelsXY = nlevels;
            nDecs = [ ndecsY ndecsX ndecsZ ];                        
            height = nsubrows_ * (ndecsY^nLevelsXY);
            width = nsubcols_ * (ndecsX^nLevelsXY);
            depth = nsublays_ * ndecsZ;
            srcImg = rand(height,width,depth);

            % Analysis filters in XY
            analysisFiltersInXY(:,:,1) = [ 1 1; 1 1 ]/2;
            analysisFiltersInXY(:,:,2) = [ 1 -1; 1 -1 ]/2;
            analysisFiltersInXY(:,:,3) = [ 1 1; -1 -1 ]/2;
            analysisFiltersInXY(:,:,4) = [ 1 -1; -1 1 ]/2;
            % Synthesis filters in XY
            synthesisFiltersInXY(:,:,1) = [ 1 1; 1 1 ]/2;
            synthesisFiltersInXY(:,:,2) = [ -1 1; -1 1 ]/2;
            synthesisFiltersInXY(:,:,3) = [ -1 -1; 1 1 ]/2;
            synthesisFiltersInXY(:,:,4) = [ 1 -1; -1 1 ]/2;
            
            % Analysis filters in Z
            analysisFiltersInZ = flipud(dctmtx(ndecsZ).');
            % Synthesis filters in Z
            synthesisFiltersInZ = dctmtx(ndecsZ).';

            % Instantiation of targets
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFiltersInXY',analysisFiltersInXY,...
                'AnalysisFiltersInZ',analysisFiltersInZ,...
                'NumberOfLevelsInXY',nLevelsXY);
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFiltersInXY',synthesisFiltersInXY,...
                'SynthesisFiltersInZ',synthesisFiltersInZ);
  
            % Step
            [ coefs, scales ] = step(testCase.analyzer,srcImg);
            recImg = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = abs(norm(coefs(:))-norm(srcImg(:)));
            testCase.verifyEqual(norm(coefs(:)),norm(srcImg(:)),...
                'AbsTol',1e-10,sprintf('diff = %g',diff));
            testCase.verifySize(recImg,[ height width depth ]);
            diff = max(abs(recImg(:)-srcImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-14,...
                sprintf('diff = %g',diff));
        end
        
    end
    
end
