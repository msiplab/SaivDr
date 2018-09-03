classdef UdHaarAnalysisSynthesisTestCase < matlab.unittest.TestCase
    %UDHAARANALYSISSYNTHESISTESTCASE Test case for analysis-synthesis proces
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
   properties (TestParameter)
        dim1  = struct('small',8,'medium',16,'large',32);
        dim2  = struct('small',8,'medium',16,'large',32);
        dim3  = struct('small',8,'medium',16,'large',32);
        dtype  = {  'double', 'single' };
        usegpu = struct( 'true', true, 'false', false);
        level = { 1, 2, 3 };
    end
    

    properties

    end
    
    methods (TestMethodSetup)
        
    end
    
    methods (TestMethodTeardown)

    end
    
    methods (Test)
        
         % Test for default construction
        function testDecRec2(testCase,dim1,dim2,level,dtype, usegpu)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            
            % パラメータ
            height = dim1;
            width  = dim2;
            nLevels = level;
            
            %
            srcImg = rand(height,width,dtype);
            if usegpu
                srcImg = gpuArray(srcImg);
            end
            resExpctd = srcImg;
            import saivdr.dictionary.udhaar.*
            analyzer    = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            synthesizer = UdHaarSynthesis2dSystem();
            
            [coefs,scales] = analyzer.step(srcImg);
            resActual      = synthesizer.step(coefs,scales);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(coefs,'gpuArray')
                testCase.verifyClass(resActual,'gpuArray')
                coefs = gather(coefs);
                srcImg = gather(srcImg);
                resExpctd = gather(resExpctd);
                resActual = gather(resActual);
            end
            testCase.verifyClass(coefs,dtype)
            testCase.verifyClass(resActual,dtype)
            testCase.verifySize(resActual,size(resExpctd));
            if strcmp(dtype,'double')
                tol = 1e-7;
            else
                tol = single(1e-5);
            end
            testCase.verifyEqual(norm(resActual(:)),norm(srcImg(:)),...
                'AbsTol',tol,sprintf('Energy is not preserved.'));
            diff = max(abs(resExpctd(:)-resActual(:)));
            testCase.verifyEqual(resActual,resExpctd,...
                'AbsTol',tol,sprintf('%g',diff));

        end
        
        % Test for default construction
        function testDecRec3(testCase,dim1,dim2,dim3,level,dtype, usegpu)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            
            % パラメータ
            height = dim1;
            width  = dim2;
            depth  = dim3;
            nLevels = level;
            
            %
            srcImg = rand(height,width,depth,dtype);
            if usegpu
                srcImg = gpuArray(srcImg);
            end
            resExpctd = srcImg;
            import saivdr.dictionary.udhaar.*
            analyzer    = UdHaarAnalysis3dSystem('NumberOfLevels',nLevels);
            synthesizer = UdHaarSynthesis3dSystem();
            
            [coefs,scales] = analyzer.step(srcImg);
            resActual      = synthesizer.step(coefs,scales);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(coefs,'gpuArray')
                testCase.verifyClass(resActual,'gpuArray')
                coefs = gather(coefs);
                srcImg = gather(srcImg);
                resExpctd = gather(resExpctd);
                resActual = gather(resActual);
            end
            testCase.verifyClass(coefs,dtype)
            testCase.verifyClass(resActual,dtype)
            testCase.verifySize(resActual,size(resExpctd));
            if strcmp(dtype,'double')
                tol = 1e-7;
            else
                tol = single(1e-5);
            end
            testCase.verifyEqual(norm(resActual(:)),norm(srcImg(:)),...
                'AbsTol',tol,sprintf('Energy is not preserved.'));
            diff = max(abs(resExpctd(:)-resActual(:)));
            testCase.verifyEqual(resActual,resExpctd,...
                'AbsTol',tol,sprintf('%g',diff));

        end

    end
end