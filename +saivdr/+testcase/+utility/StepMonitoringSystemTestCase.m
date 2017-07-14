classdef StepMonitoringSystemTestCase < matlab.unittest.TestCase
    %STEPMONITORINGSYSTEMTESTCASE Test Case for StepMonitoringSystem
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2017, Shogo MURAMATSU
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
        stepmonitor
        testfigure
    end
    
    methods (TestMethodSetup)
        function downloadSsim(~)
            if verLessThan('images','9.0') && ~exist('ssim_index.m','file')
                ssimcode = ...
                    urlread('https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m');
                fid = fopen('ssim_index.m','w');
                fwrite(fid,ssimcode);
                fprintf('ssim_index.m was downloaded.\n');
            end
        end
        
        function createFigure(testCase)
            testCase.testfigure = figure;
        end
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.stepmonitor);
        end
        
        function closeFigure(testCase)
            close(testCase.testfigure)
        end
    end
    
    methods (Static = true)
        function value = ssim_(A,ref)
            if verLessThan('images','9.0')
                value = ssim_index(A,ref);
            else
                value = ssim(A,ref);
            end
        end
    end
    
    methods (Test)
        
        % Test 
        function testMse(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            resImg = round(srcImg*2)/2;
            
            % Expected values
            srcImgInt = int16(255*srcImg);
            resImgInt = int16(255*resImg);
            mseExpctd = sum((srcImgInt(:)-resImgInt(:)).^2)/numel(srcImgInt);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'MaxIter', 1,...
                'IsMSE', true);

            % Actual values
            mseActual = step(testCase.stepmonitor,resImg);
            % Evaluation           
            testCase.assertEqual(mseActual,mseExpctd,'RelTol',1e-15);

        end
        
        % Test 
        function testMseUint8(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = im2uint8(rand(height,width));
            resImg = im2uint8(round(srcImg*2)/2);
            
            % Expected values
            srcImgInt = int16(srcImg);
            resImgInt = int16(resImg);
            mseExpctd = sum((srcImgInt(:)-resImgInt(:)).^2)/numel(srcImgInt);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'MaxIter', 1,...
                'IsMSE', true);
            
            % Actual values
            mseActual = step(testCase.stepmonitor,resImg);
            
            % Evaluation
            testCase.assertEqual(mseActual,mseExpctd,'RelTol',1e-15);
            
        end
        
        % Test 
        function testSuccessiveMses(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            
            % Expected values
            srcImgInt = int16(255*srcImg);
            %
            resImg1 =   round(srcImg*8)/8;
            resImg1Int = int16(255*resImg1);
            mseExpctd(1) = sum((srcImgInt(:)-resImg1Int(:)).^2)/numel(srcImg);
            %
            resImg2 =   round(srcImg*4)/4;
            resImg2Int = int16(255*resImg2);
            mseExpctd(2) = sum((srcImgInt(:)-resImg2Int(:)).^2)/numel(srcImg);
            %
            resImg3 =   round(srcImg*2)/2;
            resImg3Int = int16(255*resImg3);
            mseExpctd(3) = sum((srcImgInt(:)-resImg3Int(:)).^2)/numel(srcImg);
            %
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'MaxIter', 3,...
                'IsVerbose', false, ...
                'IsMSE', true);
            
            % Actual values
            step(testCase.stepmonitor,resImg1);
            step(testCase.stepmonitor,resImg2);
            mseActual = step(testCase.stepmonitor,resImg3);
            
            % Evaluation
            testCase.assertEqual(mseActual,mseExpctd,'RelTol',1e-15);
            
        end
        
        % Test 
        function testPsnr(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            resImg = round(srcImg*2)/2;
            
            % Expected values
            srcImgInt = int16(255*srcImg);
            resImgInt = int16(255*resImg);
            mse = sum((srcImgInt(:)-resImgInt(:)).^2)/numel(srcImg);
            psnrExpctd = 10*log10(255^2/mse);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'MaxIter', 1,...
                'IsPSNR', true);
            
            % Actual values
            psnrActual = step(testCase.stepmonitor,resImg);
            
            % Evaluation
            testCase.assertEqual(psnrActual,psnrExpctd,'RelTol',1e-15);
            
        end
        
        % Test 
        function testSuccessivePsnrs(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            
            % Expected values
            srcImgInt = int16(255*srcImg);
            %
            resImg1 =   round(srcImg*8)/8;
            resImg1Int = int16(255*resImg1);
            psnrExpctd(1) = 10*log10(255^2/...
                (sum((srcImgInt(:)-resImg1Int(:)).^2)/numel(srcImg)));
            %
            resImg2 =   round(srcImg*4)/4;
            resImg2Int = int16(255*resImg2);
            psnrExpctd(2) = 10*log10(255^2/...
                (sum((srcImgInt(:)-resImg2Int(:)).^2)/numel(srcImg)));
            %
            resImg3 =   round(srcImg*2)/2;
            resImg3Int = int16(255*resImg3);
            psnrExpctd(3) = 10*log10(255^2/...
                (sum((srcImgInt(:)-resImg3Int(:)).^2)/numel(srcImg)));
            %
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'MaxIter', 3,...
                'IsVerbose', false,...
                'IsPSNR', true);
            
            % Actual values
            step(testCase.stepmonitor,resImg1);
            step(testCase.stepmonitor,resImg2);
            psnrActual = step(testCase.stepmonitor,resImg3);
            
            % Evaluation
            testCase.assertEqual(psnrActual,psnrExpctd,'RelTol',1e-15);
            
        end
        
        % Test 
        function testPsnrInt(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = im2uint8(rand(height,width));
            resImg = im2uint8(round(srcImg*2)/2);
            
            % Expected values
            srcImgInt = int16(srcImg);
            resImgInt = int16(resImg);
            mse = sum((srcImgInt(:)-resImgInt(:)).^2)/numel(srcImg);
            psnrExpctd = 10*log10(255^2/mse);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'MaxIter', 1,...
                'IsPSNR', true);
            
            % Actual values
            psnrActual = step(testCase.stepmonitor,resImg);
            
            % Evaluation
            testCase.assertEqual(psnrActual,psnrExpctd,'RelTol',1e-15);
            
        end
        
        
        % Test
        function testSsim(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            resImg = round(srcImg*2)/2;
            
            % Expected values
            srcImgInt = uint8(255*srcImg);
            resImgInt = uint8(255*resImg);
            ssimExpctd = testCase.ssim_(resImgInt,srcImgInt);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'MaxIter', 1,...
                'IsSSIM', true);
            
            % Actual values
            ssimActual = step(testCase.stepmonitor,resImg);
            
            % Evaluation
            testCase.assertEqual(ssimActual,ssimExpctd,'AbsTol',1e-2);
            
        end
        
        % Test
        function testSsimInt(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = uint8(255*rand(height,width));
            resImg = uint8(round(double(srcImg)/2)*2);
            
            % Expected values
            ssimExpctd = testCase.ssim_(resImg,srcImg);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'MaxIter', 1,...
                'IsSSIM', true);
            
            % Actual values
            ssimActual = step(testCase.stepmonitor,resImg);
            
            % Evaluation
            testCase.assertEqual(ssimActual,ssimExpctd,'AbsTol',1e-2);
            
        end
              
        % Test
        function testSuccessiveSsims(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            
            % Expected values
            srcImgInt = im2uint8(srcImg);
            %
            resImg1 =   round(srcImg*8)/8;
            resImg1Int = im2uint8(resImg1);
            ssimExpctd(1) = testCase.ssim_(resImg1Int,srcImgInt);
            %
            resImg2 =   round(srcImg*4)/4;
            resImg2Int = im2uint8(resImg2);
            ssimExpctd(2) = testCase.ssim_(resImg2Int,srcImgInt);
            %
            resImg3 =   round(srcImg*2)/2;
            resImg3Int = im2uint8(resImg3);
            ssimExpctd(3) = testCase.ssim_(resImg3Int,srcImgInt);
            %
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'MaxIter', 3,...
                'IsVerbose', false, ...
                'IsSSIM', true);
            
            % Actual values
            step(testCase.stepmonitor,resImg1);
            step(testCase.stepmonitor,resImg2);
            ssimActual = step(testCase.stepmonitor,resImg3);
            
            % Evaluation
            testCase.assertEqual(ssimActual,ssimExpctd,'AbsTol',1e-2);
            
        end
        
        % Test
        function testSuccessiveMsesPsnrsSsims(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            
            % Expected values
            srcImgInt = im2uint8(srcImg);
            %
            resImg1 =   round(srcImg*8)/8;
            resImg1Int = im2uint8(resImg1);
            mseExpctd(1) = sum((int16(srcImgInt(:))-int16(resImg1Int(:))).^2)/numel(srcImg);            
            psnrExpctd(1) = 10*log10(255^2/...
                (sum((int16(srcImgInt(:))-int16(resImg1Int(:))).^2)/numel(srcImg)));            
            ssimExpctd(1) = testCase.ssim_(srcImgInt,resImg1Int);
            %
            resImg2 =   round(srcImg*4)/4;
            resImg2Int = im2uint8(resImg2);
            mseExpctd(2) = sum((int16(srcImgInt(:))-int16(resImg2Int(:))).^2)/numel(srcImg);
            psnrExpctd(2) = 10*log10(255^2/...
                (sum((int16(srcImgInt(:))-int16(resImg2Int(:))).^2)/numel(srcImg)));            
            ssimExpctd(2) = testCase.ssim_(srcImgInt,resImg2Int);
            %
            resImg3 =   round(srcImg*2)/2;
            resImg3Int = im2uint8(resImg3);
            mseExpctd(3) = sum((int16(srcImgInt(:))-int16(resImg3Int(:))).^2)/numel(srcImg);            
            psnrExpctd(3) = 10*log10(255^2/...
                (sum((int16(srcImgInt(:))-int16(resImg3Int(:))).^2)/numel(srcImg)));                        
            ssimExpctd(3) = testCase.ssim_(srcImgInt,resImg3Int);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'MaxIter', 3,...
                'IsMSE', true,...
                'IsPSNR', true,...
                'IsSSIM', true,...
                'IsVerbose', false);
            
            % Actual values
            step(testCase.stepmonitor,resImg1);
            step(testCase.stepmonitor,resImg2);
            [mseActual,psnrActual,ssimActual] = ...
                step(testCase.stepmonitor,resImg3);
            
            % Evaluation
            testCase.assertEqual(mseActual,mseExpctd,'RelTol',1e-15);
            testCase.assertEqual(psnrActual,psnrExpctd,'RelTol',1e-15);
            testCase.assertEqual(ssimActual,ssimExpctd,'AbsTol',1e-2);
            
            %
            mseActual = get(testCase.stepmonitor,'MSEs');
            testCase.assertEqual(mseActual,mseExpctd,'RelTol',1e-15);
            psnrActual = get(testCase.stepmonitor,'PSNRs');
            testCase.assertEqual(psnrActual,psnrExpctd,'RelTol',1e-15);
            ssimActual = get(testCase.stepmonitor,'SSIMs');
            testCase.assertEqual(ssimActual,ssimExpctd,'AbsTol',1e-2);
            
        end        
        
        % Test
        function testSucessiveMsesPsnrsRgb(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = rand(height,width,3);
            
            % Expected values
            srcImgInt = im2uint8(srcImg);
            %
            resImg1    =   round(srcImg*8)/8;
            resImg1Int = im2uint8(resImg1);
            mseExpctd(1) = sum((int16(srcImgInt(:))-int16(resImg1Int(:))).^2)/numel(srcImg);            
            psnrExpctd(1) = 10*log10(255^2/...
                (sum((int16(srcImgInt(:))-int16(resImg1Int(:))).^2)/numel(srcImg)));            
            %
            resImg2 =   round(srcImg*4)/4;
            resImg2Int = im2uint8(resImg2);
            mseExpctd(2) = sum((int16(srcImgInt(:))-int16(resImg2Int(:))).^2)/numel(srcImg);
            psnrExpctd(2) = 10*log10(255^2/...
                (sum((int16(srcImgInt(:))-int16(resImg2Int(:))).^2)/numel(srcImg)));            
            %
            resImg3 =   round(srcImg*2)/2;
            resImg3Int = im2uint8(resImg3);
            mseExpctd(3) = sum((int16(srcImgInt(:))-int16(resImg3Int(:))).^2)/numel(srcImg);            
            psnrExpctd(3) = 10*log10(255^2/...
                (sum((int16(srcImgInt(:))-int16(resImg3Int(:))).^2)/numel(srcImg)));                        
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'MaxIter', 3,...
                'IsMSE', true,...
                'IsPSNR', true,...
                'IsSSIM', false);
            
            % Actual values
            step(testCase.stepmonitor,resImg1);
            step(testCase.stepmonitor,resImg2);
            [mseActual,psnrActual] = ...
                step(testCase.stepmonitor,resImg3);

            % Evaluation
            testCase.assertEqual(mseActual,mseExpctd,'RelTol',1e-15);
            testCase.assertEqual(psnrActual,psnrExpctd,'RelTol',1e-15);            
        end
        
        % Test
        function testSsimsRgbException(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = rand(height,width,3);
            
            % Expected values
            resImg1 = round(srcImg*8)/8;
            resImg2 = round(srcImg*4)/4;
            resImg3 = round(srcImg*2)/2;
                     
            %
            exceptionIdExpctd = 'SaivDr:IndexOutOfBoundsException';
            messageExpctd = 'SSIM is available only for grayscale image.';
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'MaxIter', 3,...
                'IsMSE', true,...
                'IsPSNR', true,...
                'IsSSIM', true);
            
            % Actual values
            try
                step(testCase.stepmonitor,resImg1);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual,exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
            
            try
                step(testCase.stepmonitor,resImg2);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual,exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
            
            try
                step(testCase.stepmonitor,resImg3);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual,exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
            
        end
                
        % Test
        function testSrcResImage(testCase)

            % Preperation
            height = 48;
            width  = 64;
            srcImg = rand(height,width); 
            resImg = round(srcImg*2)/2;
            
            % Expected values
            srcImgExpctd = im2uint8(srcImg);
            resImgExpctd = im2uint8(resImg);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'IsVisible',true,...
                'ImageFigureHandle',testCase.testfigure,...
                'MaxIter', 1);

            % Actual values
            step(testCase.stepmonitor,resImg);
            %
            hsrcimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Source');
            srcImgActual = get(hsrcimg,'CData');
            %
            hresimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Result');
            resImgActual = get(hresimg,'CData');
            
            % Evaluation           
            testCase.assertEqual(srcImgActual,srcImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(resImgActual,resImgExpctd,'RelTol',1e-15);            
            
        end
        
        % Test
        function testSrcObsResImage(testCase)

            % Preperation
            height = 48;
            width  = 64;
            srcImg = rand(height,width); 
            obsImg = imresize(srcImg,0.5);
            resImg = round(srcImg*2)/2;
            
            % Expected values
            srcImgExpctd = im2uint8(srcImg);
            obsImgExpctd = im2uint8(obsImg);
            resImgExpctd = im2uint8(resImg);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'ObservedImage',obsImg,...
                'IsVisible',true,...
                'ImageFigureHandle',testCase.testfigure,...
                'MaxIter', 1);

            % Actual values
            step(testCase.stepmonitor,resImg);
            %
            hsrcimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Source');
            srcImgActual = get(hsrcimg,'CData');
            %
            hobsimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Observed');
            obsImgActual = get(hobsimg,'CData');
            %
            hresimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Result');
            resImgActual = get(hresimg,'CData');
            
            % Evaluation           
            testCase.assertEqual(srcImgActual,srcImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(obsImgActual,obsImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(resImgActual,resImgExpctd,'RelTol',1e-15);            
            
        end        
        
        % Test
        function testSrcObsResImageRgb(testCase)

            % Preperation
            height = 48;
            width  = 64;
            srcImg = rand(height,width,3); 
            obsImg = imresize(srcImg,0.5);
            resImg = round(srcImg*2)/2;
            
            % Expected values
            srcImgExpctd = im2uint8(srcImg);
            obsImgExpctd = im2uint8(obsImg);
            resImgExpctd = im2uint8(resImg);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'ObservedImage',obsImg,...
                'IsVisible',true,...
                'ImageFigureHandle',testCase.testfigure,...
                'MaxIter', 1);

            % Actual values
            step(testCase.stepmonitor,resImg);
            %
            hsrcimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Source');
            srcImgActual = get(hsrcimg,'CData');
            %
            hobsimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Observed');
            obsImgActual = get(hobsimg,'CData');
            %
            hresimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Result');
            resImgActual = get(hresimg,'CData');
            
            % Evaluation           
            testCase.assertEqual(srcImgActual,srcImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(obsImgActual,obsImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(resImgActual,resImgExpctd,'RelTol',1e-15);            
            
        end  
        
        % Test
        function testSuccessiveSrcObsResImageRgb(testCase)

            % Preperation
            height = 48;
            width  = 64;
            srcImg = rand(height,width,3); 
            obsImg = imresize(srcImg,0.5);
            resImg1 = round(srcImg*2)/2;
            resImg2 = round(srcImg*4)/4;
            resImg3 = round(srcImg*8)/8;
            
            % Expected values
            srcImgExpctd = im2uint8(srcImg);
            obsImgExpctd = im2uint8(obsImg);
            resImg1Expctd = im2uint8(resImg1);
            resImg2Expctd = im2uint8(resImg2);
            resImg3Expctd = im2uint8(resImg3);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'ObservedImage',obsImg,...
                'IsVisible',true,...
                'ImageFigureHandle',testCase.testfigure,...
                'MaxIter', 3,...
                'IsVerbose', false);

            % Evaluation for Step = 1
            step(testCase.stepmonitor,resImg1);
            hsrcimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Source');
            srcImgActual = get(hsrcimg,'CData');
            %
            hobsimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Observed');
            obsImgActual = get(hobsimg,'CData');
            %
            hresimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Result');
            resImg1Actual = get(hresimg,'CData');
            %
            testCase.assertEqual(srcImgActual,srcImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(obsImgActual,obsImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(resImg1Actual,resImg1Expctd,'RelTol',1e-15);      
            %
            stitleExpctd = 'Result (nItr =    1)';
            stitleActual = get(get(get(hresimg,'Parent'),'Title'),'String');
            testCase.assertEqual(stitleActual,stitleExpctd,'RelTol',1e-15);
            

            % Evaluation for Step = 2
            step(testCase.stepmonitor,resImg2);
            srcImgActual = get(hsrcimg,'CData');
            obsImgActual = get(hobsimg,'CData');            
            resImg2Actual = get(hresimg,'CData');
            testCase.assertEqual(srcImgActual,srcImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(obsImgActual,obsImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(resImg2Actual,resImg2Expctd,'RelTol',1e-15);            
            %
            stitleExpctd = 'Result (nItr =    2)';
            stitleActual = get(get(get(hresimg,'Parent'),'Title'),'String');
            testCase.assertEqual(stitleActual,stitleExpctd,'RelTol',1e-15);
            

            % Evaluation for Step = 3
            step(testCase.stepmonitor,resImg3);
            srcImgActual = get(hsrcimg,'CData');
            obsImgActual = get(hobsimg,'CData');                        
            resImg3Actual = get(hresimg,'CData');
            testCase.assertEqual(srcImgActual,srcImgExpctd,'RelTol',1e-15);
            testCase.assertEqual(obsImgActual,obsImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(resImg3Actual,resImg3Expctd,'RelTol',1e-15);                        
            %
            stitleExpctd = 'Result (nItr =    3)';
            stitleActual = get(get(get(hresimg,'Parent'),'Title'),'String');
            testCase.assertEqual(stitleActual,stitleExpctd,'RelTol',1e-15);

        end  
        
        % Test
        function testSuccessiveSrcObsResVolume(testCase)

            % Preperation
            height = 48;
            width  = 64;
            depth  = 16;
            srcImg = rand(height,width,depth); 
            obsImg = imfilter(srcImg,ones(3,3,3)/3^3);
            resImg1 = round(srcImg*2)/2;
            resImg2 = round(srcImg*4)/4;
            resImg3 = round(srcImg*8)/8;
            dataType = 'Volumetric Data';
            
            % Expected values
            srcImgExpctd = im2uint8(srcImg(:,:,depth/2));
            obsImgExpctd = im2uint8(obsImg(:,:,depth/2));
            resImg1Expctd = im2uint8(resImg1(:,:,depth/2));
            resImg2Expctd = im2uint8(resImg2(:,:,depth/2));
            resImg3Expctd = im2uint8(resImg3(:,:,depth/2));
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'DataType',dataType,...
                'SourceImage',srcImg,...
                'ObservedImage',obsImg,...
                'IsVisible',true,...
                'ImageFigureHandle',testCase.testfigure,...
                'MaxIter', 3,...
                'IsVerbose', false);

            % Evaluation for Step = 1
            step(testCase.stepmonitor,resImg1);
            hsrcimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Source');
            srcImgActual = get(hsrcimg,'CData');
            %
            hobsimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Observed');
            obsImgActual = get(hobsimg,'CData');
            %
            hresimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Result');
            resImg1Actual = get(hresimg,'CData');
            %
            testCase.assertEqual(srcImgActual,srcImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(obsImgActual,obsImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(resImg1Actual,resImg1Expctd,'RelTol',1e-15);      
            %
            stitleExpctd = 'Result (nItr =    1)';
            stitleActual = get(get(get(hresimg,'Parent'),'Title'),'String');
            testCase.assertEqual(stitleActual,stitleExpctd,'RelTol',1e-15);
            

            % Evaluation for Step = 2
            step(testCase.stepmonitor,resImg2);
            srcImgActual = get(hsrcimg,'CData');
            obsImgActual = get(hobsimg,'CData');            
            resImg2Actual = get(hresimg,'CData');
            testCase.assertEqual(srcImgActual,srcImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(obsImgActual,obsImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(resImg2Actual,resImg2Expctd,'RelTol',1e-15);            
            %
            stitleExpctd = 'Result (nItr =    2)';
            stitleActual = get(get(get(hresimg,'Parent'),'Title'),'String');
            testCase.assertEqual(stitleActual,stitleExpctd,'RelTol',1e-15);
            
            % Evaluation for Step = 3
            step(testCase.stepmonitor,resImg3);
            srcImgActual = get(hsrcimg,'CData');
            obsImgActual = get(hobsimg,'CData');                        
            resImg3Actual = get(hresimg,'CData');
            testCase.assertEqual(srcImgActual,srcImgExpctd,'RelTol',1e-15);
            testCase.assertEqual(obsImgActual,obsImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(resImg3Actual,resImg3Expctd,'RelTol',1e-15);                        
            %
            stitleExpctd = 'Result (nItr =    3)';
            stitleActual = get(get(get(hresimg,'Parent'),'Title'),'String');
            testCase.assertEqual(stitleActual,stitleExpctd,'RelTol',1e-15);

        end  
        
        % Test
        function testDataTypeImageException(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            dataType = 'Image';
            
            %
            srcImg1 = rand(height,width,1);
            srcImg2 = rand(height,width,2);
            srcImg3 = rand(height,width,3);            
            srcImg4 = rand(height,width,4);
            
            %
            obsImg1 = rand(height,width,1);
            obsImg2 = rand(height,width,2);
            obsImg3 = rand(height,width,3);            
            obsImg4 = rand(height,width,4);            
            
            %
            resImg1 = rand(height,width,1);
            resImg2 = rand(height,width,2);
            resImg3 = rand(height,width,3);
            resImg4 = rand(height,width,4);            

            %
            exceptionIdExpctd = 'SaivDr:InvalidDataFormatException';
            messageExpctd = 'The third dimension must be 1 or 3.';
            
            % 
            import saivdr.utility.*

            % size(SrcImg,3)=2, size(ObsImg,3)=1, size(ResImg,3)=1 
            try
                testCase.stepmonitor = StepMonitoringSystem(...
                    'DataType',dataType,...
                    'SourceImage',srcImg2,...
                    'ObservedImage',obsImg1);
                step(testCase.stepmonitor,resImg1);                
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));                
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual,exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);                
            end
            
            % size(SrcImg,3)=4, size(ObsImg,3)=3, size(ResImg,3)=3 
            try
                testCase.stepmonitor = StepMonitoringSystem(...
                    'DataType',dataType,...
                    'SourceImage',srcImg4,...
                    'ObservedImage',obsImg3);
                step(testCase.stepmonitor,resImg3);                
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));                
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual,exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);                
            end            

            % size(SrcImg,3)=1, size(ObsImg,3)=2, size(ResImg,3)=1 
            try
                testCase.stepmonitor = StepMonitoringSystem(...
                    'DataType',dataType,...
                    'SourceImage',srcImg1,...
                    'ObservedImage',obsImg2);
                step(testCase.stepmonitor,resImg1);                
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));                
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual,exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);                
            end      
            
            % size(SrcImg,3)=3, size(ObsImg,3)=4, size(ResImg,3)=3 
            try
                testCase.stepmonitor = StepMonitoringSystem(...
                    'DataType',dataType,...
                    'SourceImage',srcImg3,...
                    'ObservedImage',obsImg4);
                step(testCase.stepmonitor,resImg3);                
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));                
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual,exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);                
            end                            
            
            % size(SrcImg,3)=1, size(ObsImg,3)=1, size(ResImg,3)=2 
            try
                testCase.stepmonitor = StepMonitoringSystem(...
                    'DataType',dataType,...
                    'SourceImage',srcImg1,...
                    'ObservedImage',obsImg1);
                step(testCase.stepmonitor,resImg2);                
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));                
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual,exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);                
            end              
            
            % size(SrcImg,3)=3, size(ObsImg,3)=3, size(ResImg,3)=4 
            try
                testCase.stepmonitor = StepMonitoringSystem(...
                    'DataType',dataType,...
                    'SourceImage',srcImg3,...
                    'ObservedImage',obsImg3);
                step(testCase.stepmonitor,resImg4);                
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));                
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual,exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);                
            end                   
            
        end
        
        % Test
        function testSuccessiveSrcObsResVolumeImShowFcn(testCase)

            % Preperation
            height = 48;
            width  = 64;
            depth  = 16;
            srcImg = rand(height,width,depth); 
            obsImg = imfilter(srcImg,ones(3,3,3)/(3^3));
            resImg1 = round(srcImg*2)/2;
            resImg2 = round(srcImg*4)/4;
            resImg3 = round(srcImg*8)/8;
            dataType = 'Volumetric Data';
            imShowFcn = @(x) 0.5*x+128;
            
            % Expected values
            srcImgExpctd = imShowFcn(im2uint8(srcImg(:,:,depth/2)));
            obsImgExpctd = imShowFcn(im2uint8(obsImg(:,:,depth/2)));
            resImg1Expctd = imShowFcn(im2uint8(resImg1(:,:,depth/2)));
            resImg2Expctd = imShowFcn(im2uint8(resImg2(:,:,depth/2)));
            resImg3Expctd = imShowFcn(im2uint8(resImg3(:,:,depth/2)));
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'DataType',dataType,...
                'SourceImage',srcImg,...
                'ObservedImage',obsImg,...
                'IsVisible',true,...
                'ImageFigureHandle',testCase.testfigure,...
                'ImShowFcn',imShowFcn,...
                'MaxIter', 3,...
                'IsVerbose', false);

            % Evaluation for Step = 1
            step(testCase.stepmonitor,resImg1);
            hsrcimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Source');
            srcImgActual = get(hsrcimg,'CData');
            %
            hobsimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Observed');
            obsImgActual = get(hobsimg,'CData');
            %
            hresimg = findobj(testCase.testfigure,'Type','image',...
                '-and','UserData','Result');
            resImg1Actual = get(hresimg,'CData');
            %
            testCase.assertEqual(srcImgActual,srcImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(obsImgActual,obsImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(resImg1Actual,resImg1Expctd,'RelTol',1e-15);      
            %
            stitleExpctd = 'Result (nItr =    1)';
            stitleActual = get(get(get(hresimg,'Parent'),'Title'),'String');
            testCase.assertEqual(stitleActual,stitleExpctd,'RelTol',1e-15);
            

            % Evaluation for Step = 2
            step(testCase.stepmonitor,resImg2);
            srcImgActual = get(hsrcimg,'CData');
            obsImgActual = get(hobsimg,'CData');            
            resImg2Actual = get(hresimg,'CData');
            testCase.assertEqual(srcImgActual,srcImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(obsImgActual,obsImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(resImg2Actual,resImg2Expctd,'RelTol',1e-15);            
            %
            stitleExpctd = 'Result (nItr =    2)';
            stitleActual = get(get(get(hresimg,'Parent'),'Title'),'String');
            testCase.assertEqual(stitleActual,stitleExpctd,'RelTol',1e-15);
            
            % Evaluation for Step = 3
            step(testCase.stepmonitor,resImg3);
            srcImgActual = get(hsrcimg,'CData');
            obsImgActual = get(hobsimg,'CData');                        
            resImg3Actual = get(hresimg,'CData');
            testCase.assertEqual(srcImgActual,srcImgExpctd,'RelTol',1e-15);
            testCase.assertEqual(obsImgActual,obsImgExpctd,'RelTol',1e-15);            
            testCase.assertEqual(resImg3Actual,resImg3Expctd,'RelTol',1e-15);                        
            %
            stitleExpctd = 'Result (nItr =    3)';
            stitleActual = get(get(get(hresimg,'Parent'),'Title'),'String');
            testCase.assertEqual(stitleActual,stitleExpctd,'RelTol',1e-15);

        end         
        
        % Test
        function testMseIsConversionToEvaluationTypeFalse(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            resImg = round(srcImg*2)/2;
            
            % Expected values
            mseExpctd = sum((srcImg(:)-resImg(:)).^2)/numel(srcImg);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'IsConversionToEvaluationType',false,...
                'SourceImage',srcImg,...
                'MaxIter', 1,...
                'IsMSE', true);
            
            % Actual values
            mseActual = step(testCase.stepmonitor,resImg);
            
            % Evaluation
            testCase.assertEqual(mseActual,mseExpctd,'RelTol',1e-15);
            
        end
        
        % Test
        function testMseIsConversionToEvaluationTypeWithPeakValue(testCase)
            
            % Preperation
            height = 24;
            width  = 32;
            peakValue = 2^12;
            srcImg = peakValue*rand(height,width);
            resImg = round(srcImg*2)/2;
            
            
            % Expected values
            mseExpctd = sum((srcImg(:)-resImg(:)).^2)/numel(srcImg);
            psnrExpctd = 10*log10(peakValue^2/mseExpctd);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.stepmonitor = StepMonitoringSystem(...
                'IsConversionToEvaluationType',false,...
                'PeakValue',peakValue,...
                'SourceImage',srcImg,...
                'MaxIter', 1,...
                'IsMSE', true,...
                'IsPSNR', true);
            
            % Actual values
            [mseActual,psnrActual] = step(testCase.stepmonitor,resImg);
            
            % Evaluation
            testCase.assertEqual(mseActual,mseExpctd,'RelTol',1e-15);
            testCase.assertEqual(psnrActual,psnrExpctd,'RelTol',1e-15);
            
        end
                
    end
    
end
