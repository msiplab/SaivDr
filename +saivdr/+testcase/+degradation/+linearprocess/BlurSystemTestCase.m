classdef BlurSystemTestCase < matlab.unittest.TestCase
    %BLURSYSTEMTESTCASE Test case for BlurSystem
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
        linearproc
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.linearproc);
        end
    end
    
    methods (Test)
        
        % Test for default construction
        function testDefault(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);

            % Expected values
            imgExpctd = srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem();

            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');            
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff));            
            
        end

        % Test 
        function testIdentical(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);

            % Expected values
            imgExpctd = srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'ObservedDimension',size(srcImg),...
                'BlurType','Identical');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff));            
            
        end
        
        % Test 
        function testGaussianBlur(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sigmaOfKernel = 2;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel)+1;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Gaussian');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end
        
        % Test
        function testGaussianBlurSize3x3(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sigmaOfKernel = 2.0;            
            sizeOfKernel = [3 3];
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Gaussian',...
                'SizeOfKernel',sizeOfKernel);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                                    
        end                                  

        % Test 
        function testGaussianBlurSigma1(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sigmaOfKernel = 1;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel)+1;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Gaussian',...
                'SigmaOfGaussianKernel',sigmaOfKernel);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                                    
        end                                  

        % Test 
        function testGaussianBlurSize3x3Sigma1(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = [3 3];
            sigmaOfKernel = 1;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Gaussian',...
                'SizeOfKernel',sizeOfKernel,...
                'SigmaOfGaussianKernel',sigmaOfKernel);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual  = get(testCase.linearproc,'LambdaMax');
            imgActual   = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                                    
        end                                  

        % Test 
        function testBoxBlur(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = [3 3];
            kernel = fspecial('average',sizeOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Average');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end          
            
        % Test
        function testBoxBlurSize5x5(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = [ 5 5 ];
            kernel = fspecial('average',sizeOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Average',...
                'SizeOfKernel',sizeOfKernel);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end            

        % Test
        function testAdjointDefault(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);

            % Expected values
            imgExpctd = srcImg;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'ProcessingMode','Adjoint');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            imgActual = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end      

        % Test
        function testAdjointOfGaussianBlur(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sigmaOfKernel = 2;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel) + 1;            
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'corr');  
                         
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Gaussian',...
                'ProcessingMode','Adjoint');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            imgActual = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end          

        % Test
        function testAdjointOfBoxBlur(testCase)

            % Preperation
            height = 16;
            width = 17;
            srcImg = rand(height,width);
            sizeOfKernel = [ 3 3 ];
            kernel = fspecial('average',sizeOfKernel);
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'corr');  
                         
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Average',...
                'ProcessingMode','Adjoint');
            
            % Actual values
            imgActual = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end          

        % Test
        function testUseFileForLambdaMax(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            
            % Expected values
            lmaxExpctd = 1;
            fnameExpctd = './lmax.mat';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'UseFileForLambdaMax',true);
            
            % Actual values
            step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            isFileForLmax = (exist(fnameExpctd,'file')==2);
            testCase.verifyTrue(isFileForLmax,'File does not exist.');
            if isFileForLmax
                s = load(fnameExpctd,'-mat','lmax');
                lmaxActual = s.lmax;
                delete(fnameExpctd)
                diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
                testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                    sprintf('%g',diff));
            end

        end    

        % Test
        function testUseFileForLambdaMaxWithFileNameSpecification(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            
            % Expected values
            lmaxExpctd = 1;
            fnameExpctd = './savetest.mat';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'UseFileForLambdaMax',true,...
                'FileNameForLambdaMax',fnameExpctd);
            
            % Actual values
            step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            isFileForLmax = (exist(fnameExpctd,'file')==2);
            testCase.verifyTrue(isFileForLmax,'File does not exist.');
            if isFileForLmax
                s = load(fnameExpctd,'-mat','lmax');
                lmaxActual = s.lmax;
                delete(fnameExpctd)
                diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
                testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                    sprintf('%g',diff));
            end

        end    

        % Test
        function testUseFileForLambdaMaxLoad(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            
            % Expected values
            lmaxExpctd  = 0.5;
            fname = './loadtest.mat';

            isFileForLmax = (exist(fname,'file')==2);
            if isFileForLmax
                delete(fname)
            end
            lmax = lmaxExpctd; %#ok
            save(fname,'lmax');
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'UseFileForLambdaMax',true,...
                'FileNameForLambdaMax',fname);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
                                    
            % Evaluation
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff));
            
            % 
            isFileForLmax = (exist(fname,'file')==2);            
            if isFileForLmax
                delete(fname)
            end
            
        end
        
        % Test
        function testBlurWithCustomKernel(testCase)
            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            kernel = fspecial('disk',2);
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Custom',...
                'CustomKernel',kernel);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));
            import matlab.unittest.constraints.IsGreaterThan
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));
        end
        
        
        % Test
        function testBlurWithCustomKernelSobel(testCase)
            
            % Preperation
            height = 128;
            width  = 128;
            srcImg = rand(height,width);
            kernel = fspecial('sobel');
            tolPm  = 1e-15;
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = sum(abs(kernel(:)))^2;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Custom',...
                'CustomKernel',kernel,...
                'TolOfPowerMethod',tolPm);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff)); 
        end
        
        % Test
        function testAdjointOfBlurWithCustomKernel(testCase)
            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            kernel = fspecial('disk',2);
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'corr');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Custom',...
                'CustomKernel',kernel,...
                'ProcessingMode','Adjoint');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));
            import matlab.unittest.constraints.IsGreaterThan
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));
        end  
        
        % Test
        function testGaussianBlurClone(testCase)
            
            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sigmaOfKernel = 3;
            sizeOfKernel  = 2;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Gaussian',...
                'SizeOfKernel',sizeOfKernel,...
                'SigmaOfGaussianKernel',sigmaOfKernel);
            
            % Clone
            cloneLinearproc = clone(testCase.linearproc);
            
            % Actual values
            imgOrg = step(testCase.linearproc,srcImg);
            imgCln = step(cloneLinearproc,srcImg);
            
            % Evaluation
            testCase.verifyEqual(cloneLinearproc,testCase.linearproc)
            testCase.verifyFalse(cloneLinearproc == testCase.linearproc)            
            testCase.verifyEqual(imgCln,imgOrg);
        end
        
        % Test
        function testBlurWithSymmetricExtension(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            kernel = rand(3,3);
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv','symmetric');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Custom',...
                'CustomKernel',kernel,...
                'BoundaryOption','Symmetric');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff)); 
        end  
        
        % Test
        function testGaussianBlurWithSymmetricExtension(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sigmaOfKernel = 2;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel)+1;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'symmetric');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Gaussian',...
                'CustomKernel',kernel,...
                'BoundaryOption','Symmetric');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff)); 
        end  
        
        % Test
        function testBlurWithCircularExtension(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            kernel = rand(3,3);
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv','circular');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Custom',...
                'CustomKernel',kernel,...
                'BoundaryOption','Circular');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff)); 
        end          
        
        % Test
        function testBlurWithValueExtension(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sigmaOfKernel = 2;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel)+1;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv',0);
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Gaussian',...
                'BoundaryOption','Value',...
                'BoundaryValue',0);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end                  
        
        % Test
        function testBlurAdjointWithSymmetricExtension(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sigmaOfKernel = 2;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel)+1;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'corr','symmetric');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Gaussian',...
                'ProcessingMode','Adjoint',...
                'BoundaryOption','Symmetric');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(lmaxActual,IsLessThanOrEqualTo(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end                          
      
        % Test 
        function testIdenticalDataTypeImage(testCase)

            % Preperation
            height = 16;
            width = 16;
            dataType = 'Image';
            srcImg = rand(height,width);

            % Expected values
            imgExpctd = srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'ObservedDimension',size(srcImg),...
                'BlurType','Identical',...
                'DataType',dataType);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff));            
            
        end
 
        % Test 
        function testGaussianBlurDataTypeImage(testCase)

            % Preperation
            height = 16;
            width = 16;
            dataType = 'Image';
            srcImg = rand(height,width);
            sigmaOfKernel = 2;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel)+1;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Gaussian',...
                'DataType',dataType);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end
        
        % Test 
        function testIdenticalDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcVlm = rand(height,width,depth);

            % Expected values
            imgExpctd = srcVlm;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'ObservedDimension',size(srcVlm),...
                'BlurType','Identical',...
                'DataType',dataType);
            
            % Actual values
            step(testCase.linearproc,srcVlm);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcVlm);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff));            
            
        end

        % Test 
        function testGaussianBlurDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sigmaOfKernel = 2;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel)+1;
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);            
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'BlurType','Gaussian',...
                'DataType',dataType);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end

        % Test
        function testGaussianBlurSize3x3x3DataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';            
            srcImg = rand(height,width,depth);
            sigmaOfKernel = 2.0;            
            sizeOfKernel = [3 3 3];
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);            
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Gaussian',...
                'SizeOfKernel',sizeOfKernel);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                                    
        end                                  

        % Test 
        function testGaussianBlurSigma1DataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sigmaOfKernel = 1;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel)+1;
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);            
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...]
                'DataType',dataType,...
                'BlurType','Gaussian',...
                'SigmaOfGaussianKernel',sigmaOfKernel);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                                    
        end                                  

        % Test 
        function testGaussianBlurSize3x3Sigma1DataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = [3 3 3];
            sigmaOfKernel = 1;
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Gaussian',...
                'SizeOfKernel',sizeOfKernel,...
                'SigmaOfGaussianKernel',sigmaOfKernel);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual  = get(testCase.linearproc,'LambdaMax');
            imgActual   = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(lmaxActual,IsLessThanOrEqualTo(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                                    
        end                                  

        % Test 
        function testBoxBlurDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = [3 3 3];
            kernel = ones(sizeOfKernel)/prod(sizeOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Average');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(lmaxActual,IsLessThanOrEqualTo(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end          
            
        % Test
        function testBoxBlurSize5x5DataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = [ 5 5 5 ];
            kernel = ones(sizeOfKernel)/prod(sizeOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Average',...
                'SizeOfKernel',sizeOfKernel);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end            

        % Test
        function testAdjointDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);

            % Expected values
            imgExpctd = srcImg;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'ProcessingMode','Adjoint');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            imgActual = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end      

        % Test
        function testAdjointOfGaussianBlurDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sigmaOfKernel = 2;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel) + 1;
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'corr');  
                         
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Gaussian',...
                'ProcessingMode','Adjoint');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            imgActual = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end          

        % Test
        function testAdjointOfBoxBlurDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 17;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = [ 3 3 3 ];
            kernel = ones(sizeOfKernel)/prod(sizeOfKernel);
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'corr');  
                         
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Average',...
                'ProcessingMode','Adjoint');
            
            % Actual values
            imgActual = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end          

        % Test
        function testUseFileForLambdaMaxDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            
            % Expected values
            lmaxExpctd = 1;
            fnameExpctd = './lmax.mat';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'UseFileForLambdaMax',true);
            
            % Actual values
            step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            isFileForLmax = (exist(fnameExpctd,'file')==2);
            testCase.verifyTrue(isFileForLmax,'File does not exist.');
            if isFileForLmax
                s = load(fnameExpctd,'-mat','lmax');
                lmaxActual = s.lmax;
                delete(fnameExpctd)
                diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
                testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                    sprintf('%g',diff));
            end

        end    

        % Test
        function testUseFileForLambdaMaxWithFileNameSpecVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            
            % Expected values
            lmaxExpctd = 1;
            fnameExpctd = './savetest.mat';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'UseFileForLambdaMax',true,...
                'FileNameForLambdaMax',fnameExpctd);
            
            % Actual values
            step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            isFileForLmax = (exist(fnameExpctd,'file')==2);
            testCase.verifyTrue(isFileForLmax,'File does not exist.');
            if isFileForLmax
                s = load(fnameExpctd,'-mat','lmax');
                lmaxActual = s.lmax;
                delete(fnameExpctd)
                diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
                testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                    sprintf('%g',diff));
            end

        end    

        % Test
        function testUseFileForLambdaMaxLoadDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            
            % Expected values
            lmaxExpctd  = 0.5;
            fname = './loadtest.mat';

            isFileForLmax = (exist(fname,'file')==2);
            if isFileForLmax
                delete(fname)
            end
            lmax = lmaxExpctd; %#ok
            save(fname,'lmax');
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'UseFileForLambdaMax',true,...
                'FileNameForLambdaMax',fname);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
                                    
            % Evaluation
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff));
            
            % 
            isFileForLmax = (exist(fname,'file')==2);            
            if isFileForLmax
                delete(fname)
            end
            
        end
        
        % Test
        function testBlurWithCustomKernelDataTypeVolumetric(testCase)
            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            kernel = rand(3,3,3);
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Custom',...
                'CustomKernel',kernel);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));
            import matlab.unittest.constraints.IsGreaterThan
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));
        end
        
        % Test
        function testAdjointOfBlurWithCustomKernelDataTypeVolumetric(testCase)
            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            kernel = rand(3,3,3);
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'corr');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Custom',...
                'CustomKernel',kernel,...
                'ProcessingMode','Adjoint');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));
            import matlab.unittest.constraints.IsGreaterThan
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));
        end  
        
        % Test
        function testGaussianBlurCloneDataTypeVolumetric(testCase)
            
            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sigmaOfKernel = 3;
            sizeOfKernel  = 2;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Gaussian',...
                'SizeOfKernel',sizeOfKernel,...
                'SigmaOfGaussianKernel',sigmaOfKernel);
            
            % Clone
            cloneLinearproc = clone(testCase.linearproc);
            
            % Actual values
            imgOrg = step(testCase.linearproc,srcImg);
            imgCln = step(cloneLinearproc,srcImg);
            
            % Evaluation
            testCase.verifyEqual(cloneLinearproc,testCase.linearproc)
            testCase.verifyFalse(cloneLinearproc == testCase.linearproc)            
            testCase.verifyEqual(imgCln,imgOrg);
        end
        
        % Test
        function testBlurWithSymmetricExtensionDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            kernel = rand(3,3,3);
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv','symmetric');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Custom',...
                'CustomKernel',kernel,...
                'BoundaryOption','Symmetric');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff)); 
        end  
        
        % Test
        function testGaussianBlurWithSymmetricExtensionDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sigmaOfKernel = 2;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel)+1;
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'symmetric');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Gaussian',...
                'CustomKernel',kernel,...
                'BoundaryOption','Symmetric');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff)); 
        end  

        % Test
        function testBlurWithCircularExtensionDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            kernel = rand(3,3,3);
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv','circular');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Custom',...
                'CustomKernel',kernel,...
                'BoundaryOption','Circular');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-3,...
                sprintf('%g',diff)); 
        end          
        
        % Test
        function testBlurWithValueExtensionDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sigmaOfKernel = 2;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel)+1;
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));            

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv',0);
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Gaussian',...
                'BoundaryOption','Value',...
                'BoundaryValue',0);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end                  
        
        % Test
        function testBlurAdjointWithSymmetricExtensionDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sigmaOfKernel = 2;
            sizeOfKernel = 2*ceil(4*sigmaOfKernel)+1;
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));    

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'corr','symmetric');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = BlurSystem(...
                'DataType',dataType,...
                'BlurType','Gaussian',...
                'ProcessingMode','Adjoint',...
                'BoundaryOption','Symmetric');
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(lmaxActual,IsLessThanOrEqualTo(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end  
        
    end

end
