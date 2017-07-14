classdef ComplexBlurSystemTestCase < matlab.unittest.TestCase
    %ComplexBlurSystemTESTCASE Test case for ComplexBlurSystem
    %
    % SVN identifier:
    % $Id: ComplexBlurSystemTestCase.m 714 2015-07-30 21:44:30Z sho $
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));

            % Expected values
            imgExpctd = srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem();

            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');            
            imgActual  = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));            
            
        end

        % Test 
        function testIdentical(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));

            % Expected values
            imgExpctd = srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));            
            
        end
 
        % Test 
        function testGaussianBlur(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            sigmaOfKernel = 2;
            sizeOfKernel = 2*round(4*sigmaOfKernel)+1;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            sigmaOfKernel = 2.0;            
            sizeOfKernel = [3 3];
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            sigmaOfKernel = 1;
            sizeOfKernel = 2*round(4*sigmaOfKernel)+1;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            sizeOfKernel = [3 3];
            sigmaOfKernel = 1;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            sizeOfKernel = [3 3];
            kernel = fspecial('average',sizeOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            sizeOfKernel = [ 5 5 ];
            kernel = fspecial('average',sizeOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));

            % Expected values
            imgExpctd = srcImg;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            sigmaOfKernel = 2;
            sizeOfKernel = 2*round(4*sigmaOfKernel) + 1;            
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'corr');  
                         
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            sizeOfKernel = [ 3 3 ];
            kernel = fspecial('average',sizeOfKernel);
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'corr');  
                         
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            
            % Expected values
            lmaxExpctd = 1;
            fnameExpctd = './lmax.mat';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
                testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-10,...
                    sprintf('%g',diff));
            end

        end    

        % Test
        function testUseFileForLambdaMaxWithFileNameSpecification(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            
            % Expected values
            lmaxExpctd = 1;
            fnameExpctd = './savetest.mat';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
                testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-10,...
                    sprintf('%g',diff));
            end

        end    

        % Test
        function testUseFileForLambdaMaxLoad(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            
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
            testCase.linearproc = ComplexBlurSystem(...
                'UseFileForLambdaMax',true,...
                'FileNameForLambdaMax',fname);
            
            % Actual values
            step(testCase.linearproc,srcImg);
            lmaxActual = get(testCase.linearproc,'LambdaMax');
                                    
            % Evaluation
            diff = abs(lmaxExpctd - lmaxActual)/abs(lmaxExpctd);
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-10,...
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            kernel = fspecial('disk',2);
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
        function testAdjointOfBlurWithCustomKernel(testCase)
            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            kernel = fspecial('disk',2);
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'corr');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            sigmaOfKernel = 3;
            sizeOfKernel  = 2;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            kernel = fspecial('disk',2);
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'symmetric');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-10,...
                sprintf('%g',diff)); 
        end  
        
        % Test
        function testBlurWithCircularExtension(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            kernel = fspecial('disk',2);
            
            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'circular');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-10,...
                sprintf('%g',diff)); 
        end          
        
        % Test
        function testBlurAdjointWithSymmetricExtension(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            sigmaOfKernel = 2;
            sizeOfKernel = 2*round(4*sigmaOfKernel)+1;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'conv',0);
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
        
        function testBlurWithValueExtension(testCase)

            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width).*exp(1i*2*pi*rand(height,width));
            sigmaOfKernel = 2;
            sizeOfKernel = 2*round(4*sigmaOfKernel)+1;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = imfilter(srcImg,kernel,'corr','symmetric');
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = ComplexBlurSystem(...
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
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end                          
        
    end

end