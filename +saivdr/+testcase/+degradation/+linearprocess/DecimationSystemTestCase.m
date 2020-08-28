classdef DecimationSystemTestCase < matlab.unittest.TestCase
    %DECIMATIONSYSTEMTESTCASE Test case for DecimationSystem
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
        function testDecimationDefault(testCase)

            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            srcImg = rand(height,width);

            % Expected values
            imgExpctd = downsample(downsample(srcImg,dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem();
            
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
        function testDecimation2x2Identical(testCase)

            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            srcImg = rand(height,width);

            % Expected values
            imgExpctd = downsample(downsample(srcImg,dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
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
        function testDecimation4x4Identical(testCase)

            % Preperation
            dFactor = [4 4];
            height = 16;
            width = 16;
            srcImg = rand(height,width);

            % Expected values
            imgExpctd = downsample(downsample(srcImg,dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
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
        function testDecimationWithGaussianBlur(testCase)

            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = downsample(downsample(...
                imfilter(srcImg,kernel,'conv'),dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testDecimation2x2WithGaussianBlur(testCase)

            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = downsample(downsample(...
                imfilter(srcImg,kernel,'conv'),dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...                
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
        function testDecimation4x4WithGaussianBlur(testCase)

            % Preperation
            dFactor = [4 4];
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = downsample(downsample(...
                imfilter(srcImg,kernel,'conv'),dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...                
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
        function testDecimation2x2WithGaussianBlurSize3x3(testCase)

            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = [3 3];
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = downsample(downsample(...
                imfilter(srcImg,kernel,'conv'),dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...                
                'BlurType','Gaussian',...
                'SizeOfGaussianKernel',sizeOfKernel);
            
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
        function testDecimation2x2WithGaussianBlurSigma0_5d(testCase)

            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = 0.5;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = downsample(downsample(...
                imfilter(srcImg,kernel,'conv'),dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...                
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
        function testDecimation2x2WithGaussianBlurSize3x3Sigma0_5d(testCase)

            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = [3 3];
            sigmaOfKernel = 0.5;
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = downsample(downsample(...
                imfilter(srcImg,kernel,'conv'),dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...                
                'BlurType','Gaussian',...
                'SizeOfGaussianKernel',sizeOfKernel,...
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
        function testDecimationWithBoxBlur(testCase)

            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = dFactor;
            kernel = fspecial('average',sizeOfKernel);

            % Expected values
            imgExpctd = downsample(downsample(...
                imfilter(srcImg,kernel,'conv'),dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testDecimation4x4WithBoxBlur(testCase)

            % Preperation
            dFactor = [4 4];
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = dFactor;
            kernel = fspecial('average',sizeOfKernel);

            % Expected values
            imgExpctd = downsample(downsample(...
                imfilter(srcImg,kernel,'conv'),dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
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
        function testAdjointDefault(testCase)

            % Preperation
            dFactor = [2 2];
            height = 8;
            width = 8;
            srcImg = rand(height,width);

            % Expected values
            imgExpctd = upsample(upsample(srcImg,...
                dFactor(1)).',dFactor(2)).';
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'ProcessingMode','Adjoint');
            
            % Actual values
            imgActual = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end      
        
        % Test
        function testAdjoint4x4Default(testCase)

            % Preperation
            dFactor = [4 4];
            height = 4;
            width = 4;
            srcImg = rand(height,width);

            % Expected values
            imgExpctd = upsample(upsample(srcImg,...
                dFactor(1)).',dFactor(2)).';
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
                'ProcessingMode','Adjoint');
            
            % Actual values
            imgActual = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end    

        % Test
        function testAdjointWithGaussianBlur(testCase)

            % Preperation
            dFactor = [2 2];
            height = 8;
            width = 8;
            srcImg = rand(height,width);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);
            
            % Expected values
            v = upsample(upsample(srcImg,...
                dFactor(1)).',dFactor(2)).';
            imgExpctd = imfilter(v,kernel,'corr');  
                         
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'BlurType','Gaussian',...
                'ProcessingMode','Adjoint');
            
            % Actual values
            imgActual = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end          
        
        % Test
        function testAdjoint4x4WithGaussianBlur(testCase)

            % Preperation
            dFactor = [4 4];
            height = 4;
            width = 4;
            srcImg = rand(height,width);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);
            
            % Expected values
            v = upsample(upsample(srcImg,...
                dFactor(1)).',dFactor(2)).';
            imgExpctd = imfilter(v,kernel,'corr');  
                         
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'BlurType','Gaussian',...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
                'ProcessingMode','Adjoint');
            
            % Actual values
            imgActual = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end         
        
        % Test
        function testAdjointWithBoxBlur(testCase)

            % Preperation
            dFactor = [2 2];
            height = 8;
            width = 8;
            srcImg = rand(height,width);
            kernel = fspecial('average',dFactor);
            
            % Expected values
            v = upsample(upsample(srcImg,...
                dFactor(1),1).',dFactor(2),1).';
            imgExpctd = imfilter(v,kernel,'corr');  
                         
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testAdjoint4x4WithBoxBlur(testCase)

            % Preperation
            dFactor = [4 4];
            height = 4;
            width = 4;
            srcImg = rand(height,width);
            kernel = fspecial('average',dFactor);
            
            % Expected values
            v = upsample(upsample(srcImg,...
                dFactor(1),1).',dFactor(2),1).';
            imgExpctd = imfilter(v,kernel,'corr');  
                         
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'BlurType','Average',...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
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
            height = 8;
            width = 8;
            srcImg = rand(height,width);
            
            % Expected values
            lmaxExpctd = 1;
            fnameExpctd = 'lmax.mat';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
            height = 8;
            width = 8;
            srcImg = rand(height,width);
            
            % Expected values
            lmaxExpctd = 1;
            fnameExpctd = 'savetest.mat';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
            height = 8;
            width = 8;
            srcImg = rand(height,width);
            
            % Expected values
            lmaxExpctd  = 0.5;
            fname = 'loadtest.mat';

            isFileForLmax = (exist(fname,'file')==2);
            if isFileForLmax
                delete(fname)
            end
            lmax = lmaxExpctd; %#ok
            save(fname,'lmax');
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testDecimation2x2WithCustomBlur(testCase)
            
            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            kernel = fspecial('disk',2);
            
            % Expected values
            imgExpctd = downsample(downsample(...
                imfilter(srcImg,kernel,'conv'),dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testAdjoint2x2WithCustomBlur(testCase)

            % Preperation
            dFactor = [2 2];
            height = 8;
            width = 8;
            srcImg = rand(height,width);
            kernel = fspecial('disk',2);
            
            % Expected values
            v = upsample(upsample(srcImg,dFactor(1)).',dFactor(2)).';            
            imgExpctd = imfilter(v,kernel,'corr');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testDecimationWithGaussianBlurClone(testCase)
            
            % Preperation
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'BlurType','Gaussian');

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
        function testDecimationWithGaussianBlurSymmstricExtention(testCase)

            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = downsample(downsample(...
                imfilter(srcImg,kernel,'conv','symmetric'),...
                dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'BlurType','Gaussian',...
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
        
        % Test
        function testDecimationWithGaussianBlurValueExtention(testCase)

            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            kernel = fspecial('gaussian',sizeOfKernel,sigmaOfKernel);

            % Expected values
            imgExpctd = downsample(downsample(...
                imfilter(srcImg,kernel,'conv',0),...
                dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testAdjoint2x2WithCustomBlurCircular(testCase)

            % Preperation
            dFactor = [2 2];
            height = 8;
            width = 8;
            srcImg = rand(height,width);
            kernel = fspecial('disk',2);
            
            % Expected values
            v = upsample(upsample(srcImg,dFactor(1)).',dFactor(2)).';
            imgExpctd = imfilter(v,kernel,'corr','circular');
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'BlurType','Custom',...
                'CustomKernel',kernel,...
                'BoundaryOption','Circular',...
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
        
        % Test for default construction
        function testDecimationDataTypeImage(testCase)
            
            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            dataType = 'Image';
            srcImg = rand(height,width);
            
            % Expected values
            imgExpctd = downsample(downsample(srcImg,dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testDecimation2x2IdenticalDataTypeImage(testCase)
            
            % Preperation
            dFactor = [2 2];
            height = 16;
            width = 16;
            dataType = 'Image';
            srcImg = rand(height,width);
            
            % Expected values
            imgExpctd = downsample(downsample(srcImg,dFactor(1)).',dFactor(2)).';
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
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
        function testDecimationDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);

            % Expected values
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(srcImg,...
                    dFactor(1)),1),...
                    dFactor(2)),1),...
                    dFactor(3)),1);
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testDecimation2x2x2IdenticalDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);

            % Expected values
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(srcImg,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
                'DepthDecimationFactor',dFactor(3),...
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
        function testDecimation4x4x4IdenticalDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [4 4 4];
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);

            % Expected values
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(srcImg,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
                'DepthDecimationFactor',dFactor(3),...
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
        function testDecimationWithGaussianBlurDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);            
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));

            % Expected values
            v = imfilter(srcImg,kernel,'conv');
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(v,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);           
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
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
        function testDecimation2x2x2WithGaussianBlurDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            v = imfilter(srcImg,kernel,'conv');
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(v,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...                
                'DepthDecimationFactor',dFactor(3),... 
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
        function testDecimation4x4x4WithGaussianBlurDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [4 4 4];
            height  = 16;
            width   = 16;
            depth   = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));
            tolPm  = 1e-2;
            itrPm  = 10;
            
            % Expected values
            v = imfilter(srcImg,kernel,'conv');
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(v,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...                
                'DepthDecimationFactor',dFactor(3),...  
                'BlurType','Gaussian',...
                'TolOfPowerMethod',tolPm,...
                'MaxIterOfPowerMethod',itrPm);            
            
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
        function testDecimation2x2x2WithGaussianBlurSize3x3x3DataTypeVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = [3 3 3];
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));

            % Expected values
            v = imfilter(srcImg,kernel,'conv');
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(v,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...                
                'DepthDecimationFactor',dFactor(3),...                                
                'BlurType','Gaussian',...
                'SizeOfGaussianKernel',sizeOfKernel);
            
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
        function testDecimation2x2x2WithGaussianBlurSigma0_5dDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = 0.5;
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));
            tolPm  = 1e-2;
            itrPm  = 10;            
            
            % Expected values
            v = imfilter(srcImg,kernel,'conv');
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(v,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...                
                'DepthDecimationFactor',dFactor(3),...                                
                'BlurType','Gaussian',...
                'SigmaOfGaussianKernel',sigmaOfKernel,...
                'TolOfPowerMethod',tolPm,...
                'MaxIterOfPowerMethod',itrPm);            
            
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
        function testDecimation2x2x2WithGaussianBlurSize3x3x3Sigma0_5dVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 16;
            width = 16;
            depth = 8;
            srcImg = rand(height,width,depth);
            sizeOfKernel = [3 3 3];
            sigmaOfKernel = 0.5;
            dataType = 'Volumetric Data';
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));
            tolPm  = 1e-2;
            itrPm  = 10;            
            
            % Expected values
            v = imfilter(srcImg,kernel,'conv');
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(v,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1); 
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
                'DepthDecimationFactor',dFactor(3),...               
                'BlurType','Gaussian',...
                'SizeOfGaussianKernel',sizeOfKernel,...
                'SigmaOfGaussianKernel',sigmaOfKernel,...
                'TolOfPowerMethod',tolPm,...
                'MaxIterOfPowerMethod',itrPm);            
            
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
        function testDecimationWithBoxBlurDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = dFactor;
            kernel = ones(sizeOfKernel)/prod(sizeOfKernel);
            
            % Expected values
            v = imfilter(srcImg,kernel,'conv');
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(v,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1); 
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(lmaxActual,IsLessThan(lmaxExpctd));            
            import matlab.unittest.constraints.IsGreaterThan            
            testCase.verifyThat(lmaxActual,IsGreaterThan(0));                        
        end          

        % Test
        function testDecimation4x4x4WithBoxBlurDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [4 4 4];
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = dFactor;
            kernel = ones(sizeOfKernel)/prod(sizeOfKernel);

            % Expected values
            v = imfilter(srcImg,kernel,'conv');
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(v,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1); 
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,....
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
                'DepthDecimationFactor',dFactor(3),...
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
        function testAdjointDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 8;
            width = 8;
            depth = 4;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);

            % Expected values
            imgExpctd = ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(srcImg,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1); 
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'ProcessingMode','Adjoint');
            
            % Actual values
            imgActual = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end      
        
        % Test
        function testAdjoint4x4x4DataTypeVolumetric(testCase)

            % Preperation
            dFactor = [4 4 4];
            height = 4;
            width = 4;
            depth = 2;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);

            % Expected values
            imgExpctd = ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(srcImg,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1); 
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
                'DepthDecimationFactor',dFactor(3),...
                'ProcessingMode','Adjoint');
            
            % Actual values
            imgActual = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end    

        % Test
        function testAdjointWithGaussianBlurDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 8;
            width = 8;
            depth = 4;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));
           
            % Expected values
            v = ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(srcImg,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1); 
            imgExpctd = imfilter(v,kernel,'corr');  
                         
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'BlurType','Gaussian',...
                'ProcessingMode','Adjoint');
            
            % Actual values
            imgActual = step(testCase.linearproc,srcImg);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end          
        
        % Test
        function testAdjoint4x4x4WithGaussianBlurDataTypeVolumetric(testCase)
            
            % Preperation
            dFactor = [4 4 4];
            height = 4;
            width = 4;
            depth = 2;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            v = shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(srcImg,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);
            imgExpctd = imfilter(v,kernel,'corr');
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'BlurType','Gaussian',...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
                'DepthDecimationFactor',dFactor(3),...
                'ProcessingMode','Adjoint');
            
            % Actual values
            imgActual = step(testCase.linearproc,srcImg);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
        end
        
        % Test
        function testAdjointWithBoxBlurDataTypeVolumetric(testCase)
            
            % Preperation
            dFactor = [2 2 2];
            height = 8;
            width = 8;
            depth = 4;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            kernel = ones(dFactor)/prod(dFactor);
            
            % Expected values
            v = shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(srcImg,...
                dFactor(1),1),1),...
                dFactor(2),1),1),...
                dFactor(3),1),1);
            imgExpctd = imfilter(v,kernel,'corr');
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testAdjoint4x4x4WithBoxBlurDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [4 4 4];
            height = 4;
            width = 4;
            depth = 2;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            kernel = ones(dFactor)/prod(dFactor);
            
            % Expected values
            v = shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(srcImg,...
                dFactor(1),1),1),...
                dFactor(2),1),1),...
                dFactor(3),1),1);
            imgExpctd = imfilter(v,kernel,'corr');  
                         
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'BlurType','Average',...
                'VerticalDecimationFactor',dFactor(1),...
                'HorizontalDecimationFactor',dFactor(2),...
                'DepthDecimationFactor',dFactor(3),...
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
            height = 8;
            width = 8;
            depth  = 4;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            
            % Expected values
            lmaxExpctd = 1;
            fnameExpctd = 'lmax.mat';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
            height = 8;
            width = 8;
            depth = 4;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            
            % Expected values
            lmaxExpctd = 1;
            fnameExpctd = 'savetest.mat';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
            height = 8;
            width = 8;
            depth = 4;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            
            % Expected values
            lmaxExpctd  = 0.5;
            fname = 'loadtest.mat';

            isFileForLmax = (exist(fname,'file')==2);
            if isFileForLmax
                delete(fname)
            end
            lmax = lmaxExpctd; %#ok
            save(fname,'lmax');
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testDecimation2x2x2WithCustomBlurDataTypeVolumetric(testCase)
            
            % Preperation
            dFactor = [2 2 2];
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            kernel = rand(3,3,3);
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            v = imfilter(srcImg,kernel,'conv');
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(v,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testAdjoint2x2x2WithCustomBlurDataTypeVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 8;
            width = 8;
            depth = 4;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            kernel = rand(3,3,3);
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            v = shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(srcImg,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);
            imgExpctd = imfilter(v,kernel,'corr');  
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testDecimationWithGaussianBlurCloneDataTypeVolumetric(testCase)
            
            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'BlurType','Gaussian');

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
        function testDecimationWithGaussianBlurSymExtVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));

            % Expected values
            v = imfilter(srcImg,kernel,'conv','symmetric');
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(v,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...
                'BlurType','Gaussian',...
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
        
        % Test
        function testDecimationWithGaussianBlurValueExtVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 16;
            width = 16;
            depth = 8;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            sizeOfKernel = 4*dFactor + 1;
            sigmaOfKernel = max(dFactor);%(max(dFactor)/pi)*sqrt(2*log(2));
            hs = (sizeOfKernel-1)/2;
            [ X, Y, Z ] = meshgrid(-hs:hs,-hs:hs,-hs:hs);
            kernel = exp(-(X.^2+Y.^2+Z.^2)/(2*sigmaOfKernel^2));
            kernel = kernel/sum(kernel(:));


            % Expected values
            v = imfilter(srcImg,kernel,'conv',0);
            imgExpctd = ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(v,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
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
        function testAdjoint2x2x2WithCustomBlurCircularVolumetric(testCase)

            % Preperation
            dFactor = [2 2 2];
            height = 8;
            width = 8;
            depth = 4;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            kernel = rand(3,3,3);
            kernel = kernel/sum(kernel(:));
            
            % Expected values
            v = shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(srcImg,...
                dFactor(1)),1),...
                dFactor(2)),1),...
                dFactor(3)),1);
            imgExpctd = imfilter(v,kernel,'corr','circular');  
            lmaxExpctd = 1;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = DecimationSystem(...
                'DataType',dataType,...         
                'BlurType','Custom',...
                'CustomKernel',kernel,...
                'BoundaryOption','Circular',...
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
             
    end

end
