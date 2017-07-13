classdef PixelLossSystemTestCase < matlab.unittest.TestCase
    %PIXELLOSSSYSTEMTESTCASE Test case for PixelLossSystem
    %
    % Requirements: MATLAB R2013a
    %
    % Copyright (c) 2013-2017, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
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
        function testPixelLossDefault(testCase)

            % Preperation
            height = 16;
            width = 16;
            density = 0.5;
            seed = 0;
            srcImg = rand(height,width);

            % Expected values
            rng(seed);
            mask = rand(size(srcImg)) > density;
            imgExpctd = mask.*srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem();
            
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
        function testPixelLossWithSeed1(testCase)
            
            % Preperation
            height = 16;
            width = 16;
            density = 0.5;
            seed = 1;
            srcImg = rand(height,width);

            % Expected values
            rng(seed);
            mask = rand(size(srcImg)) > density;
            imgExpctd = mask.*srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
                'Seed',seed);
            
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
        function testPixelLossWithDensity0_2(testCase)
            
            % Preperation
            height = 16;
            width = 16;
            seed = 0;
            density = 0.2;
            srcImg = rand(height,width);

            % Expected values
            rng(seed);
            mask = rand(size(srcImg)) > density;
            imgExpctd = mask.*srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
                'Density',density);
            
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
        function testPixelLossWithSpecifiedMask(testCase)

            % Preperation
            height = 16;
            width = 16;
            seed = 0;
            density = 0.2;
            srcImg = rand(height,width);

            % Expected values
            rng(seed);
            mask = rand(size(srcImg)) > density;
            imgExpctd = mask.*srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
                'LossType','Specified',...
                'Mask',mask);
            
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
        function testAdjointDefault(testCase)
            
            % Preperation
            height = 16;
            width = 16;
            density = 0.5;
            seed = 0;
            srcImg = rand(height,width);
            
            % Expected values
            rng(seed);
            mask = rand(size(srcImg)) > density;
            imgExpctd = mask.*srcImg;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
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
            testCase.linearproc = PixelLossSystem(...
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
            height = 8;
            width = 8;
            srcImg = rand(height,width);
            
            % Expected values
            lmaxExpctd = 1;
            fnameExpctd = 'savetest.mat';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
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
            testCase.linearproc = PixelLossSystem(...
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
        function testRandomClone(testCase)
            
            % Preperation
            height = 16;
            width = 16;
            seed = 1;
            srcImg = rand(height,width);
            
            % Expected values
           % typeExpctd = 'Specified';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
                'Seed',seed);
            
            % Clone
            cloneLinearproc = clone(testCase.linearproc);
            
            % Actual values
            imgOrg = step(testCase.linearproc,srcImg);
            imgCln = step(cloneLinearproc,srcImg);
           % typeActual = get(cloneLinearproc,'LossType');
            
            % Evaluation
            %testCase.verifyEqual(typeActual,typeExpctd);
            testCase.verifyEqual(cloneLinearproc,testCase.linearproc);
            testCase.verifyFalse(cloneLinearproc == testCase.linearproc)
            testCase.verifyEqual(imgCln,imgOrg);
        end 
        
        % Test 
        function testPixelLossDataTypeImage(testCase)

            % Preperation
            height = 16;
            width = 16;
            density = 0.5;
            seed = 0;
            dataType = 'Image';
            srcImg = rand(height,width);

            % Expected values
            rng(seed);
            mask = rand(size(srcImg)) > density;
            imgExpctd = mask.*srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
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
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));            
            
        end
        
        % Test
        function testPixelLossDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            density = 0.5;
            seed = 0;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);

            % Expected values
            rng(seed);
            mask = rand(size(srcImg)) > density;
            imgExpctd = mask.*srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
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
            testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));            
            
        end
        
        % Test
        function testPixelLossWithSeed1DataTypeVolumetric(testCase)
            
            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            density = 0.5;
            seed = 1;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);

            % Expected values
            rng(seed);
            mask = rand(size(srcImg)) > density;
            imgExpctd = mask.*srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
                'DataType',dataType,...
                'Seed',seed);
            
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
        function testPixelLossWithDensity0_2DataTypeVolumetric(testCase)
            
            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            seed = 0;
            density = 0.2;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);

            % Expected values
            rng(seed);
            mask = rand(size(srcImg)) > density;
            imgExpctd = mask.*srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
                'DataType',dataType,...
                'Density',density);
            
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
        function testPixelLossWithSpecifiedMaskDataTypeVolumetric(testCase)

            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            seed = 0;
            density = 0.2;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);

            % Expected values
            rng(seed);
            mask = rand(size(srcImg)) > density;
            imgExpctd = mask.*srcImg;
            lmaxExpctd = 1;
             
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
                'DataType',dataType,...
                'LossType','Specified',...
                'Mask',mask);
            
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
        function testAdjointDefaultDataTypeVolumetric(testCase)
            
            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            density = 0.5;
            seed = 0;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            
            % Expected values
            rng(seed);
            mask = rand(size(srcImg)) > density;
            imgExpctd = mask.*srcImg;
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
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
        function testUseFileForLambdaMaxDataTypeVolumetric(testCase)

            % Preperation
            height = 8;
            width = 8;
            depth = 4;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            
            % Expected values
            lmaxExpctd = 1;
            fnameExpctd = 'lmax.mat';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
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
                testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-10,...
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
            testCase.linearproc = PixelLossSystem(...
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
                testCase.verifyEqual(lmaxActual,lmaxExpctd,'RelTol',1e-10,...
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
            testCase.linearproc = PixelLossSystem(...
                'DataType',dataType,...
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
        function testRandomCloneDataTypeVolumetric(testCase)
            
            % Preperation
            height = 16;
            width = 16;
            depth = 8;
            seed = 1;
            dataType = 'Volumetric Data';
            srcImg = rand(height,width,depth);
            
            % Expected values
            % typeExpctd = 'Specified';
            
            % Instantiation of target class
            import saivdr.degradation.linearprocess.*
            testCase.linearproc = PixelLossSystem(...
                'DataType', dataType,...
                'Seed',seed);
            
            % Clone
            cloneLinearproc = clone(testCase.linearproc);
            
            % Actual values
            imgOrg = step(testCase.linearproc,srcImg);
            imgCln = step(cloneLinearproc,srcImg);
            % typeActual = get(cloneLinearproc,'LossType');
            
            % Evaluation
            %testCase.verifyEqual(typeActual,typeExpctd);
            testCase.verifyEqual(cloneLinearproc,testCase.linearproc);
            testCase.verifyFalse(cloneLinearproc == testCase.linearproc)
            testCase.verifyEqual(imgCln,imgOrg);
        end 
        
    end

end
