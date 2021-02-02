classdef UdHaarAnalysis3dSystemTestCase < matlab.unittest.TestCase
    %UDHAARANALYSISSYSTEMTESTCASE Test case for UdHaarAnalysisSystem
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

    properties (TestParameter)
        useparallel = { true, false };
    end   
    
    properties
        analyzer
        kernels
    end
    
    methods (TestMethodSetup)
        function kernelSetup(testCase)
            testCase.kernels.haa(:,:,1) = [ 1 1 ; 1 1 ]/8;   % AA
            testCase.kernels.haa(:,:,2) = [ 1 1 ; 1 1 ]/8;   
            testCase.kernels.hha(:,:,1) = [ 1 -1 ; 1 -1 ]/8; % HA
            testCase.kernels.hha(:,:,2) = [ 1 -1 ; 1 -1 ]/8; 
            testCase.kernels.hva(:,:,1) = [ 1 1 ; -1 -1 ]/8; % VA
            testCase.kernels.hva(:,:,2) = [ 1 1 ; -1 -1 ]/8; 
            testCase.kernels.hda(:,:,1) = [ 1 -1 ; -1 1 ]/8; % DA
            testCase.kernels.hda(:,:,2) = [ 1 -1 ; -1 1 ]/8; 
            %
            testCase.kernels.had(:,:,1) = [ 1 1 ; 1 1 ]/8;   % AD
            testCase.kernels.had(:,:,2) = -[ 1 1 ; 1 1 ]/8;   
            testCase.kernels.hhd(:,:,1) = [ 1 -1 ; 1 -1 ]/8; % HD
            testCase.kernels.hhd(:,:,2) = -[ 1 -1 ; 1 -1 ]/8; 
            testCase.kernels.hvd(:,:,1) = [ 1 1 ; -1 -1 ]/8; % VD
            testCase.kernels.hvd(:,:,2) = -[ 1 1 ; -1 -1 ]/8; 
            testCase.kernels.hdd(:,:,1) = [ 1 -1 ; -1 1 ]/8; % DD
            testCase.kernels.hdd(:,:,2) = -[ 1 -1 ; -1 1 ]/8; 
        end
    end
    
    methods (TestMethodTeardown)
        function deteleObject(testCase)
            delete(testCase.analyzer);
        end
    end
    
    methods (Test)

        % Test for default construction
        function testLevel1Size16x16x16(testCase)
            
            nLevels = 1;
            height = 16;
            width = 16;
            depth = 16;
            srcImg = rand(height,width,depth);
            
            % Expected values
            haa = testCase.kernels.haa;
            had = testCase.kernels.had;
            hva = testCase.kernels.hva;
            hvd = testCase.kernels.hvd;
            hha = testCase.kernels.hha;
            hhd = testCase.kernels.hhd;
            hda = testCase.kernels.hda;
            hdd = testCase.kernels.hdd;
            yaa = imfilter(srcImg,haa,'corr','circular');
            yad = imfilter(srcImg,had,'corr','circular');
            yva = imfilter(srcImg,hva,'corr','circular');
            yvd = imfilter(srcImg,hvd,'corr','circular');
            yha = imfilter(srcImg,hha,'corr','circular');
            yhd = imfilter(srcImg,hhd,'corr','circular');
            yda = imfilter(srcImg,hda,'corr','circular');
            ydd = imfilter(srcImg,hdd,'corr','circular');
            coefExpctd = [ 
                yaa(:) 
                yha(:)                
                yva(:)                
                yda(:)                
                yad(:)
                yhd(:)                
                yvd(:)
                ydd(:)].';
            scalesExpctd = repmat([ height width depth ],[8 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [ coefActual, scalesActual ] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            testCase.verifyEqual(norm(coefActual(:)),norm(srcImg(:)),...
                'RelTol',1e-7,sprintf('Energy is not preserved.'));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifyEqual(scalesActual, scalesExpctd);
        end

        
        function testLevel1Size16x32x64(testCase)
            
            nLevels = 1;
            height = 16;
            width = 32;
            depth = 64;
            srcImg = rand(height,width,depth);
            
            % Expected values
            haa = testCase.kernels.haa;
            had = testCase.kernels.had;
            hva = testCase.kernels.hva;
            hvd = testCase.kernels.hvd;
            hha = testCase.kernels.hha;
            hhd = testCase.kernels.hhd;
            hda = testCase.kernels.hda;
            hdd = testCase.kernels.hdd;
            yaa = imfilter(srcImg,haa,'corr','circular');
            yad = imfilter(srcImg,had,'corr','circular');
            yva = imfilter(srcImg,hva,'corr','circular');
            yvd = imfilter(srcImg,hvd,'corr','circular');
            yha = imfilter(srcImg,hha,'corr','circular');
            yhd = imfilter(srcImg,hhd,'corr','circular');
            yda = imfilter(srcImg,hda,'corr','circular');
            ydd = imfilter(srcImg,hdd,'corr','circular');
            coefExpctd = [ 
                yaa(:) 
                yha(:)                
                yva(:)                
                yda(:)                
                yad(:)
                yhd(:)                
                yvd(:)
                ydd(:)].';            
            scalesExpctd = repmat([ height, width depth],[8 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [ coefActual, scalesActual ] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            testCase.verifyEqual(norm(coefActual(:)),norm(srcImg(:)),...
                'RelTol',1e-7,sprintf('Energy is not preserved.'));            
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifyEqual(scalesActual, scalesExpctd);
        end

        % Test for default construction
        function testLevel2Size16x16x16(testCase)
            
            nLevels = 2;
            height = 16;
            width = 16;
            depth = 16;
            srcImg = rand(height,width,depth);
            
            % Expected values
            upsample3_ = @(x,u,s)...
                    shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(x,...
                    u,s),1),...
                    u,s),1),...
                    u,s),1);
            haa2 = testCase.kernels.haa;
            had2 = testCase.kernels.had;
            hva2 = testCase.kernels.hva;
            hvd2 = testCase.kernels.hvd;
            hha2 = testCase.kernels.hha;
            hhd2 = testCase.kernels.hhd;
            hda2 = testCase.kernels.hda;
            hdd2 = testCase.kernels.hdd;
            haa1 = imfilter(upsample3_(haa2,2,1),haa2);
            had1 = imfilter(upsample3_(had2,2,1),haa2);
            hva1 = imfilter(upsample3_(hva2,2,1),haa2);
            hvd1 = imfilter(upsample3_(hvd2,2,1),haa2);
            hha1 = imfilter(upsample3_(hha2,2,1),haa2);
            hhd1 = imfilter(upsample3_(hhd2,2,1),haa2);
            hda1 = imfilter(upsample3_(hda2,2,1),haa2);
            hdd1 = imfilter(upsample3_(hdd2,2,1),haa2);
            yaa1 = imfilter(srcImg,haa1,'corr','circular');
            yad1 = imfilter(srcImg,had1,'corr','circular');
            yva1 = imfilter(srcImg,hva1,'corr','circular');
            yvd1 = imfilter(srcImg,hvd1,'corr','circular');
            yha1 = imfilter(srcImg,hha1,'corr','circular');
            yhd1 = imfilter(srcImg,hhd1,'corr','circular');
            yda1 = imfilter(srcImg,hda1,'corr','circular');
            ydd1 = imfilter(srcImg,hdd1,'corr','circular');
            yad2 = imfilter(srcImg,had2,'corr','circular');
            yva2 = imfilter(srcImg,hva2,'corr','circular');
            yvd2 = imfilter(srcImg,hvd2,'corr','circular');
            yha2 = imfilter(srcImg,hha2,'corr','circular');
            yhd2 = imfilter(srcImg,hhd2,'corr','circular');
            yda2 = imfilter(srcImg,hda2,'corr','circular');
            ydd2 = imfilter(srcImg,hdd2,'corr','circular');            
            coefExpctd = [
                yaa1(:) 
                yha1(:)                
                yva1(:)                
                yda1(:)                
                yad1(:)
                yhd1(:)                
                yvd1(:)
                ydd1(:)
                yha2(:)                
                yva2(:)                
                yda2(:)                
                yad2(:)
                yhd2(:)                
                yvd2(:)
                ydd2(:)                
                ].';            
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            testCase.verifyEqual(norm(coefActual(:)),norm(srcImg(:)),...
                'RelTol',1e-7,sprintf('Energy is not preserved.'));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test for default construction
        function testLevel2Size16x32x64(testCase)
            
            nLevels = 2;
            height = 16;
            width = 32;
            depth = 64;
            srcImg = rand(height,width,depth);
            
            % Expected values
            upsample3_ = @(x,u,s)...
                    shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(x,...
                    u,s),1),...
                    u,s),1),...
                    u,s),1);
            haa2 = testCase.kernels.haa;
            had2 = testCase.kernels.had;
            hva2 = testCase.kernels.hva;
            hvd2 = testCase.kernels.hvd;
            hha2 = testCase.kernels.hha;
            hhd2 = testCase.kernels.hhd;
            hda2 = testCase.kernels.hda;
            hdd2 = testCase.kernels.hdd;
            haa1 = imfilter(upsample3_(haa2,2,1),haa2);
            had1 = imfilter(upsample3_(had2,2,1),haa2);
            hva1 = imfilter(upsample3_(hva2,2,1),haa2);
            hvd1 = imfilter(upsample3_(hvd2,2,1),haa2);
            hha1 = imfilter(upsample3_(hha2,2,1),haa2);
            hhd1 = imfilter(upsample3_(hhd2,2,1),haa2);
            hda1 = imfilter(upsample3_(hda2,2,1),haa2);
            hdd1 = imfilter(upsample3_(hdd2,2,1),haa2);
            yaa1 = imfilter(srcImg,haa1,'corr','circular');
            yad1 = imfilter(srcImg,had1,'corr','circular');
            yva1 = imfilter(srcImg,hva1,'corr','circular');
            yvd1 = imfilter(srcImg,hvd1,'corr','circular');
            yha1 = imfilter(srcImg,hha1,'corr','circular');
            yhd1 = imfilter(srcImg,hhd1,'corr','circular');
            yda1 = imfilter(srcImg,hda1,'corr','circular');
            ydd1 = imfilter(srcImg,hdd1,'corr','circular');
            yad2 = imfilter(srcImg,had2,'corr','circular');
            yva2 = imfilter(srcImg,hva2,'corr','circular');
            yvd2 = imfilter(srcImg,hvd2,'corr','circular');
            yha2 = imfilter(srcImg,hha2,'corr','circular');
            yhd2 = imfilter(srcImg,hhd2,'corr','circular');
            yda2 = imfilter(srcImg,hda2,'corr','circular');
            ydd2 = imfilter(srcImg,hdd2,'corr','circular');            
            coefExpctd = [
                yaa1(:) 
                yha1(:)                
                yva1(:)                
                yda1(:)                
                yad1(:)
                yhd1(:)                
                yvd1(:)
                ydd1(:)
                yha2(:)                
                yva2(:)                
                yda2(:)                
                yad2(:)
                yhd2(:)                
                yvd2(:)
                ydd2(:)                
                ].';            
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            testCase.verifyEqual(norm(coefActual(:)),norm(srcImg(:)),...
                'RelTol',1e-7,sprintf('Energy is not preserved.'));            
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));

        end        

        % Test for default construction
        function testLevel3Size16x16x16(testCase)
            
            nLevels = 3;
            height = 16;
            width = 16;
            depth = 16;
            srcImg = rand(height,width,depth);
            
            % Expected values
            upsample3_ = @(x,u,s)...
                    shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(x,...
                    u,s),1),...
                    u,s),1),...
                    u,s),1);
            haa3 = testCase.kernels.haa;
            had3 = testCase.kernels.had;
            hva3 = testCase.kernels.hva;
            hvd3 = testCase.kernels.hvd;
            hha3 = testCase.kernels.hha;
            hhd3 = testCase.kernels.hhd;
            hda3 = testCase.kernels.hda;
            hdd3 = testCase.kernels.hdd;
            haa2 = imfilter(upsample3_(haa3,2,1),haa3);
            had2 = imfilter(upsample3_(had3,2,1),haa3);
            hva2 = imfilter(upsample3_(hva3,2,1),haa3);
            hvd2 = imfilter(upsample3_(hvd3,2,1),haa3);
            hha2 = imfilter(upsample3_(hha3,2,1),haa3);
            hhd2 = imfilter(upsample3_(hhd3,2,1),haa3);
            hda2 = imfilter(upsample3_(hda3,2,1),haa3);
            hdd2 = imfilter(upsample3_(hdd3,2,1),haa3);
            haa1 = imfilter(upsample3_(haa3,4,2),haa2);
            had1 = imfilter(upsample3_(had3,4,2),haa2);
            hva1 = imfilter(upsample3_(hva3,4,2),haa2);
            hvd1 = imfilter(upsample3_(hvd3,4,2),haa2);
            hha1 = imfilter(upsample3_(hha3,4,2),haa2);
            hhd1 = imfilter(upsample3_(hhd3,4,2),haa2);
            hda1 = imfilter(upsample3_(hda3,4,2),haa2);
            hdd1 = imfilter(upsample3_(hdd3,4,2),haa2);            
            yaa1 = imfilter(srcImg,haa1,'corr','circular');
            yad1 = imfilter(srcImg,had1,'corr','circular');
            yva1 = imfilter(srcImg,hva1,'corr','circular');
            yvd1 = imfilter(srcImg,hvd1,'corr','circular');
            yha1 = imfilter(srcImg,hha1,'corr','circular');
            yhd1 = imfilter(srcImg,hhd1,'corr','circular');
            yda1 = imfilter(srcImg,hda1,'corr','circular');
            ydd1 = imfilter(srcImg,hdd1,'corr','circular');            
            yad2 = imfilter(srcImg,had2,'corr','circular');
            yva2 = imfilter(srcImg,hva2,'corr','circular');
            yvd2 = imfilter(srcImg,hvd2,'corr','circular');
            yha2 = imfilter(srcImg,hha2,'corr','circular');
            yhd2 = imfilter(srcImg,hhd2,'corr','circular');
            yda2 = imfilter(srcImg,hda2,'corr','circular');
            ydd2 = imfilter(srcImg,hdd2,'corr','circular');
            yad3 = imfilter(srcImg,had3,'corr','circular');
            yva3 = imfilter(srcImg,hva3,'corr','circular');
            yvd3 = imfilter(srcImg,hvd3,'corr','circular');
            yha3 = imfilter(srcImg,hha3,'corr','circular');
            yhd3 = imfilter(srcImg,hhd3,'corr','circular');
            yda3 = imfilter(srcImg,hda3,'corr','circular');
            ydd3 = imfilter(srcImg,hdd3,'corr','circular');            
            coefExpctd = [
                yaa1(:) 
                yha1(:)                
                yva1(:)                
                yda1(:)                
                yad1(:)
                yhd1(:)                
                yvd1(:)
                ydd1(:)
                yha2(:)                
                yva2(:)                
                yda2(:)                
                yad2(:)
                yhd2(:)                
                yvd2(:)
                ydd2(:)                
                yha3(:)                
                yva3(:)                
                yda3(:)                
                yad3(:)
                yhd3(:)                
                yvd3(:)
                ydd3(:)                                
                ].';            
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            testCase.verifyEqual(norm(coefActual(:)),norm(srcImg(:)),...
                'RelTol',1e-7,sprintf('Energy is not preserved.'));        
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));

        end

        % Test for default construction
        function testLevel3Size16x32x64(testCase)
            
            nLevels = 3;
            height = 16;
            width = 32;
            depth = 64;
            srcImg = rand(height,width,depth);
            
            % Expected values
            upsample3_ = @(x,u,s)...
                    shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(x,...
                    u,s),1),...
                    u,s),1),...
                    u,s),1);
            haa3 = testCase.kernels.haa;
            had3 = testCase.kernels.had;
            hva3 = testCase.kernels.hva;
            hvd3 = testCase.kernels.hvd;
            hha3 = testCase.kernels.hha;
            hhd3 = testCase.kernels.hhd;
            hda3 = testCase.kernels.hda;
            hdd3 = testCase.kernels.hdd;
            haa2 = imfilter(upsample3_(haa3,2,1),haa3);
            had2 = imfilter(upsample3_(had3,2,1),haa3);
            hva2 = imfilter(upsample3_(hva3,2,1),haa3);
            hvd2 = imfilter(upsample3_(hvd3,2,1),haa3);
            hha2 = imfilter(upsample3_(hha3,2,1),haa3);
            hhd2 = imfilter(upsample3_(hhd3,2,1),haa3);
            hda2 = imfilter(upsample3_(hda3,2,1),haa3);
            hdd2 = imfilter(upsample3_(hdd3,2,1),haa3);
            haa1 = imfilter(upsample3_(haa3,4,2),haa2);
            had1 = imfilter(upsample3_(had3,4,2),haa2);
            hva1 = imfilter(upsample3_(hva3,4,2),haa2);
            hvd1 = imfilter(upsample3_(hvd3,4,2),haa2);
            hha1 = imfilter(upsample3_(hha3,4,2),haa2);
            hhd1 = imfilter(upsample3_(hhd3,4,2),haa2);
            hda1 = imfilter(upsample3_(hda3,4,2),haa2);
            hdd1 = imfilter(upsample3_(hdd3,4,2),haa2);            
            yaa1 = imfilter(srcImg,haa1,'corr','circular');
            yad1 = imfilter(srcImg,had1,'corr','circular');
            yva1 = imfilter(srcImg,hva1,'corr','circular');
            yvd1 = imfilter(srcImg,hvd1,'corr','circular');
            yha1 = imfilter(srcImg,hha1,'corr','circular');
            yhd1 = imfilter(srcImg,hhd1,'corr','circular');
            yda1 = imfilter(srcImg,hda1,'corr','circular');
            ydd1 = imfilter(srcImg,hdd1,'corr','circular');            
            yad2 = imfilter(srcImg,had2,'corr','circular');
            yva2 = imfilter(srcImg,hva2,'corr','circular');
            yvd2 = imfilter(srcImg,hvd2,'corr','circular');
            yha2 = imfilter(srcImg,hha2,'corr','circular');
            yhd2 = imfilter(srcImg,hhd2,'corr','circular');
            yda2 = imfilter(srcImg,hda2,'corr','circular');
            ydd2 = imfilter(srcImg,hdd2,'corr','circular');
            yad3 = imfilter(srcImg,had3,'corr','circular');
            yva3 = imfilter(srcImg,hva3,'corr','circular');
            yvd3 = imfilter(srcImg,hvd3,'corr','circular');
            yha3 = imfilter(srcImg,hha3,'corr','circular');
            yhd3 = imfilter(srcImg,hhd3,'corr','circular');
            yda3 = imfilter(srcImg,hda3,'corr','circular');
            ydd3 = imfilter(srcImg,hdd3,'corr','circular');            
            coefExpctd = [
                yaa1(:) 
                yha1(:)                
                yva1(:)                
                yda1(:)                
                yad1(:)
                yhd1(:)                
                yvd1(:)
                ydd1(:)
                yha2(:)                
                yva2(:)                
                yda2(:)                
                yad2(:)
                yhd2(:)                
                yvd2(:)
                ydd2(:)                
                yha3(:)                
                yva3(:)                
                yda3(:)                
                yad3(:)
                yhd3(:)                
                yvd3(:)
                ydd3(:)                                
                ].';            
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            testCase.verifyEqual(norm(coefActual(:)),norm(srcImg(:)),...
                'RelTol',1e-7,sprintf('Energy is not preserved.'));        
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-6,sprintf('%g',diff));

        end
        
       % Test for default construction
        function testLevel1ReconstructionSize16x16x16(testCase)
            
            nLevels = 1;
            height = 16;
            width = 16;
            depth = 16;
                        
            % Expected values
            imgExpctd = rand(height,width,depth);
            dimExpctd = [ height, width depth];
            nPixels = numel(imgExpctd);

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels);
        
            % Analysis
            coefs = step(testCase.analyzer,imgExpctd);
            caa = reshape(coefs(1:nPixels),height,width,depth);
            cha = reshape(coefs(nPixels+1:2*nPixels),height,width,depth);
            cva = reshape(coefs(2*nPixels+1:3*nPixels),height,width,depth);
            cda = reshape(coefs(3*nPixels+1:4*nPixels),height,width,depth);
            cad = reshape(coefs(4*nPixels+1:5*nPixels),height,width,depth);
            chd = reshape(coefs(5*nPixels+1:6*nPixels),height,width,depth);
            cvd = reshape(coefs(6*nPixels+1:7*nPixels),height,width,depth);
            cdd = reshape(coefs(7*nPixels+1:8*nPixels),height,width,depth);            
        
            % Reconstruction
            haa = testCase.kernels.haa;
            had = testCase.kernels.had;
            hva = testCase.kernels.hva;
            hvd = testCase.kernels.hvd;
            hha = testCase.kernels.hha;
            hhd = testCase.kernels.hhd;
            hda = testCase.kernels.hda;
            hdd = testCase.kernels.hdd;
            yaa = circshift(imfilter(caa,haa,'conv','circular'),[1 1 1]);
            yha = circshift(imfilter(cha,hha,'conv','circular'),[1 1 1]);
            yva = circshift(imfilter(cva,hva,'conv','circular'),[1 1 1]);
            yda = circshift(imfilter(cda,hda,'conv','circular'),[1 1 1]);
            yad = circshift(imfilter(cad,had,'conv','circular'),[1 1 1]);
            yhd = circshift(imfilter(chd,hhd,'conv','circular'),[1 1 1]);
            yvd = circshift(imfilter(cvd,hvd,'conv','circular'),[1 1 1]);
            ydd = circshift(imfilter(cdd,hdd,'conv','circular'),[1 1 1]);

            % Actual values
            imgActual = yaa + yha + yva + yda ...
                    + yad + yhd + yvd + ydd;
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

       % Test for default construction
        function testLevel1ReconstructionSize16x32x64(testCase)
            
            nLevels = 1;
            height = 16;
            width = 32;
            depth = 64;
                        
            % Expected values
            imgExpctd = rand(height,width,depth);
            dimExpctd = [ height, width depth];
            nPixels = numel(imgExpctd);

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels);
        
            % Analysis
            coefs = step(testCase.analyzer,imgExpctd);
            caa = reshape(coefs(1:nPixels),height,width,depth);
            cha = reshape(coefs(nPixels+1:2*nPixels),height,width,depth);
            cva = reshape(coefs(2*nPixels+1:3*nPixels),height,width,depth);
            cda = reshape(coefs(3*nPixels+1:4*nPixels),height,width,depth);
            cad = reshape(coefs(4*nPixels+1:5*nPixels),height,width,depth);
            chd = reshape(coefs(5*nPixels+1:6*nPixels),height,width,depth);
            cvd = reshape(coefs(6*nPixels+1:7*nPixels),height,width,depth);
            cdd = reshape(coefs(7*nPixels+1:8*nPixels),height,width,depth);            
        
            % Reconstruction
            haa = testCase.kernels.haa;
            had = testCase.kernels.had;
            hva = testCase.kernels.hva;
            hvd = testCase.kernels.hvd;
            hha = testCase.kernels.hha;
            hhd = testCase.kernels.hhd;
            hda = testCase.kernels.hda;
            hdd = testCase.kernels.hdd;
            yaa = circshift(imfilter(caa,haa,'conv','circular'),[1 1 1]);
            yha = circshift(imfilter(cha,hha,'conv','circular'),[1 1 1]);
            yva = circshift(imfilter(cva,hva,'conv','circular'),[1 1 1]);
            yda = circshift(imfilter(cda,hda,'conv','circular'),[1 1 1]);
            yad = circshift(imfilter(cad,had,'conv','circular'),[1 1 1]);
            yhd = circshift(imfilter(chd,hhd,'conv','circular'),[1 1 1]);
            yvd = circshift(imfilter(cvd,hvd,'conv','circular'),[1 1 1]);
            ydd = circshift(imfilter(cdd,hdd,'conv','circular'),[1 1 1]);

            % Actual values
            imgActual = yaa + yha + yva + yda ...
                    + yad + yhd + yvd + ydd;
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test for default construction
        function testLevel2ReconstructionSize16x16x16(testCase)
            
            nLevels = 2;
            height = 16;
            width = 16;    
            depth = 16;
                        
            % Expected values
            imgExpctd = rand(height,width,depth);
            dimExpctd = [ height, width, depth ];
            nPixels = numel(imgExpctd);

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels);
        
            % Analysis
            coefs = step(testCase.analyzer,imgExpctd);
            caa1 = reshape(coefs(1:nPixels),height,width,depth);
            cha1 = reshape(coefs(nPixels+1:2*nPixels),height,width,depth);
            cva1 = reshape(coefs(2*nPixels+1:3*nPixels),height,width,depth);
            cda1 = reshape(coefs(3*nPixels+1:4*nPixels),height,width,depth);
            cad1 = reshape(coefs(4*nPixels+1:5*nPixels),height,width,depth);
            chd1 = reshape(coefs(5*nPixels+1:6*nPixels),height,width,depth);
            cvd1 = reshape(coefs(6*nPixels+1:7*nPixels),height,width,depth);
            cdd1 = reshape(coefs(7*nPixels+1:8*nPixels),height,width,depth);            
            cha2 = reshape(coefs(8*nPixels+1:9*nPixels),height,width,depth);
            cva2 = reshape(coefs(9*nPixels+1:10*nPixels),height,width,depth);
            cda2 = reshape(coefs(10*nPixels+1:11*nPixels),height,width,depth);
            cad2 = reshape(coefs(11*nPixels+1:12*nPixels),height,width,depth);
            chd2 = reshape(coefs(12*nPixels+1:13*nPixels),height,width,depth);
            cvd2 = reshape(coefs(13*nPixels+1:14*nPixels),height,width,depth);
            cdd2 = reshape(coefs(14*nPixels+1:15*nPixels),height,width,depth);                    
         
           % Reconstruction
            upsample3_ = @(x,u,s)...
                    shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(x,...
                    u,s),1),...
                    u,s),1),...
                    u,s),1);        
            haa2 = testCase.kernels.haa;
            had2 = testCase.kernels.had;
            hva2 = testCase.kernels.hva;
            hvd2 = testCase.kernels.hvd;
            hha2 = testCase.kernels.hha;
            hhd2 = testCase.kernels.hhd;
            hda2 = testCase.kernels.hda;
            hdd2 = testCase.kernels.hdd;
            haa1 = imfilter(upsample3_(haa2,2,1),haa2);
            had1 = imfilter(upsample3_(had2,2,1),haa2);
            hva1 = imfilter(upsample3_(hva2,2,1),haa2);
            hvd1 = imfilter(upsample3_(hvd2,2,1),haa2);
            hha1 = imfilter(upsample3_(hha2,2,1),haa2);
            hhd1 = imfilter(upsample3_(hhd2,2,1),haa2);
            hda1 = imfilter(upsample3_(hda2,2,1),haa2);
            hdd1 = imfilter(upsample3_(hdd2,2,1),haa2);
            %
            yaa1 = circshift(imfilter(caa1,haa1,'conv','circular'),[1 1 1]);
            yha1 = circshift(imfilter(cha1,hha1,'conv','circular'),[1 1 1]);
            yva1 = circshift(imfilter(cva1,hva1,'conv','circular'),[1 1 1]);
            yda1 = circshift(imfilter(cda1,hda1,'conv','circular'),[1 1 1]);
            yad1 = circshift(imfilter(cad1,had1,'conv','circular'),[1 1 1]);
            yhd1 = circshift(imfilter(chd1,hhd1,'conv','circular'),[1 1 1]);
            yvd1 = circshift(imfilter(cvd1,hvd1,'conv','circular'),[1 1 1]);
            ydd1 = circshift(imfilter(cdd1,hdd1,'conv','circular'),[1 1 1]);        
            yha2 = circshift(imfilter(cha2,hha2,'conv','circular'),[1 1 1]);
            yva2 = circshift(imfilter(cva2,hva2,'conv','circular'),[1 1 1]);
            yda2 = circshift(imfilter(cda2,hda2,'conv','circular'),[1 1 1]);
            yad2 = circshift(imfilter(cad2,had2,'conv','circular'),[1 1 1]);
            yhd2 = circshift(imfilter(chd2,hhd2,'conv','circular'),[1 1 1]);
            yvd2 = circshift(imfilter(cvd2,hvd2,'conv','circular'),[1 1 1]);
            ydd2 = circshift(imfilter(cdd2,hdd2,'conv','circular'),[1 1 1]);
            
            % Actual values
            imgActual = yaa1 + yha1 + yva1 + yda1 + yad1 + yhd1 + yvd1 + ydd1 ...
                + yha2 + yva2 + yda2 + yad2 + yhd2 + yvd2 + ydd2;
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test for default construction
        function testLevel2ReconstructionSize16x32x64(testCase)
            
            nLevels = 2;
            height = 16;
            width = 32;
            depth = 64;
                        
            % Expected values
            imgExpctd = rand(height,width,depth);
            dimExpctd = [ height, width, depth ];
            nPixels = numel(imgExpctd);

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels);
        
            % Analysis
            coefs = step(testCase.analyzer,imgExpctd);
            caa1 = reshape(coefs(1:nPixels),height,width,depth);
            cha1 = reshape(coefs(nPixels+1:2*nPixels),height,width,depth);
            cva1 = reshape(coefs(2*nPixels+1:3*nPixels),height,width,depth);
            cda1 = reshape(coefs(3*nPixels+1:4*nPixels),height,width,depth);
            cad1 = reshape(coefs(4*nPixels+1:5*nPixels),height,width,depth);
            chd1 = reshape(coefs(5*nPixels+1:6*nPixels),height,width,depth);
            cvd1 = reshape(coefs(6*nPixels+1:7*nPixels),height,width,depth);
            cdd1 = reshape(coefs(7*nPixels+1:8*nPixels),height,width,depth);            
            cha2 = reshape(coefs(8*nPixels+1:9*nPixels),height,width,depth);
            cva2 = reshape(coefs(9*nPixels+1:10*nPixels),height,width,depth);
            cda2 = reshape(coefs(10*nPixels+1:11*nPixels),height,width,depth);
            cad2 = reshape(coefs(11*nPixels+1:12*nPixels),height,width,depth);
            chd2 = reshape(coefs(12*nPixels+1:13*nPixels),height,width,depth);
            cvd2 = reshape(coefs(13*nPixels+1:14*nPixels),height,width,depth);
            cdd2 = reshape(coefs(14*nPixels+1:15*nPixels),height,width,depth);                    
         
           % Reconstruction
            upsample3_ = @(x,u,s)...
                    shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(x,...
                    u,s),1),...
                    u,s),1),...
                    u,s),1);        
            haa2 = testCase.kernels.haa;
            had2 = testCase.kernels.had;
            hva2 = testCase.kernels.hva;
            hvd2 = testCase.kernels.hvd;
            hha2 = testCase.kernels.hha;
            hhd2 = testCase.kernels.hhd;
            hda2 = testCase.kernels.hda;
            hdd2 = testCase.kernels.hdd;
            haa1 = imfilter(upsample3_(haa2,2,1),haa2);
            had1 = imfilter(upsample3_(had2,2,1),haa2);
            hva1 = imfilter(upsample3_(hva2,2,1),haa2);
            hvd1 = imfilter(upsample3_(hvd2,2,1),haa2);
            hha1 = imfilter(upsample3_(hha2,2,1),haa2);
            hhd1 = imfilter(upsample3_(hhd2,2,1),haa2);
            hda1 = imfilter(upsample3_(hda2,2,1),haa2);
            hdd1 = imfilter(upsample3_(hdd2,2,1),haa2);
            %
            yaa1 = circshift(imfilter(caa1,haa1,'conv','circular'),[1 1 1]);
            yha1 = circshift(imfilter(cha1,hha1,'conv','circular'),[1 1 1]);
            yva1 = circshift(imfilter(cva1,hva1,'conv','circular'),[1 1 1]);
            yda1 = circshift(imfilter(cda1,hda1,'conv','circular'),[1 1 1]);
            yad1 = circshift(imfilter(cad1,had1,'conv','circular'),[1 1 1]);
            yhd1 = circshift(imfilter(chd1,hhd1,'conv','circular'),[1 1 1]);
            yvd1 = circshift(imfilter(cvd1,hvd1,'conv','circular'),[1 1 1]);
            ydd1 = circshift(imfilter(cdd1,hdd1,'conv','circular'),[1 1 1]);        
            yha2 = circshift(imfilter(cha2,hha2,'conv','circular'),[1 1 1]);
            yva2 = circshift(imfilter(cva2,hva2,'conv','circular'),[1 1 1]);
            yda2 = circshift(imfilter(cda2,hda2,'conv','circular'),[1 1 1]);
            yad2 = circshift(imfilter(cad2,had2,'conv','circular'),[1 1 1]);
            yhd2 = circshift(imfilter(chd2,hhd2,'conv','circular'),[1 1 1]);
            yvd2 = circshift(imfilter(cvd2,hvd2,'conv','circular'),[1 1 1]);
            ydd2 = circshift(imfilter(cdd2,hdd2,'conv','circular'),[1 1 1]);
            
            % Actual values
            imgActual = yaa1 + yha1 + yva1 + yda1 + yad1 + yhd1 + yvd1 + ydd1 ...
                + yha2 + yva2 + yda2 + yad2 + yhd2 + yvd2 + ydd2;
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test for default construction
        function testLevel3ReconstructionSize16x16(testCase)
            
            nLevels = 3;
            height = 16;
            width = 16;
            depth = 16;
                        
            % Expected values
            imgExpctd = rand(height,width,depth);
            dimExpctd = [ height, width, depth ];
            nPixels = numel(imgExpctd);

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels);
        
            % Analysis
            coefs = step(testCase.analyzer,imgExpctd);
            caa1 = reshape(coefs(1:nPixels),height,width,depth);
            cha1 = reshape(coefs(nPixels+1:2*nPixels),height,width,depth);
            cva1 = reshape(coefs(2*nPixels+1:3*nPixels),height,width,depth);
            cda1 = reshape(coefs(3*nPixels+1:4*nPixels),height,width,depth);
            cad1 = reshape(coefs(4*nPixels+1:5*nPixels),height,width,depth);
            chd1 = reshape(coefs(5*nPixels+1:6*nPixels),height,width,depth);
            cvd1 = reshape(coefs(6*nPixels+1:7*nPixels),height,width,depth);
            cdd1 = reshape(coefs(7*nPixels+1:8*nPixels),height,width,depth);            
            cha2 = reshape(coefs(8*nPixels+1:9*nPixels),height,width,depth);
            cva2 = reshape(coefs(9*nPixels+1:10*nPixels),height,width,depth);
            cda2 = reshape(coefs(10*nPixels+1:11*nPixels),height,width,depth);
            cad2 = reshape(coefs(11*nPixels+1:12*nPixels),height,width,depth);
            chd2 = reshape(coefs(12*nPixels+1:13*nPixels),height,width,depth);
            cvd2 = reshape(coefs(13*nPixels+1:14*nPixels),height,width,depth);
            cdd2 = reshape(coefs(14*nPixels+1:15*nPixels),height,width,depth);                    
            cha3 = reshape(coefs(15*nPixels+1:16*nPixels),height,width,depth);
            cva3 = reshape(coefs(16*nPixels+1:17*nPixels),height,width,depth);
            cda3 = reshape(coefs(17*nPixels+1:18*nPixels),height,width,depth);
            cad3 = reshape(coefs(18*nPixels+1:19*nPixels),height,width,depth);
            chd3 = reshape(coefs(19*nPixels+1:20*nPixels),height,width,depth);
            cvd3 = reshape(coefs(20*nPixels+1:21*nPixels),height,width,depth);
            cdd3 = reshape(coefs(21*nPixels+1:22*nPixels),height,width,depth);                                
         
           % Reconstruction
            upsample3_ = @(x,u,s)...
                    shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(x,...
                    u,s),1),...
                    u,s),1),...
                    u,s),1);        
            haa3 = testCase.kernels.haa;
            had3 = testCase.kernels.had;
            hva3 = testCase.kernels.hva;
            hvd3 = testCase.kernels.hvd;
            hha3 = testCase.kernels.hha;
            hhd3 = testCase.kernels.hhd;
            hda3 = testCase.kernels.hda;
            hdd3 = testCase.kernels.hdd;
            haa2 = imfilter(upsample3_(haa3,2,1),haa3);
            had2 = imfilter(upsample3_(had3,2,1),haa3);
            hva2 = imfilter(upsample3_(hva3,2,1),haa3);
            hvd2 = imfilter(upsample3_(hvd3,2,1),haa3);
            hha2 = imfilter(upsample3_(hha3,2,1),haa3);
            hhd2 = imfilter(upsample3_(hhd3,2,1),haa3);
            hda2 = imfilter(upsample3_(hda3,2,1),haa3);
            hdd2 = imfilter(upsample3_(hdd3,2,1),haa3);
            haa1 = imfilter(upsample3_(haa3,4,2),haa2);
            had1 = imfilter(upsample3_(had3,4,2),haa2);
            hva1 = imfilter(upsample3_(hva3,4,2),haa2);
            hvd1 = imfilter(upsample3_(hvd3,4,2),haa2);
            hha1 = imfilter(upsample3_(hha3,4,2),haa2);
            hhd1 = imfilter(upsample3_(hhd3,4,2),haa2);
            hda1 = imfilter(upsample3_(hda3,4,2),haa2);
            hdd1 = imfilter(upsample3_(hdd3,4,2),haa2);            
            %
            yaa1 = circshift(imfilter(caa1,haa1,'conv','circular'),[1 1 1]);
            yha1 = circshift(imfilter(cha1,hha1,'conv','circular'),[1 1 1]);
            yva1 = circshift(imfilter(cva1,hva1,'conv','circular'),[1 1 1]);
            yda1 = circshift(imfilter(cda1,hda1,'conv','circular'),[1 1 1]);
            yad1 = circshift(imfilter(cad1,had1,'conv','circular'),[1 1 1]);
            yhd1 = circshift(imfilter(chd1,hhd1,'conv','circular'),[1 1 1]);
            yvd1 = circshift(imfilter(cvd1,hvd1,'conv','circular'),[1 1 1]);
            ydd1 = circshift(imfilter(cdd1,hdd1,'conv','circular'),[1 1 1]);        
            yha2 = circshift(imfilter(cha2,hha2,'conv','circular'),[1 1 1]);
            yva2 = circshift(imfilter(cva2,hva2,'conv','circular'),[1 1 1]);
            yda2 = circshift(imfilter(cda2,hda2,'conv','circular'),[1 1 1]);
            yad2 = circshift(imfilter(cad2,had2,'conv','circular'),[1 1 1]);
            yhd2 = circshift(imfilter(chd2,hhd2,'conv','circular'),[1 1 1]);
            yvd2 = circshift(imfilter(cvd2,hvd2,'conv','circular'),[1 1 1]);
            ydd2 = circshift(imfilter(cdd2,hdd2,'conv','circular'),[1 1 1]);        
            yha3 = circshift(imfilter(cha3,hha3,'conv','circular'),[1 1 1]);
            yva3 = circshift(imfilter(cva3,hva3,'conv','circular'),[1 1 1]);
            yda3 = circshift(imfilter(cda3,hda3,'conv','circular'),[1 1 1]);
            yad3 = circshift(imfilter(cad3,had3,'conv','circular'),[1 1 1]);
            yhd3 = circshift(imfilter(chd3,hhd3,'conv','circular'),[1 1 1]);
            yvd3 = circshift(imfilter(cvd3,hvd3,'conv','circular'),[1 1 1]);
            ydd3 = circshift(imfilter(cdd3,hdd3,'conv','circular'),[1 1 1]);
            
            % Actual values
            imgActual = yaa1 + yha1 + yva1 + yda1 + yad1 + yhd1 + yvd1 + ydd1 ...
                + yha2 + yva2 + yda2 + yad2 + yhd2 + yvd2 + ydd2 ...
                + yha3 + yva3 + yda3 + yad3 + yhd3 + yvd3 + ydd3;
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test for default construction
        function testLevel3ReconstructionSize16x32x64(testCase)
            
            nLevels = 3;
            height = 16;
            width = 32;
            depth = 64;
                        
            % Expected values
            imgExpctd = rand(height,width,depth);
            dimExpctd = [ height, width, depth ];
            nPixels = numel(imgExpctd);

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels);
        
            % Analysis
            coefs = step(testCase.analyzer,imgExpctd);
            caa1 = reshape(coefs(1:nPixels),height,width,depth);
            cha1 = reshape(coefs(nPixels+1:2*nPixels),height,width,depth);
            cva1 = reshape(coefs(2*nPixels+1:3*nPixels),height,width,depth);
            cda1 = reshape(coefs(3*nPixels+1:4*nPixels),height,width,depth);
            cad1 = reshape(coefs(4*nPixels+1:5*nPixels),height,width,depth);
            chd1 = reshape(coefs(5*nPixels+1:6*nPixels),height,width,depth);
            cvd1 = reshape(coefs(6*nPixels+1:7*nPixels),height,width,depth);
            cdd1 = reshape(coefs(7*nPixels+1:8*nPixels),height,width,depth);            
            cha2 = reshape(coefs(8*nPixels+1:9*nPixels),height,width,depth);
            cva2 = reshape(coefs(9*nPixels+1:10*nPixels),height,width,depth);
            cda2 = reshape(coefs(10*nPixels+1:11*nPixels),height,width,depth);
            cad2 = reshape(coefs(11*nPixels+1:12*nPixels),height,width,depth);
            chd2 = reshape(coefs(12*nPixels+1:13*nPixels),height,width,depth);
            cvd2 = reshape(coefs(13*nPixels+1:14*nPixels),height,width,depth);
            cdd2 = reshape(coefs(14*nPixels+1:15*nPixels),height,width,depth);                    
            cha3 = reshape(coefs(15*nPixels+1:16*nPixels),height,width,depth);
            cva3 = reshape(coefs(16*nPixels+1:17*nPixels),height,width,depth);
            cda3 = reshape(coefs(17*nPixels+1:18*nPixels),height,width,depth);
            cad3 = reshape(coefs(18*nPixels+1:19*nPixels),height,width,depth);
            chd3 = reshape(coefs(19*nPixels+1:20*nPixels),height,width,depth);
            cvd3 = reshape(coefs(20*nPixels+1:21*nPixels),height,width,depth);
            cdd3 = reshape(coefs(21*nPixels+1:22*nPixels),height,width,depth);                                
         
           % Reconstruction
            upsample3_ = @(x,u,s)...
                    shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(x,...
                    u,s),1),...
                    u,s),1),...
                    u,s),1);        
            haa3 = testCase.kernels.haa;
            had3 = testCase.kernels.had;
            hva3 = testCase.kernels.hva;
            hvd3 = testCase.kernels.hvd;
            hha3 = testCase.kernels.hha;
            hhd3 = testCase.kernels.hhd;
            hda3 = testCase.kernels.hda;
            hdd3 = testCase.kernels.hdd;
            haa2 = imfilter(upsample3_(haa3,2,1),haa3);
            had2 = imfilter(upsample3_(had3,2,1),haa3);
            hva2 = imfilter(upsample3_(hva3,2,1),haa3);
            hvd2 = imfilter(upsample3_(hvd3,2,1),haa3);
            hha2 = imfilter(upsample3_(hha3,2,1),haa3);
            hhd2 = imfilter(upsample3_(hhd3,2,1),haa3);
            hda2 = imfilter(upsample3_(hda3,2,1),haa3);
            hdd2 = imfilter(upsample3_(hdd3,2,1),haa3);
            haa1 = imfilter(upsample3_(haa3,4,2),haa2);
            had1 = imfilter(upsample3_(had3,4,2),haa2);
            hva1 = imfilter(upsample3_(hva3,4,2),haa2);
            hvd1 = imfilter(upsample3_(hvd3,4,2),haa2);
            hha1 = imfilter(upsample3_(hha3,4,2),haa2);
            hhd1 = imfilter(upsample3_(hhd3,4,2),haa2);
            hda1 = imfilter(upsample3_(hda3,4,2),haa2);
            hdd1 = imfilter(upsample3_(hdd3,4,2),haa2);            
            %
            yaa1 = circshift(imfilter(caa1,haa1,'conv','circular'),[1 1 1]);
            yha1 = circshift(imfilter(cha1,hha1,'conv','circular'),[1 1 1]);
            yva1 = circshift(imfilter(cva1,hva1,'conv','circular'),[1 1 1]);
            yda1 = circshift(imfilter(cda1,hda1,'conv','circular'),[1 1 1]);
            yad1 = circshift(imfilter(cad1,had1,'conv','circular'),[1 1 1]);
            yhd1 = circshift(imfilter(chd1,hhd1,'conv','circular'),[1 1 1]);
            yvd1 = circshift(imfilter(cvd1,hvd1,'conv','circular'),[1 1 1]);
            ydd1 = circshift(imfilter(cdd1,hdd1,'conv','circular'),[1 1 1]);        
            yha2 = circshift(imfilter(cha2,hha2,'conv','circular'),[1 1 1]);
            yva2 = circshift(imfilter(cva2,hva2,'conv','circular'),[1 1 1]);
            yda2 = circshift(imfilter(cda2,hda2,'conv','circular'),[1 1 1]);
            yad2 = circshift(imfilter(cad2,had2,'conv','circular'),[1 1 1]);
            yhd2 = circshift(imfilter(chd2,hhd2,'conv','circular'),[1 1 1]);
            yvd2 = circshift(imfilter(cvd2,hvd2,'conv','circular'),[1 1 1]);
            ydd2 = circshift(imfilter(cdd2,hdd2,'conv','circular'),[1 1 1]);        
            yha3 = circshift(imfilter(cha3,hha3,'conv','circular'),[1 1 1]);
            yva3 = circshift(imfilter(cva3,hva3,'conv','circular'),[1 1 1]);
            yda3 = circshift(imfilter(cda3,hda3,'conv','circular'),[1 1 1]);
            yad3 = circshift(imfilter(cad3,had3,'conv','circular'),[1 1 1]);
            yhd3 = circshift(imfilter(chd3,hhd3,'conv','circular'),[1 1 1]);
            yvd3 = circshift(imfilter(cvd3,hvd3,'conv','circular'),[1 1 1]);
            ydd3 = circshift(imfilter(cdd3,hdd3,'conv','circular'),[1 1 1]);
            
            % Actual values
            imgActual = yaa1 + yha1 + yva1 + yda1 + yad1 + yhd1 + yvd1 + ydd1 ...
                + yha2 + yva2 + yda2 + yad2 + yhd2 + yvd2 + ydd2 ...
                + yha3 + yva3 + yda3 + yad3 + yhd3 + yvd3 + ydd3;
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
     
        % Test for default construction
        function testLevel4Size32x32x32(testCase,useparallel)
            
            nLevels = 4;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            
            % Expected values
            upsample3_ = @(x,u,s)...
                    shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(x,...
                    u,s),1),...
                    u,s),1),...
                    u,s),1);
            haa4 = testCase.kernels.haa;
            had4 = testCase.kernels.had;
            hva4 = testCase.kernels.hva;
            hvd4 = testCase.kernels.hvd;
            hha4 = testCase.kernels.hha;
            hhd4 = testCase.kernels.hhd;
            hda4 = testCase.kernels.hda;
            hdd4 = testCase.kernels.hdd;
            haa3 = imfilter(upsample3_(haa4,2,1),haa4);
            had3 = imfilter(upsample3_(had4,2,1),haa4);
            hva3 = imfilter(upsample3_(hva4,2,1),haa4);
            hvd3 = imfilter(upsample3_(hvd4,2,1),haa4);
            hha3 = imfilter(upsample3_(hha4,2,1),haa4);
            hhd3 = imfilter(upsample3_(hhd4,2,1),haa4);
            hda3 = imfilter(upsample3_(hda4,2,1),haa4);
            hdd3 = imfilter(upsample3_(hdd4,2,1),haa4);
            haa2 = imfilter(upsample3_(haa4,4,2),haa3);
            had2 = imfilter(upsample3_(had4,4,2),haa3);
            hva2 = imfilter(upsample3_(hva4,4,2),haa3);
            hvd2 = imfilter(upsample3_(hvd4,4,2),haa3);
            hha2 = imfilter(upsample3_(hha4,4,2),haa3);
            hhd2 = imfilter(upsample3_(hhd4,4,2),haa3);
            hda2 = imfilter(upsample3_(hda4,4,2),haa3);
            hdd2 = imfilter(upsample3_(hdd4,4,2),haa3);            
            haa1 = imfilter(upsample3_(haa4,8,4),haa2);
            had1 = imfilter(upsample3_(had4,8,4),haa2);
            hva1 = imfilter(upsample3_(hva4,8,4),haa2);
            hvd1 = imfilter(upsample3_(hvd4,8,4),haa2);
            hha1 = imfilter(upsample3_(hha4,8,4),haa2);
            hhd1 = imfilter(upsample3_(hhd4,8,4),haa2);
            hda1 = imfilter(upsample3_(hda4,8,4),haa2);
            hdd1 = imfilter(upsample3_(hdd4,8,4),haa2);                        
            yaa1 = imfilter(srcImg,haa1,'corr','circular');
            yad1 = imfilter(srcImg,had1,'corr','circular');
            yva1 = imfilter(srcImg,hva1,'corr','circular');
            yvd1 = imfilter(srcImg,hvd1,'corr','circular');
            yha1 = imfilter(srcImg,hha1,'corr','circular');
            yhd1 = imfilter(srcImg,hhd1,'corr','circular');
            yda1 = imfilter(srcImg,hda1,'corr','circular');
            ydd1 = imfilter(srcImg,hdd1,'corr','circular');            
            yad2 = imfilter(srcImg,had2,'corr','circular');
            yva2 = imfilter(srcImg,hva2,'corr','circular');
            yvd2 = imfilter(srcImg,hvd2,'corr','circular');
            yha2 = imfilter(srcImg,hha2,'corr','circular');
            yhd2 = imfilter(srcImg,hhd2,'corr','circular');
            yda2 = imfilter(srcImg,hda2,'corr','circular');
            ydd2 = imfilter(srcImg,hdd2,'corr','circular');
            yad3 = imfilter(srcImg,had3,'corr','circular');
            yva3 = imfilter(srcImg,hva3,'corr','circular');
            yvd3 = imfilter(srcImg,hvd3,'corr','circular');
            yha3 = imfilter(srcImg,hha3,'corr','circular');
            yhd3 = imfilter(srcImg,hhd3,'corr','circular');
            yda3 = imfilter(srcImg,hda3,'corr','circular');
            ydd3 = imfilter(srcImg,hdd3,'corr','circular');            
            yad4 = imfilter(srcImg,had4,'corr','circular');
            yva4 = imfilter(srcImg,hva4,'corr','circular');
            yvd4 = imfilter(srcImg,hvd4,'corr','circular');
            yha4 = imfilter(srcImg,hha4,'corr','circular');
            yhd4 = imfilter(srcImg,hhd4,'corr','circular');
            yda4 = imfilter(srcImg,hda4,'corr','circular');
            ydd4 = imfilter(srcImg,hdd4,'corr','circular');            
            coefExpctd = [
                yaa1(:) 
                yha1(:)                
                yva1(:)                
                yda1(:)                
                yad1(:)
                yhd1(:)                
                yvd1(:)
                ydd1(:)
                yha2(:)                
                yva2(:)                
                yda2(:)                
                yad2(:)
                yhd2(:)                
                yvd2(:)
                ydd2(:)                
                yha3(:)                
                yva3(:)                
                yda3(:)                
                yad3(:)
                yhd3(:)                
                yvd3(:)
                ydd3(:)                                
                yha4(:)                
                yva4(:)                
                yda4(:)                
                yad4(:)
                yhd4(:)                
                yvd4(:)
                ydd4(:)                                
                ].';            
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem(...
                'NumberOfLevels',nLevels,...
                'UseParallel',useparallel);
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-5,sprintf('%g',diff));

        end
        
        %{
        % Test for default construction
        function testLevel5Size64x64x64(testCase)
            
            nLevels = 5;
            height = 64;
            width = 64;
            depth = 64;
            
            % Expected values            
            srcImg = rand(height,width,depth);
            
            % Expected values
            upsample3_ = @(x,u,s)...
                    shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(x,...
                    u,s),1),...
                    u,s),1),...
                    u,s),1);
            haa5 = testCase.kernels.haa;
            had5 = testCase.kernels.had;
            hva5 = testCase.kernels.hva;
            hvd5 = testCase.kernels.hvd;
            hha5 = testCase.kernels.hha;
            hhd5 = testCase.kernels.hhd;
            hda5 = testCase.kernels.hda;
            hdd5 = testCase.kernels.hdd;
            haa4 = imfilter(upsample3_(haa5,2,1),haa5);
            had4 = imfilter(upsample3_(had5,2,1),haa5);
            hva4 = imfilter(upsample3_(hva5,2,1),haa5);
            hvd4 = imfilter(upsample3_(hvd5,2,1),haa5);
            hha4 = imfilter(upsample3_(hha5,2,1),haa5);
            hhd4 = imfilter(upsample3_(hhd5,2,1),haa5);
            hda4 = imfilter(upsample3_(hda5,2,1),haa5);
            hdd4 = imfilter(upsample3_(hdd5,2,1),haa5);
            haa3 = imfilter(upsample3_(haa5,4,2),haa4);
            had3 = imfilter(upsample3_(had5,4,2),haa4);
            hva3 = imfilter(upsample3_(hva5,4,2),haa4);
            hvd3 = imfilter(upsample3_(hvd5,4,2),haa4);
            hha3 = imfilter(upsample3_(hha5,4,2),haa4);
            hhd3 = imfilter(upsample3_(hhd5,4,2),haa4);
            hda3 = imfilter(upsample3_(hda5,4,2),haa4);
            hdd3 = imfilter(upsample3_(hdd5,4,2),haa4);            
            haa2 = imfilter(upsample3_(haa5,8,4),haa3);
            had2 = imfilter(upsample3_(had5,8,4),haa3);
            hva2 = imfilter(upsample3_(hva5,8,4),haa3);
            hvd2 = imfilter(upsample3_(hvd5,8,4),haa3);
            hha2 = imfilter(upsample3_(hha5,8,4),haa3);
            hhd2 = imfilter(upsample3_(hhd5,8,4),haa3);
            hda2 = imfilter(upsample3_(hda5,8,4),haa3);
            hdd2 = imfilter(upsample3_(hdd5,8,4),haa3);                        
            haa1 = imfilter(upsample3_(haa5,16,8),haa2);
            had1 = imfilter(upsample3_(had5,16,8),haa2);
            hva1 = imfilter(upsample3_(hva5,16,8),haa2);
            hvd1 = imfilter(upsample3_(hvd5,16,8),haa2);
            hha1 = imfilter(upsample3_(hha5,16,8),haa2);
            hhd1 = imfilter(upsample3_(hhd5,16,8),haa2);
            hda1 = imfilter(upsample3_(hda5,16,8),haa2);
            hdd1 = imfilter(upsample3_(hdd5,16,8),haa2);                                    
            yaa1 = imfilter(srcImg,haa1,'corr','circular');
            yad1 = imfilter(srcImg,had1,'corr','circular');
            yva1 = imfilter(srcImg,hva1,'corr','circular');
            yvd1 = imfilter(srcImg,hvd1,'corr','circular');
            yha1 = imfilter(srcImg,hha1,'corr','circular');
            yhd1 = imfilter(srcImg,hhd1,'corr','circular');
            yda1 = imfilter(srcImg,hda1,'corr','circular');
            ydd1 = imfilter(srcImg,hdd1,'corr','circular');            
            yad2 = imfilter(srcImg,had2,'corr','circular');
            yva2 = imfilter(srcImg,hva2,'corr','circular');
            yvd2 = imfilter(srcImg,hvd2,'corr','circular');
            yha2 = imfilter(srcImg,hha2,'corr','circular');
            yhd2 = imfilter(srcImg,hhd2,'corr','circular');
            yda2 = imfilter(srcImg,hda2,'corr','circular');
            ydd2 = imfilter(srcImg,hdd2,'corr','circular');
            yad3 = imfilter(srcImg,had3,'corr','circular');
            yva3 = imfilter(srcImg,hva3,'corr','circular');
            yvd3 = imfilter(srcImg,hvd3,'corr','circular');
            yha3 = imfilter(srcImg,hha3,'corr','circular');
            yhd3 = imfilter(srcImg,hhd3,'corr','circular');
            yda3 = imfilter(srcImg,hda3,'corr','circular');
            ydd3 = imfilter(srcImg,hdd3,'corr','circular');            
            yad4 = imfilter(srcImg,had4,'corr','circular');
            yva4 = imfilter(srcImg,hva4,'corr','circular');
            yvd4 = imfilter(srcImg,hvd4,'corr','circular');
            yha4 = imfilter(srcImg,hha4,'corr','circular');
            yhd4 = imfilter(srcImg,hhd4,'corr','circular');
            yda4 = imfilter(srcImg,hda4,'corr','circular');
            ydd4 = imfilter(srcImg,hdd4,'corr','circular');            
            yad5 = imfilter(srcImg,had5,'corr','circular');
            yva5 = imfilter(srcImg,hva5,'corr','circular');
            yvd5 = imfilter(srcImg,hvd5,'corr','circular');
            yha5 = imfilter(srcImg,hha5,'corr','circular');
            yhd5 = imfilter(srcImg,hhd5,'corr','circular');
            yda5 = imfilter(srcImg,hda5,'corr','circular');
            ydd5 = imfilter(srcImg,hdd5,'corr','circular');            
            coefExpctd = [
                yaa1(:) 
                yha1(:)                
                yva1(:)                
                yda1(:)                
                yad1(:)
                yhd1(:)                
                yvd1(:)
                ydd1(:)
                yha2(:)                
                yva2(:)                
                yda2(:)                
                yad2(:)
                yhd2(:)                
                yvd2(:)
                ydd2(:)                
                yha3(:)                
                yva3(:)                
                yda3(:)                
                yad3(:)
                yhd3(:)                
                yvd3(:)
                ydd3(:)                                
                yha4(:)                
                yva4(:)                
                yda4(:)                
                yad4(:)
                yhd4(:)                
                yvd4(:)
                ydd4(:)                                
                yha5(:)                
                yva5(:)                
                yda5(:)                
                yad5(:)
                yhd5(:)                
                yvd5(:)
                ydd5(:)                                                
                ].';            
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis3dSystem();
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-6,sprintf('%g',diff));

        end
        %}
    end
    
end