classdef UdHaarSynthesis3dSystemTestCase < matlab.unittest.TestCase
    %UDHAARSYNTHESIZERTESTCASE Test case for UdHaarSynthesis3dSystem
    %
    % SVN identifier:
    % $Id: UdHaarSynthesis3dSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
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
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    
    properties (TestParameter)
        useparallel = { true, false };
    end       
    properties
        synthesizer
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
        function deleteObject(testCase)
            delete(testCase.synthesizer);
        end
    end
    
    methods (Test)
        
        % Test for default construction
        function testLevel1Size16x16x16(testCase)
            
            nLevels = 1;
            height = 16;
            width = 16;
            depth = 16;
            caa  = rand(height,width,depth);
            cha  = rand(height,width,depth);
            cva  = rand(height,width,depth);
            cda  = rand(height,width,depth);
            cad  = rand(height,width,depth);
            chd  = rand(height,width,depth);
            cvd  = rand(height,width,depth);
            cdd  = rand(height,width,depth);
            subCoefs = [
                caa(:)
                cha(:)
                cva(:)
                cda(:)
                cad(:)
                chd(:)
                cvd(:)
                cdd(:) ].';
            
            % Expected values
            haa = testCase.kernels.haa;
            hha = testCase.kernels.hha;
            hva = testCase.kernels.hva;
            hda = testCase.kernels.hda;
            had = testCase.kernels.had;
            hhd = testCase.kernels.hhd;
            hvd = testCase.kernels.hvd;
            hdd = testCase.kernels.hdd;
            yaa = circshift(imfilter(caa,haa,'conv','circular'),[1 1 1]);
            yha = circshift(imfilter(cha,hha,'conv','circular'),[1 1 1]);
            yva = circshift(imfilter(cva,hva,'conv','circular'),[1 1 1]);
            yda = circshift(imfilter(cda,hda,'conv','circular'),[1 1 1]);
            yad = circshift(imfilter(cad,had,'conv','circular'),[1 1 1]);
            yhd = circshift(imfilter(chd,hhd,'conv','circular'),[1 1 1]);
            yvd = circshift(imfilter(cvd,hvd,'conv','circular'),[1 1 1]);
            ydd = circshift(imfilter(cdd,hdd,'conv','circular'),[1 1 1]);
            imgExpctd = yaa + yha + yva + yda + yad + yhd + yvd + ydd;
            dimExpctd = [ height width depth ];
            scalesExpctd = repmat([ height width depth ],[7*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        % Test for default construction
        function testLevel1Size16x32x64(testCase)
            
            nLevels = 1;
            height = 16;
            width = 32;
            depth = 64;
            caa  = rand(height,width,depth);
            cha  = rand(height,width,depth);
            cva  = rand(height,width,depth);
            cda  = rand(height,width,depth);
            cad  = rand(height,width,depth);
            chd  = rand(height,width,depth);
            cvd  = rand(height,width,depth);
            cdd  = rand(height,width,depth);
            subCoefs = [
                caa(:)
                cha(:)
                cva(:)
                cda(:)
                cad(:)
                chd(:)
                cvd(:)
                cdd(:) ].';
            
            % Expected values
            haa = testCase.kernels.haa;
            hha = testCase.kernels.hha;
            hva = testCase.kernels.hva;
            hda = testCase.kernels.hda;
            had = testCase.kernels.had;
            hhd = testCase.kernels.hhd;
            hvd = testCase.kernels.hvd;
            hdd = testCase.kernels.hdd;
            yaa = circshift(imfilter(caa,haa,'conv','circular'),[1 1 1]);
            yha = circshift(imfilter(cha,hha,'conv','circular'),[1 1 1]);
            yva = circshift(imfilter(cva,hva,'conv','circular'),[1 1 1]);
            yda = circshift(imfilter(cda,hda,'conv','circular'),[1 1 1]);
            yad = circshift(imfilter(cad,had,'conv','circular'),[1 1 1]);
            yhd = circshift(imfilter(chd,hhd,'conv','circular'),[1 1 1]);
            yvd = circshift(imfilter(cvd,hvd,'conv','circular'),[1 1 1]);
            ydd = circshift(imfilter(cdd,hdd,'conv','circular'),[1 1 1]);
            imgExpctd = yaa + yha + yva + yda + yad + yhd + yvd + ydd;
            dimExpctd = [ height width depth ];
            scalesExpctd = repmat([ height width depth ],[7*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        % Test for default construction
        function testLevel2Size16x16(testCase)
            
            nLevels = 2;
            height = 16;
            width = 16;
            depth = 16;
            caa1 = rand(height,width,depth);
            cha1 = rand(height,width,depth);
            cva1 = rand(height,width,depth);
            cda1 = rand(height,width,depth);
            cad1 = rand(height,width,depth);
            chd1 = rand(height,width,depth);
            cvd1 = rand(height,width,depth);
            cdd1 = rand(height,width,depth);
            cha2 = rand(height,width,depth);
            cva2 = rand(height,width,depth);
            cda2 = rand(height,width,depth);
            cad2 = rand(height,width,depth);
            chd2 = rand(height,width,depth);
            cvd2 = rand(height,width,depth);
            cdd2 = rand(height,width,depth);
            subCoefs = [
                caa1(:)
                cha1(:)
                cva1(:)
                cda1(:)
                cad1(:)
                chd1(:)
                cvd1(:)
                cdd1(:)
                cha2(:)
                cva2(:)
                cda2(:)
                cad2(:)
                chd2(:)
                cvd2(:)
                cdd2(:) ].';
            
            % Expected values
            upsample3_ = @(x,u,s) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,...
                u,s),1),...
                u,s),1),...
                u,s),1);
            haa2 = testCase.kernels.haa;
            hha2 = testCase.kernels.hha;
            hva2 = testCase.kernels.hva;
            hda2 = testCase.kernels.hda;
            had2 = testCase.kernels.had;
            hhd2 = testCase.kernels.hhd;
            hvd2 = testCase.kernels.hvd;
            hdd2 = testCase.kernels.hdd;
            haa1 = imfilter(upsample3_(haa2,2,1),haa2);
            hha1 = imfilter(upsample3_(hha2,2,1),haa2);
            hva1 = imfilter(upsample3_(hva2,2,1),haa2);
            hda1 = imfilter(upsample3_(hda2,2,1),haa2);
            had1 = imfilter(upsample3_(had2,2,1),haa2);
            hhd1 = imfilter(upsample3_(hhd2,2,1),haa2);
            hvd1 = imfilter(upsample3_(hvd2,2,1),haa2);
            hdd1 = imfilter(upsample3_(hdd2,2,1),haa2);
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
            imgExpctd = yaa1 + yha1 + yva1 + yda1 + yad1 + yhd1 + yvd1 + ydd1 ...
                + yha2 + yva2 + yda2 + yad2 + yhd2 + yvd2 + ydd2;
            dimExpctd = [ height width depth ];
            scalesExpctd = repmat([ height width depth ], [7*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        % Test for default construction
        function testLevel2Size16x32x64(testCase)
            
            nLevels = 2;
            height = 16;
            width = 32;
            depth = 64;
            caa1 = rand(height,width,depth);
            cha1 = rand(height,width,depth);
            cva1 = rand(height,width,depth);
            cda1 = rand(height,width,depth);
            cad1 = rand(height,width,depth);
            chd1 = rand(height,width,depth);
            cvd1 = rand(height,width,depth);
            cdd1 = rand(height,width,depth);
            cha2 = rand(height,width,depth);
            cva2 = rand(height,width,depth);
            cda2 = rand(height,width,depth);
            cad2 = rand(height,width,depth);
            chd2 = rand(height,width,depth);
            cvd2 = rand(height,width,depth);
            cdd2 = rand(height,width,depth);
            subCoefs = [
                caa1(:)
                cha1(:)
                cva1(:)
                cda1(:)
                cad1(:)
                chd1(:)
                cvd1(:)
                cdd1(:)
                cha2(:)
                cva2(:)
                cda2(:)
                cad2(:)
                chd2(:)
                cvd2(:)
                cdd2(:) ].';
            
            % Expected values
            upsample3_ = @(x,u,s) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,...
                u,s),1),...
                u,s),1),...
                u,s),1);
            haa2 = testCase.kernels.haa;
            hha2 = testCase.kernels.hha;
            hva2 = testCase.kernels.hva;
            hda2 = testCase.kernels.hda;
            had2 = testCase.kernels.had;
            hhd2 = testCase.kernels.hhd;
            hvd2 = testCase.kernels.hvd;
            hdd2 = testCase.kernels.hdd;
            haa1 = imfilter(upsample3_(haa2,2,1),haa2);
            hha1 = imfilter(upsample3_(hha2,2,1),haa2);
            hva1 = imfilter(upsample3_(hva2,2,1),haa2);
            hda1 = imfilter(upsample3_(hda2,2,1),haa2);
            had1 = imfilter(upsample3_(had2,2,1),haa2);
            hhd1 = imfilter(upsample3_(hhd2,2,1),haa2);
            hvd1 = imfilter(upsample3_(hvd2,2,1),haa2);
            hdd1 = imfilter(upsample3_(hdd2,2,1),haa2);
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
            imgExpctd = yaa1 + yha1 + yva1 + yda1 + yad1 + yhd1 + yvd1 + ydd1 ...
                + yha2 + yva2 + yda2 + yad2 + yhd2 + yvd2 + ydd2;
            dimExpctd = [ height width depth ];
            scalesExpctd = repmat([ height width depth ], [7*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        % Test for default construction
        function testLevel3Size16x16x16(testCase)
            
            nLevels = 3;
            height = 16;
            width = 16;
            depth = 16;
            caa1 = rand(height,width,depth);
            cha1 = rand(height,width,depth);
            cva1 = rand(height,width,depth);
            cda1 = rand(height,width,depth);
            cad1 = rand(height,width,depth);
            chd1 = rand(height,width,depth);
            cvd1 = rand(height,width,depth);
            cdd1 = rand(height,width,depth);
            cha2 = rand(height,width,depth);
            cva2 = rand(height,width,depth);
            cda2 = rand(height,width,depth);
            cad2 = rand(height,width,depth);
            chd2 = rand(height,width,depth);
            cvd2 = rand(height,width,depth);
            cdd2 = rand(height,width,depth);
            cha3 = rand(height,width,depth);
            cva3 = rand(height,width,depth);
            cda3 = rand(height,width,depth);
            cad3 = rand(height,width,depth);
            chd3 = rand(height,width,depth);
            cvd3 = rand(height,width,depth);
            cdd3 = rand(height,width,depth);
            subCoefs = [
                caa1(:)
                cha1(:)
                cva1(:)
                cda1(:)
                cad1(:)
                chd1(:)
                cvd1(:)
                cdd1(:)
                cha2(:)
                cva2(:)
                cda2(:)
                cad2(:)
                chd2(:)
                cvd2(:)
                cdd2(:)
                cha3(:)
                cva3(:)
                cda3(:)
                cad3(:)
                chd3(:)
                cvd3(:)
                cdd3(:) ].';
            
            % Expected values
            upsample3_ = @(x,u,s) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,...
                u,s),1),...
                u,s),1),...
                u,s),1);
            haa3 = testCase.kernels.haa;
            hha3 = testCase.kernels.hha;
            hva3 = testCase.kernels.hva;
            hda3 = testCase.kernels.hda;
            had3 = testCase.kernels.had;
            hhd3 = testCase.kernels.hhd;
            hvd3 = testCase.kernels.hvd;
            hdd3 = testCase.kernels.hdd;
            haa2 = imfilter(upsample3_(haa3,2,1),haa3);
            hha2 = imfilter(upsample3_(hha3,2,1),haa3);
            hva2 = imfilter(upsample3_(hva3,2,1),haa3);
            hda2 = imfilter(upsample3_(hda3,2,1),haa3);
            had2 = imfilter(upsample3_(had3,2,1),haa3);
            hhd2 = imfilter(upsample3_(hhd3,2,1),haa3);
            hvd2 = imfilter(upsample3_(hvd3,2,1),haa3);
            hdd2 = imfilter(upsample3_(hdd3,2,1),haa3);
            haa1 = imfilter(upsample3_(haa3,4,2),haa2);
            hha1 = imfilter(upsample3_(hha3,4,2),haa2);
            hva1 = imfilter(upsample3_(hva3,4,2),haa2);
            hda1 = imfilter(upsample3_(hda3,4,2),haa2);
            had1 = imfilter(upsample3_(had3,4,2),haa2);
            hhd1 = imfilter(upsample3_(hhd3,4,2),haa2);
            hvd1 = imfilter(upsample3_(hvd3,4,2),haa2);
            hdd1 = imfilter(upsample3_(hdd3,4,2),haa2);
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
            imgExpctd = yaa1 + yha1 + yva1 + yda1 + yad1 + yhd1 + yvd1 + ydd1 ...
                + yha2 + yva2 + yda2 + yad2 + yhd2 + yvd2 + ydd2 ...
                + yha3 + yva3 + yda3 + yad3 + yhd3 + yvd3 + ydd3;
            dimExpctd = [ height width depth ];
            scalesExpctd = repmat([ height width depth ], [7*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        % Test for default construction
        function testLevel3Size16x32x64(testCase)
            
            nLevels = 3;
            height = 16;
            width = 32;
            depth = 64;
            caa1 = rand(height,width,depth);
            cha1 = rand(height,width,depth);
            cva1 = rand(height,width,depth);
            cda1 = rand(height,width,depth);
            cad1 = rand(height,width,depth);
            chd1 = rand(height,width,depth);
            cvd1 = rand(height,width,depth);
            cdd1 = rand(height,width,depth);
            cha2 = rand(height,width,depth);
            cva2 = rand(height,width,depth);
            cda2 = rand(height,width,depth);
            cad2 = rand(height,width,depth);
            chd2 = rand(height,width,depth);
            cvd2 = rand(height,width,depth);
            cdd2 = rand(height,width,depth);
            cha3 = rand(height,width,depth);
            cva3 = rand(height,width,depth);
            cda3 = rand(height,width,depth);
            cad3 = rand(height,width,depth);
            chd3 = rand(height,width,depth);
            cvd3 = rand(height,width,depth);
            cdd3 = rand(height,width,depth);
            subCoefs = [
                caa1(:)
                cha1(:)
                cva1(:)
                cda1(:)
                cad1(:)
                chd1(:)
                cvd1(:)
                cdd1(:)
                cha2(:)
                cva2(:)
                cda2(:)
                cad2(:)
                chd2(:)
                cvd2(:)
                cdd2(:)
                cha3(:)
                cva3(:)
                cda3(:)
                cad3(:)
                chd3(:)
                cvd3(:)
                cdd3(:) ].';
            
            % Expected values
            upsample3_ = @(x,u,s) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,...
                u,s),1),...
                u,s),1),...
                u,s),1);
            haa3 = testCase.kernels.haa;
            hha3 = testCase.kernels.hha;
            hva3 = testCase.kernels.hva;
            hda3 = testCase.kernels.hda;
            had3 = testCase.kernels.had;
            hhd3 = testCase.kernels.hhd;
            hvd3 = testCase.kernels.hvd;
            hdd3 = testCase.kernels.hdd;
            haa2 = imfilter(upsample3_(haa3,2,1),haa3);
            hha2 = imfilter(upsample3_(hha3,2,1),haa3);
            hva2 = imfilter(upsample3_(hva3,2,1),haa3);
            hda2 = imfilter(upsample3_(hda3,2,1),haa3);
            had2 = imfilter(upsample3_(had3,2,1),haa3);
            hhd2 = imfilter(upsample3_(hhd3,2,1),haa3);
            hvd2 = imfilter(upsample3_(hvd3,2,1),haa3);
            hdd2 = imfilter(upsample3_(hdd3,2,1),haa3);
            haa1 = imfilter(upsample3_(haa3,4,2),haa2);
            hha1 = imfilter(upsample3_(hha3,4,2),haa2);
            hva1 = imfilter(upsample3_(hva3,4,2),haa2);
            hda1 = imfilter(upsample3_(hda3,4,2),haa2);
            had1 = imfilter(upsample3_(had3,4,2),haa2);
            hhd1 = imfilter(upsample3_(hhd3,4,2),haa2);
            hvd1 = imfilter(upsample3_(hvd3,4,2),haa2);
            hdd1 = imfilter(upsample3_(hdd3,4,2),haa2);
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
            imgExpctd = yaa1 + yha1 + yva1 + yda1 + yad1 + yhd1 + yvd1 + ydd1 ...
                + yha2 + yva2 + yda2 + yad2 + yhd2 + yvd2 + ydd2 ...
                + yha3 + yva3 + yda3 + yad3 + yhd3 + yvd3 + ydd3;
            dimExpctd = [ height width depth ];
            scalesExpctd = repmat([ height width depth ], [7*nLevels+1, 1]);
            
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
 
        % Test for default construction
        function testLevel1ReconstructionSize16x16x16(testCase)
            
            nLevels = 1;
            height = 16;
            width = 16;
            depth = 16;
            
            % Expected values
            imgExpctd = rand(height,width,depth);
            dimExpctd = [ height width depth];
            
            % Coefs
            haa = testCase.kernels.haa;
            hha = testCase.kernels.hha;
            hva = testCase.kernels.hva;
            hda = testCase.kernels.hda;
            had = testCase.kernels.had;
            hhd = testCase.kernels.hhd;
            hvd = testCase.kernels.hvd;
            hdd = testCase.kernels.hdd;            
            yaa = imfilter(imgExpctd,haa,'corr','circular');
            yha = imfilter(imgExpctd,hha,'corr','circular');
            yva = imfilter(imgExpctd,hva,'corr','circular');
            yda = imfilter(imgExpctd,hda,'corr','circular');
            yad = imfilter(imgExpctd,had,'corr','circular');
            yhd = imfilter(imgExpctd,hhd,'corr','circular');
            yvd = imfilter(imgExpctd,hvd,'corr','circular');
            ydd = imfilter(imgExpctd,hdd,'corr','circular');            
            coef = [ 
                yaa(:)
                yha(:)
                yva(:)
                yda(:) 
                yad(:)
                yhd(:)
                yvd(:)
                ydd(:) ].';
            scales = repmat([ height width depth],[7*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coef,scales);
            
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
            dimExpctd = [ height width depth];
            
            % Coefs
            haa = testCase.kernels.haa;
            hha = testCase.kernels.hha;
            hva = testCase.kernels.hva;
            hda = testCase.kernels.hda;
            had = testCase.kernels.had;
            hhd = testCase.kernels.hhd;
            hvd = testCase.kernels.hvd;
            hdd = testCase.kernels.hdd;            
            yaa = imfilter(imgExpctd,haa,'corr','circular');
            yha = imfilter(imgExpctd,hha,'corr','circular');
            yva = imfilter(imgExpctd,hva,'corr','circular');
            yda = imfilter(imgExpctd,hda,'corr','circular');
            yad = imfilter(imgExpctd,had,'corr','circular');
            yhd = imfilter(imgExpctd,hhd,'corr','circular');
            yvd = imfilter(imgExpctd,hvd,'corr','circular');
            ydd = imfilter(imgExpctd,hdd,'corr','circular');            
            coef = [ 
                yaa(:)
                yha(:)
                yva(:)
                yda(:) 
                yad(:)
                yhd(:)
                yvd(:)
                ydd(:) ].';
            scales = repmat([ height width depth],[7*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coef,scales);
            
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
            dimExpctd = [ height width depth ];
            
            % Coefs
            upsample3_ = @(x,u,s) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,...
                u,s),1),...
                u,s),1),...
                u,s),1);
            haa2 = testCase.kernels.haa;
            hha2 = testCase.kernels.hha;
            hva2 = testCase.kernels.hva;
            hda2 = testCase.kernels.hda;
            had2 = testCase.kernels.had;
            hhd2 = testCase.kernels.hhd;
            hvd2 = testCase.kernels.hvd;
            hdd2 = testCase.kernels.hdd;              
            haa1 = imfilter(upsample3_(haa2,2,1),haa2);
            hha1 = imfilter(upsample3_(hha2,2,1),haa2);
            hva1 = imfilter(upsample3_(hva2,2,1),haa2);
            hda1 = imfilter(upsample3_(hda2,2,1),haa2);
            had1 = imfilter(upsample3_(had2,2,1),haa2);
            hhd1 = imfilter(upsample3_(hhd2,2,1),haa2);
            hvd1 = imfilter(upsample3_(hvd2,2,1),haa2);
            hdd1 = imfilter(upsample3_(hdd2,2,1),haa2);            
            yaa1 = imfilter(imgExpctd,haa1,'corr','circular');
            yha1 = imfilter(imgExpctd,hha1,'corr','circular');
            yva1 = imfilter(imgExpctd,hva1,'corr','circular');
            yda1 = imfilter(imgExpctd,hda1,'corr','circular');
            yad1 = imfilter(imgExpctd,had1,'corr','circular');
            yhd1 = imfilter(imgExpctd,hhd1,'corr','circular');
            yvd1 = imfilter(imgExpctd,hvd1,'corr','circular');
            ydd1 = imfilter(imgExpctd,hdd1,'corr','circular');            
            yha2 = imfilter(imgExpctd,hha2,'corr','circular');
            yva2 = imfilter(imgExpctd,hva2,'corr','circular');
            yda2 = imfilter(imgExpctd,hda2,'corr','circular');
            yad2 = imfilter(imgExpctd,had2,'corr','circular');
            yhd2 = imfilter(imgExpctd,hhd2,'corr','circular');
            yvd2 = imfilter(imgExpctd,hvd2,'corr','circular');
            ydd2 = imfilter(imgExpctd,hdd2,'corr','circular');                        
            coef = [ 
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
                ydd2(:) ].';
            scales = repmat([ height width depth ],[7*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coef,scales);
            
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
            dimExpctd = [ height width depth ];
            
            % Coefs
            upsample3_ = @(x,u,s) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,...
                u,s),1),...
                u,s),1),...
                u,s),1);
            haa2 = testCase.kernels.haa;
            hha2 = testCase.kernels.hha;
            hva2 = testCase.kernels.hva;
            hda2 = testCase.kernels.hda;
            had2 = testCase.kernels.had;
            hhd2 = testCase.kernels.hhd;
            hvd2 = testCase.kernels.hvd;
            hdd2 = testCase.kernels.hdd;              
            haa1 = imfilter(upsample3_(haa2,2,1),haa2);
            hha1 = imfilter(upsample3_(hha2,2,1),haa2);
            hva1 = imfilter(upsample3_(hva2,2,1),haa2);
            hda1 = imfilter(upsample3_(hda2,2,1),haa2);
            had1 = imfilter(upsample3_(had2,2,1),haa2);
            hhd1 = imfilter(upsample3_(hhd2,2,1),haa2);
            hvd1 = imfilter(upsample3_(hvd2,2,1),haa2);
            hdd1 = imfilter(upsample3_(hdd2,2,1),haa2);            
            yaa1 = imfilter(imgExpctd,haa1,'corr','circular');
            yha1 = imfilter(imgExpctd,hha1,'corr','circular');
            yva1 = imfilter(imgExpctd,hva1,'corr','circular');
            yda1 = imfilter(imgExpctd,hda1,'corr','circular');
            yad1 = imfilter(imgExpctd,had1,'corr','circular');
            yhd1 = imfilter(imgExpctd,hhd1,'corr','circular');
            yvd1 = imfilter(imgExpctd,hvd1,'corr','circular');
            ydd1 = imfilter(imgExpctd,hdd1,'corr','circular');            
            yha2 = imfilter(imgExpctd,hha2,'corr','circular');
            yva2 = imfilter(imgExpctd,hva2,'corr','circular');
            yda2 = imfilter(imgExpctd,hda2,'corr','circular');
            yad2 = imfilter(imgExpctd,had2,'corr','circular');
            yhd2 = imfilter(imgExpctd,hhd2,'corr','circular');
            yvd2 = imfilter(imgExpctd,hvd2,'corr','circular');
            ydd2 = imfilter(imgExpctd,hdd2,'corr','circular');                        
            coef = [ 
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
                ydd2(:) ].';
            scales = repmat([ height width depth ],[7*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coef,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        % Test for default construction
        function testLevel3ReconstructionSize16x16x16(testCase)
            
            nLevels = 3;
            height = 16;
            width = 16;
            depth = 16;
            
            % Expected values
            imgExpctd = rand(height,width,depth);
            dimExpctd = [ height width depth ];
            
            % Coefs
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
            yaa1 = imfilter(imgExpctd,haa1,'corr','circular');
            yad1 = imfilter(imgExpctd,had1,'corr','circular');
            yva1 = imfilter(imgExpctd,hva1,'corr','circular');
            yvd1 = imfilter(imgExpctd,hvd1,'corr','circular');
            yha1 = imfilter(imgExpctd,hha1,'corr','circular');
            yhd1 = imfilter(imgExpctd,hhd1,'corr','circular');
            yda1 = imfilter(imgExpctd,hda1,'corr','circular');
            ydd1 = imfilter(imgExpctd,hdd1,'corr','circular');            
            yad2 = imfilter(imgExpctd,had2,'corr','circular');
            yva2 = imfilter(imgExpctd,hva2,'corr','circular');
            yvd2 = imfilter(imgExpctd,hvd2,'corr','circular');
            yha2 = imfilter(imgExpctd,hha2,'corr','circular');
            yhd2 = imfilter(imgExpctd,hhd2,'corr','circular');
            yda2 = imfilter(imgExpctd,hda2,'corr','circular');
            ydd2 = imfilter(imgExpctd,hdd2,'corr','circular');
            yad3 = imfilter(imgExpctd,had3,'corr','circular');
            yva3 = imfilter(imgExpctd,hva3,'corr','circular');
            yvd3 = imfilter(imgExpctd,hvd3,'corr','circular');
            yha3 = imfilter(imgExpctd,hha3,'corr','circular');
            yhd3 = imfilter(imgExpctd,hhd3,'corr','circular');
            yda3 = imfilter(imgExpctd,hda3,'corr','circular');
            ydd3 = imfilter(imgExpctd,hdd3,'corr','circular');            
            coefs = [
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
            scales = repmat([ height width depth ],[ 7*nLevels+1, 1 ]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

        % Test for default construction
        function testLevel4Size32x32x32(testCase)
            
            nLevels = 4;
            height = 32;
            width = 32;
            depth = 32;
            caa1 = rand(height,width,depth);
            cha1 = rand(height,width,depth);
            cva1 = rand(height,width,depth);
            cda1 = rand(height,width,depth);
            cad1 = rand(height,width,depth);
            chd1 = rand(height,width,depth);
            cvd1 = rand(height,width,depth);
            cdd1 = rand(height,width,depth);
            cha2 = rand(height,width,depth);
            cva2 = rand(height,width,depth);
            cda2 = rand(height,width,depth);
            cad2 = rand(height,width,depth);
            chd2 = rand(height,width,depth);
            cvd2 = rand(height,width,depth);
            cdd2 = rand(height,width,depth);
            cha3 = rand(height,width,depth);
            cva3 = rand(height,width,depth);
            cda3 = rand(height,width,depth);
            cad3 = rand(height,width,depth);
            chd3 = rand(height,width,depth);
            cvd3 = rand(height,width,depth);
            cdd3 = rand(height,width,depth);
            cha4 = rand(height,width,depth);
            cva4 = rand(height,width,depth);
            cda4 = rand(height,width,depth);
            cad4 = rand(height,width,depth);
            chd4 = rand(height,width,depth);
            cvd4 = rand(height,width,depth);
            cdd4 = rand(height,width,depth);            
            subCoefs = [
                caa1(:)
                cha1(:)
                cva1(:)
                cda1(:)
                cad1(:)
                chd1(:)
                cvd1(:)
                cdd1(:)
                cha2(:)
                cva2(:)
                cda2(:)
                cad2(:)
                chd2(:)
                cvd2(:)
                cdd2(:)
                cha3(:)
                cva3(:)
                cda3(:)
                cad3(:)
                chd3(:)
                cvd3(:)
                cdd3(:) 
                cha4(:)
                cva4(:)
                cda4(:)
                cad4(:)
                chd4(:)
                cvd4(:)
                cdd4(:) ].';
            
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
            yha4 = circshift(imfilter(cha4,hha4,'conv','circular'),[1 1 1]);
            yva4 = circshift(imfilter(cva4,hva4,'conv','circular'),[1 1 1]);
            yda4 = circshift(imfilter(cda4,hda4,'conv','circular'),[1 1 1]);
            yad4 = circshift(imfilter(cad4,had4,'conv','circular'),[1 1 1]);
            yhd4 = circshift(imfilter(chd4,hhd4,'conv','circular'),[1 1 1]);
            yvd4 = circshift(imfilter(cvd4,hvd4,'conv','circular'),[1 1 1]);
            ydd4 = circshift(imfilter(cdd4,hdd4,'conv','circular'),[1 1 1]);
            imgExpctd = yaa1 + yha1 + yva1 + yda1 + yad1 + yhd1 + yvd1 + ydd1 ...
                + yha2 + yva2 + yda2 + yad2 + yhd2 + yvd2 + ydd2 ...
                + yha3 + yva3 + yda3 + yad3 + yhd3 + yvd3 + ydd3 ...
                + yha4 + yva4 + yda4 + yad4 + yhd4 + yvd4 + ydd4;
            dimExpctd = [ height width depth ];
            scalesExpctd = repmat([ height width depth ],[ 7*nLevels+1, 1 ]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test for default construction
        function testLevel4ReconstructionSize32x32x32(testCase,useparallel)
            
            nLevels = 4;
            height = 32;
            width = 32;
            depth = 32;
            
            % Expected values
            imgExpctd = rand(height,width,depth);
            dimExpctd = [ height width depth];
            
            % Coefs
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
            yaa1 = imfilter(imgExpctd,haa1,'corr','circular');
            yad1 = imfilter(imgExpctd,had1,'corr','circular');
            yva1 = imfilter(imgExpctd,hva1,'corr','circular');
            yvd1 = imfilter(imgExpctd,hvd1,'corr','circular');
            yha1 = imfilter(imgExpctd,hha1,'corr','circular');
            yhd1 = imfilter(imgExpctd,hhd1,'corr','circular');
            yda1 = imfilter(imgExpctd,hda1,'corr','circular');
            ydd1 = imfilter(imgExpctd,hdd1,'corr','circular');            
            yad2 = imfilter(imgExpctd,had2,'corr','circular');
            yva2 = imfilter(imgExpctd,hva2,'corr','circular');
            yvd2 = imfilter(imgExpctd,hvd2,'corr','circular');
            yha2 = imfilter(imgExpctd,hha2,'corr','circular');
            yhd2 = imfilter(imgExpctd,hhd2,'corr','circular');
            yda2 = imfilter(imgExpctd,hda2,'corr','circular');
            ydd2 = imfilter(imgExpctd,hdd2,'corr','circular');
            yad3 = imfilter(imgExpctd,had3,'corr','circular');
            yva3 = imfilter(imgExpctd,hva3,'corr','circular');
            yvd3 = imfilter(imgExpctd,hvd3,'corr','circular');
            yha3 = imfilter(imgExpctd,hha3,'corr','circular');
            yhd3 = imfilter(imgExpctd,hhd3,'corr','circular');
            yda3 = imfilter(imgExpctd,hda3,'corr','circular');
            ydd3 = imfilter(imgExpctd,hdd3,'corr','circular');            
            yad4 = imfilter(imgExpctd,had4,'corr','circular');
            yva4 = imfilter(imgExpctd,hva4,'corr','circular');
            yvd4 = imfilter(imgExpctd,hvd4,'corr','circular');
            yha4 = imfilter(imgExpctd,hha4,'corr','circular');
            yhd4 = imfilter(imgExpctd,hhd4,'corr','circular');
            yda4 = imfilter(imgExpctd,hda4,'corr','circular');
            ydd4 = imfilter(imgExpctd,hdd4,'corr','circular');            
            coefs = [
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
            scales = repmat([ height width depth ],[7*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis3dSystem(...
                'UseParallel',useparallel);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

    end
    
end

