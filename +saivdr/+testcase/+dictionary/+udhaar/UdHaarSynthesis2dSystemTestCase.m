classdef UdHaarSynthesis2dSystemTestCase < matlab.unittest.TestCase
    %UDHAARSYNTHESIZERTESTCASE Test case for UdHaarSynthesis2dSystem
    %
    % SVN identifier:
    % $Id: UdHaarSynthesis2dSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
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
   
    properties
        synthesizer
    end
        
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.synthesizer);
        end
    end
    
    methods (Test)
      
        % Test for default construction
        function testLevel1Size16x16(testCase)
            
            nLevels = 1;
            height = 16;
            width = 16;
            ca  = rand(height,width);
            ch  = rand(height,width);
            cv  = rand(height,width);
            cd  = rand(height,width);
            subCoefs = [ ca(:).' ch(:).' cv(:).' cd(:).' ];
            
            % Expected values
            ha = [ 1 1 ; 1 1 ]/4;
            hh = [ 1 -1 ; 1 -1 ]/4;
            hv = [ 1 1 ; -1 -1 ]/4;
            hd = [ 1 -1 ; -1 1 ]/4;            
            ya = circshift(imfilter(ca,ha,'conv','circular'),[1 1]);
            yh = circshift(imfilter(ch,hh,'conv','circular'),[1 1]);
            yv = circshift(imfilter(cv,hv,'conv','circular'),[1 1]);
            yd = circshift(imfilter(cd,hd,'conv','circular'),[1 1]);
            imgExpctd = ya + yh + yv + yd;
            dimExpctd = [ height width ];
            scalesExpctd = repmat([ height width ],[3*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
    
        % Test for default construction
        function testLevel1Size16x32(testCase)
            
            nLevels = 1;
            height = 16;
            width = 32;
            ca  = rand(height,width);
            ch  = rand(height,width);
            cv  = rand(height,width);
            cd  = rand(height,width);
            subCoefs = [ ca(:).' ch(:).' cv(:).' cd(:).' ];
            
            % Expected values
            ha = [ 1 1 ; 1 1 ]/4;
            hh = [ 1 -1 ; 1 -1 ]/4;
            hv = [ 1 1 ; -1 -1 ]/4;
            hd = [ 1 -1 ; -1 1 ]/4;
            ya = circshift(imfilter(ca,ha,'conv','circular'),[1 1]);
            yh = circshift(imfilter(ch,hh,'conv','circular'),[1 1]);
            yv = circshift(imfilter(cv,hv,'conv','circular'),[1 1]);
            yd = circshift(imfilter(cd,hd,'conv','circular'),[1 1]);
            imgExpctd = ya + yh + yv + yd;
            dimExpctd = [ height width ];
            scalesExpctd = repmat([ height width ],[3*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
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
            ca1  = rand(height,width);
            ch1  = rand(height,width);
            cv1  = rand(height,width);
            cd1  = rand(height,width);
            ch2  = rand(height,width);
            cv2  = rand(height,width);
            cd2  = rand(height,width);            
            subCoefs = [ ca1(:).' ch1(:).' cv1(:).' cd1(:).' ...
                ch2(:).' cv2(:).' cd2(:).' ];
            
            % Expected values
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hh2 = [ 1 -1 ; 1 -1 ]/4;
            hv2 = [ 1 1 ; -1 -1 ]/4;
            hd2 = [ 1 -1 ; -1 1 ]/4;                        
            ya1 = circshift(imfilter(ca1,ha1,'conv','circular'),[1 1]);
            yh1 = circshift(imfilter(ch1,hh1,'conv','circular'),[1 1]);
            yv1 = circshift(imfilter(cv1,hv1,'conv','circular'),[1 1]);
            yd1 = circshift(imfilter(cd1,hd1,'conv','circular'),[1 1]);
            yh2 = circshift(imfilter(ch2,hh2,'conv','circular'),[1 1]);
            yv2 = circshift(imfilter(cv2,hv2,'conv','circular'),[1 1]);
            yd2 = circshift(imfilter(cd2,hd2,'conv','circular'),[1 1]); 
            imgExpctd = ya1 + yh1 + yv1 + yd1 + yh2 + yv2 + yd2;
            dimExpctd = [ height width ];
            scalesExpctd = repmat([ height width ],[3*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test for default construction
        function testLevel2Size16x32(testCase)
            
            nLevels = 2;
            height = 16;
            width = 16;
            ca1  = rand(height,width);
            ch1  = rand(height,width);
            cv1  = rand(height,width);
            cd1  = rand(height,width);
            ch2  = rand(height,width);
            cv2  = rand(height,width);
            cd2  = rand(height,width);            
            subCoefs = [ ca1(:).' ch1(:).' cv1(:).' cd1(:).' ...
                ch2(:).' cv2(:).' cd2(:).' ];
            
            % Expected values
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hh2 = [ 1 -1 ; 1 -1 ]/4;
            hv2 = [ 1 1 ; -1 -1 ]/4;
            hd2 = [ 1 -1 ; -1 1 ]/4;                        
            ya1 = circshift(imfilter(ca1,ha1,'conv','circular'),[1 1]);
            yh1 = circshift(imfilter(ch1,hh1,'conv','circular'),[1 1]);
            yv1 = circshift(imfilter(cv1,hv1,'conv','circular'),[1 1]);
            yd1 = circshift(imfilter(cd1,hd1,'conv','circular'),[1 1]);
            yh2 = circshift(imfilter(ch2,hh2,'conv','circular'),[1 1]);
            yv2 = circshift(imfilter(cv2,hv2,'conv','circular'),[1 1]);
            yd2 = circshift(imfilter(cd2,hd2,'conv','circular'),[1 1]); 
            imgExpctd = ya1 + yh1 + yv1 + yd1 + yh2 + yv2 + yd2;
            dimExpctd = [ height width ];
            scalesExpctd = repmat([ height width ],[3*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test for default construction
        function testLevel3Size16x16(testCase)
            
            nLevels = 3;
            height = 16;
            width = 16;
            ca1  = rand(height,width);
            ch1  = rand(height,width);
            cv1  = rand(height,width);
            cd1  = rand(height,width);
            ch2  = rand(height,width);
            cv2  = rand(height,width);
            cd2  = rand(height,width);            
            ch3  = rand(height,width);
            cv3  = rand(height,width);
            cd3  = rand(height,width);                        
            subCoefs = [ ca1(:).' ch1(:).' cv1(:).' cd1(:).' ...
                ch2(:).' cv2(:).' cd2(:).' ...
                ch3(:).' cv3(:).' cd3(:).' ];
            
            % Expected values
            ha3 = [ 1 1 ; 1 1 ]/4;
            hh3 = [ 1 -1 ; 1 -1 ]/4;
            hv3 = [ 1 1 ; -1 -1 ]/4;
            hd3 = [ 1 -1 ; -1 1 ]/4;                                    
            ha2 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',ha3);
            hh2 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',ha3);
            hv2 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',ha3);
            hd2 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',ha3);
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,4,2).',4,2).',ha2);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,4,2).',4,2).',ha2);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,4,2).',4,2).',ha2);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,4,2).',4,2).',ha2);
            ya1 = circshift(imfilter(ca1,ha1,'conv','circular'),[1 1]);
            yh1 = circshift(imfilter(ch1,hh1,'conv','circular'),[1 1]);
            yv1 = circshift(imfilter(cv1,hv1,'conv','circular'),[1 1]);
            yd1 = circshift(imfilter(cd1,hd1,'conv','circular'),[1 1]);
            yh2 = circshift(imfilter(ch2,hh2,'conv','circular'),[1 1]);
            yv2 = circshift(imfilter(cv2,hv2,'conv','circular'),[1 1]);
            yd2 = circshift(imfilter(cd2,hd2,'conv','circular'),[1 1]);            
            yh3 = circshift(imfilter(ch3,hh3,'conv','circular'),[1 1]);
            yv3 = circshift(imfilter(cv3,hv3,'conv','circular'),[1 1]);
            yd3 = circshift(imfilter(cd3,hd3,'conv','circular'),[1 1]);           
            imgExpctd = ya1 + yh1 + yv1 + yd1 + yh2 + yv2 + yd2 + yh3 + yv3 + yd3;
            dimExpctd = [ height width ];
            scalesExpctd = repmat([ height width ],[3*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test for default construction
        function testLevel3Size16x32(testCase)
            
            nLevels = 3;
            height = 16;
            width = 32;
            ca1  = rand(height,width);
            ch1  = rand(height,width);
            cv1  = rand(height,width);
            cd1  = rand(height,width);
            ch2  = rand(height,width);
            cv2  = rand(height,width);
            cd2  = rand(height,width);            
            ch3  = rand(height,width);
            cv3  = rand(height,width);
            cd3  = rand(height,width);                        
            subCoefs = [ ca1(:).' ch1(:).' cv1(:).' cd1(:).' ...
                ch2(:).' cv2(:).' cd2(:).' ...
                ch3(:).' cv3(:).' cd3(:).' ];
            
            % Expected values
            ha3 = [ 1 1 ; 1 1 ]/4;
            hh3 = [ 1 -1 ; 1 -1 ]/4;
            hv3 = [ 1 1 ; -1 -1 ]/4;
            hd3 = [ 1 -1 ; -1 1 ]/4;                                    
            ha2 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',ha3);
            hh2 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',ha3);
            hv2 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',ha3);
            hd2 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',ha3);
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,4,2).',4,2).',ha2);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,4,2).',4,2).',ha2);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,4,2).',4,2).',ha2);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,4,2).',4,2).',ha2);
            ya1 = circshift(imfilter(ca1,ha1,'conv','circular'),[1 1]);
            yh1 = circshift(imfilter(ch1,hh1,'conv','circular'),[1 1]);
            yv1 = circshift(imfilter(cv1,hv1,'conv','circular'),[1 1]);
            yd1 = circshift(imfilter(cd1,hd1,'conv','circular'),[1 1]);
            yh2 = circshift(imfilter(ch2,hh2,'conv','circular'),[1 1]);
            yv2 = circshift(imfilter(cv2,hv2,'conv','circular'),[1 1]);
            yd2 = circshift(imfilter(cd2,hd2,'conv','circular'),[1 1]);            
            yh3 = circshift(imfilter(ch3,hh3,'conv','circular'),[1 1]);
            yv3 = circshift(imfilter(cv3,hv3,'conv','circular'),[1 1]);
            yd3 = circshift(imfilter(cd3,hd3,'conv','circular'),[1 1]);             
            imgExpctd = ya1 + yh1 + yv1 + yd1 + yh2 + yv2 + yd2 + yh3 + yv3 + yd3;
            dimExpctd = [ height width ];
            scalesExpctd = repmat([ height width ],[3*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        % Test for default construction
        function testLevel1ReconstructionSize16x16(testCase)
            
            nLevels = 1;
            height = 16;
            width = 16;
            
            % Expected values            
            imgExpctd = rand(height,width);
            dimExpctd = [ height width ];
            
            % Coefs
            ha = [ 1 1 ; 1 1 ]/4;
            hh = [ 1 -1 ; 1 -1 ]/4;
            hv = [ 1 1 ; -1 -1 ]/4;
            hd = [ 1 -1 ; -1 1 ]/4;            
            ya = imfilter(imgExpctd,ha,'corr','circular');
            yh = imfilter(imgExpctd,hh,'corr','circular');
            yv = imfilter(imgExpctd,hv,'corr','circular');
            yd = imfilter(imgExpctd,hd,'corr','circular');
            coef = [ ya(:).' yh(:).' yv(:).' yd(:).' ];
            scales = repmat([ height width ],[3*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coef,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test for default construction
        function testLevel1ReconstructionSize16x32(testCase)
            
            nLevels = 1;
            height = 16;
            width = 32;
            
            % Expected values
            imgExpctd = rand(height,width);
            dimExpctd = [ height width ];
            
            % Coefs            
            ha = [ 1 1 ; 1 1 ]/4;
            hh = [ 1 -1 ; 1 -1 ]/4;
            hv = [ 1 1 ; -1 -1 ]/4;
            hd = [ 1 -1 ; -1 1 ]/4;            
            ya = imfilter(imgExpctd,ha,'corr','circular');
            yh = imfilter(imgExpctd,hh,'corr','circular');
            yv = imfilter(imgExpctd,hv,'corr','circular');
            yd = imfilter(imgExpctd,hd,'corr','circular');
            coef = [ ya(:).' yh(:).' yv(:).' yd(:).' ];
            scales = repmat([ height width ],[3*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coef,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        % Test for default construction
        function testLevel2ReconstructionSize16x16(testCase)
            
            nLevels = 2;
            height = 16;
            width = 16;
            
            % Expected values
            imgExpctd = rand(height,width);
            dimExpctd = [ height width ];
            
            % Coefs
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hh2 = [ 1 -1 ; 1 -1 ]/4;
            hv2 = [ 1 1 ; -1 -1 ]/4;
            hd2 = [ 1 -1 ; -1 1 ]/4;                        
            ya1 = imfilter(imgExpctd,ha1,'corr','circular');
            yh1 = imfilter(imgExpctd,hh1,'corr','circular');
            yv1 = imfilter(imgExpctd,hv1,'corr','circular');
            yd1 = imfilter(imgExpctd,hd1,'corr','circular');
            yh2 = imfilter(imgExpctd,hh2,'corr','circular');
            yv2 = imfilter(imgExpctd,hv2,'corr','circular');
            yd2 = imfilter(imgExpctd,hd2,'corr','circular');            
            coef = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ];
            scales = repmat([ height width ],[3*nLevels+1, 1]);            

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coef,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test for default construction
        function testLevel2ReconstructionSize16x32(testCase)
            
            nLevels = 2;
            height = 16;
            width = 32;
            
            % Expected values
            imgExpctd = rand(height,width);
            dimExpctd = [ height width ];
            
            % Coefs
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hh2 = [ 1 -1 ; 1 -1 ]/4;
            hv2 = [ 1 1 ; -1 -1 ]/4;
            hd2 = [ 1 -1 ; -1 1 ]/4;                        
            ya1 = imfilter(imgExpctd,ha1,'corr','circular');
            yh1 = imfilter(imgExpctd,hh1,'corr','circular');
            yv1 = imfilter(imgExpctd,hv1,'corr','circular');
            yd1 = imfilter(imgExpctd,hd1,'corr','circular');
            yh2 = imfilter(imgExpctd,hh2,'corr','circular');
            yv2 = imfilter(imgExpctd,hv2,'corr','circular');
            yd2 = imfilter(imgExpctd,hd2,'corr','circular');            
            coef = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ];
            scales = repmat([ height width ],[3*nLevels+1, 1 ]);            

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coef,scales);
            
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
            
            % Expected values
            imgExpctd = rand(height,width);
            dimExpctd = [ height width ];
            
            % Coefs
            ha3 = [ 1 1 ; 1 1 ]/4;
            hh3 = [ 1 -1 ; 1 -1 ]/4;
            hv3 = [ 1 1 ; -1 -1 ]/4;
            hd3 = [ 1 -1 ; -1 1 ]/4;                                    
            ha2 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',ha3);
            hh2 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',ha3);
            hv2 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',ha3);
            hd2 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',ha3);
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,4,2).',4,2).',ha2);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,4,2).',4,2).',ha2);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,4,2).',4,2).',ha2);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,4,2).',4,2).',ha2);
            ya1 = imfilter(imgExpctd,ha1,'corr','circular');
            yh1 = imfilter(imgExpctd,hh1,'corr','circular');
            yv1 = imfilter(imgExpctd,hv1,'corr','circular');
            yd1 = imfilter(imgExpctd,hd1,'corr','circular');
            yh2 = imfilter(imgExpctd,hh2,'corr','circular');
            yv2 = imfilter(imgExpctd,hv2,'corr','circular');
            yd2 = imfilter(imgExpctd,hd2,'corr','circular');            
            yh3 = imfilter(imgExpctd,hh3,'corr','circular');
            yv3 = imfilter(imgExpctd,hv3,'corr','circular');
            yd3 = imfilter(imgExpctd,hd3,'corr','circular');                        
            coefs = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ...
                yh3(:).' yv3(:).' yd3(:).' ];
            scales = repmat([ height width ],[ 3*nLevels+1, 1 ]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));

        end
        
        % Test for default construction
        function testLevel4Size32x32(testCase)
            
            nLevels = 4;
            height = 32;
            width = 32;
            ca1  = rand(height,width);
            ch1  = rand(height,width);
            cv1  = rand(height,width);
            cd1  = rand(height,width);
            ch2  = rand(height,width);
            cv2  = rand(height,width);
            cd2  = rand(height,width);            
            ch3  = rand(height,width);
            cv3  = rand(height,width);
            cd3  = rand(height,width);                        
            ch4  = rand(height,width);
            cv4  = rand(height,width);
            cd4  = rand(height,width);                                    
            subCoefs = [ ca1(:).' ch1(:).' cv1(:).' cd1(:).' ...
                ch2(:).' cv2(:).' cd2(:).' ...
                ch3(:).' cv3(:).' cd3(:).' ...
                ch4(:).' cv4(:).' cd4(:).' ];
            
            % Expected values
            ha4 = [ 1 1 ; 1 1 ]/4;
            hh4 = [ 1 -1 ; 1 -1 ]/4;
            hv4 = [ 1 1 ; -1 -1 ]/4;
            hd4 = [ 1 -1 ; -1 1 ]/4;                                    
            ha3 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',ha4);
            hh3 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',ha4);
            hv3 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',ha4);
            hd3 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',ha4);
            ha2 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,4,2).',4,2).',ha3);
            hh2 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,4,2).',4,2).',ha3);
            hv2 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,4,2).',4,2).',ha3);
            hd2 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,4,2).',4,2).',ha3);
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,8,4).',8,4).',ha2);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,8,4).',8,4).',ha2);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,8,4).',8,4).',ha2);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,8,4).',8,4).',ha2);            
            ya1 = circshift(imfilter(ca1,ha1,'conv','circular'),[1 1]);
            yh1 = circshift(imfilter(ch1,hh1,'conv','circular'),[1 1]);
            yv1 = circshift(imfilter(cv1,hv1,'conv','circular'),[1 1]);
            yd1 = circshift(imfilter(cd1,hd1,'conv','circular'),[1 1]);
            yh2 = circshift(imfilter(ch2,hh2,'conv','circular'),[1 1]);
            yv2 = circshift(imfilter(cv2,hv2,'conv','circular'),[1 1]);
            yd2 = circshift(imfilter(cd2,hd2,'conv','circular'),[1 1]);
            yh3 = circshift(imfilter(ch3,hh3,'conv','circular'),[1 1]);
            yv3 = circshift(imfilter(cv3,hv3,'conv','circular'),[1 1]);
            yd3 = circshift(imfilter(cd3,hd3,'conv','circular'),[1 1]);            
            yh4 = circshift(imfilter(ch4,hh4,'conv','circular'),[1 1]);
            yv4 = circshift(imfilter(cv4,hv4,'conv','circular'),[1 1]);
            yd4 = circshift(imfilter(cd4,hd4,'conv','circular'),[1 1]);           
            imgExpctd = ya1 + yh1 + yv1 + yd1 + yh2 + yv2 + yd2 + yh3 + yv3 + yd3 + yh4 + yv4 + yd4;
            dimExpctd = [ height width ];
            scalesExpctd = repmat([ height width ],[ 3*nLevels+1, 1 ]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        % Test for default construction
        function testLevel4Size32x64(testCase)
            
            nLevels = 4;
            height = 32;
            width = 64;
            ca1  = rand(height,width);
            ch1  = rand(height,width);
            cv1  = rand(height,width);
            cd1  = rand(height,width);
            ch2  = rand(height,width);
            cv2  = rand(height,width);
            cd2  = rand(height,width);            
            ch3  = rand(height,width);
            cv3  = rand(height,width);
            cd3  = rand(height,width);                        
            ch4  = rand(height,width);
            cv4  = rand(height,width);
            cd4  = rand(height,width);                                    
            subCoefs = [ ca1(:).' ch1(:).' cv1(:).' cd1(:).' ...
                ch2(:).' cv2(:).' cd2(:).' ...
                ch3(:).' cv3(:).' cd3(:).' ...
                ch4(:).' cv4(:).' cd4(:).' ];
            
            % Expected values
            ha4 = [ 1 1 ; 1 1 ]/4;
            hh4 = [ 1 -1 ; 1 -1 ]/4;
            hv4 = [ 1 1 ; -1 -1 ]/4;
            hd4 = [ 1 -1 ; -1 1 ]/4;                                    
            ha3 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',ha4);
            hh3 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',ha4);
            hv3 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',ha4);
            hd3 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',ha4);
            ha2 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,4,2).',4,2).',ha3);
            hh2 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,4,2).',4,2).',ha3);
            hv2 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,4,2).',4,2).',ha3);
            hd2 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,4,2).',4,2).',ha3);
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,8,4).',8,4).',ha2);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,8,4).',8,4).',ha2);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,8,4).',8,4).',ha2);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,8,4).',8,4).',ha2);            
            ya1 = circshift(imfilter(ca1,ha1,'conv','circular'),[1 1]);
            yh1 = circshift(imfilter(ch1,hh1,'conv','circular'),[1 1]);
            yv1 = circshift(imfilter(cv1,hv1,'conv','circular'),[1 1]);
            yd1 = circshift(imfilter(cd1,hd1,'conv','circular'),[1 1]);
            yh2 = circshift(imfilter(ch2,hh2,'conv','circular'),[1 1]);
            yv2 = circshift(imfilter(cv2,hv2,'conv','circular'),[1 1]);
            yd2 = circshift(imfilter(cd2,hd2,'conv','circular'),[1 1]);
            yh3 = circshift(imfilter(ch3,hh3,'conv','circular'),[1 1]);
            yv3 = circshift(imfilter(cv3,hv3,'conv','circular'),[1 1]);
            yd3 = circshift(imfilter(cd3,hd3,'conv','circular'),[1 1]);            
            yh4 = circshift(imfilter(ch4,hh4,'conv','circular'),[1 1]);
            yv4 = circshift(imfilter(cv4,hv4,'conv','circular'),[1 1]);
            yd4 = circshift(imfilter(cd4,hd4,'conv','circular'),[1 1]);           
            imgExpctd = ya1 + yh1 + yv1 + yd1 + yh2 + yv2 + yd2 + yh3 + yv3 + yd3 + yh4 + yv4 + yd4;
            dimExpctd = [ height width ];
            scalesExpctd = repmat([ height width ],[ 3*nLevels+1, 1 ]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        % Test for default construction
        function testLevel5Size64x64(testCase)
            
            nLevels = 5;
            height = 64;
            width = 64;
            ca1  = rand(height,width);
            ch1  = rand(height,width);
            cv1  = rand(height,width);
            cd1  = rand(height,width);
            ch2  = rand(height,width);
            cv2  = rand(height,width);
            cd2  = rand(height,width);            
            ch3  = rand(height,width);
            cv3  = rand(height,width);
            cd3  = rand(height,width);                        
            ch4  = rand(height,width);
            cv4  = rand(height,width);
            cd4  = rand(height,width);                                    
            ch5  = rand(height,width);
            cv5  = rand(height,width);
            cd5  = rand(height,width);                                    
            subCoefs = [ ca1(:).' ch1(:).' cv1(:).' cd1(:).' ...
                ch2(:).' cv2(:).' cd2(:).' ...
                ch3(:).' cv3(:).' cd3(:).' ...
                ch4(:).' cv4(:).' cd4(:).' ....
                ch5(:).' cv5(:).' cd5(:).' ];            
            
            % Expected values
            ha5 = [ 1 1 ; 1 1 ]/4;
            hh5 = [ 1 -1 ; 1 -1 ]/4;
            hv5 = [ 1 1 ; -1 -1 ]/4;
            hd5 = [ 1 -1 ; -1 1 ]/4;                                    
            ha4 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',ha5);
            hh4 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',ha5);
            hv4 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',ha5);
            hd4 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',ha5);
            ha3 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,4,2).',4,2).',ha4);
            hh3 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,4,2).',4,2).',ha4);
            hv3 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,4,2).',4,2).',ha4);
            hd3 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,4,2).',4,2).',ha4);
            ha2 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,8,4).',8,4).',ha3);
            hh2 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,8,4).',8,4).',ha3);
            hv2 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,8,4).',8,4).',ha3);
            hd2 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,8,4).',8,4).',ha3);            
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,16,8).',16,8).',ha2);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,16,8).',16,8).',ha2);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,16,8).',16,8).',ha2);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,16,8).',16,8).',ha2);            
            ya1 = circshift(imfilter(ca1,ha1,'conv','circular'),[1 1]);
            yh1 = circshift(imfilter(ch1,hh1,'conv','circular'),[1 1]);
            yv1 = circshift(imfilter(cv1,hv1,'conv','circular'),[1 1]);
            yd1 = circshift(imfilter(cd1,hd1,'conv','circular'),[1 1]);
            yh2 = circshift(imfilter(ch2,hh2,'conv','circular'),[1 1]);
            yv2 = circshift(imfilter(cv2,hv2,'conv','circular'),[1 1]);
            yd2 = circshift(imfilter(cd2,hd2,'conv','circular'),[1 1]);
            yh3 = circshift(imfilter(ch3,hh3,'conv','circular'),[1 1]);
            yv3 = circshift(imfilter(cv3,hv3,'conv','circular'),[1 1]);
            yd3 = circshift(imfilter(cd3,hd3,'conv','circular'),[1 1]);            
            yh4 = circshift(imfilter(ch4,hh4,'conv','circular'),[1 1]);
            yv4 = circshift(imfilter(cv4,hv4,'conv','circular'),[1 1]);
            yd4 = circshift(imfilter(cd4,hd4,'conv','circular'),[1 1]);           
            yh5 = circshift(imfilter(ch5,hh5,'conv','circular'),[1 1]);
            yv5 = circshift(imfilter(cv5,hv5,'conv','circular'),[1 1]);
            yd5 = circshift(imfilter(cd5,hd5,'conv','circular'),[1 1]);                       
            imgExpctd = ya1 + yh1 + yv1 + yd1 + yh2 + yv2 + yd2 + yh3 + yv3 + yd3 + yh4 + yv4 + yd4 + yh5 + yv5 + yd5;
            dimExpctd = [ height width ];
            scalesExpctd = repmat([ height width ], [ 3*nLevels+1, 1 ]) ;
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        % Test for default construction
        function testLevel3ReconstructionSize16x32(testCase)
            
            nLevels = 3;
            height = 16;
            width = 32;
            
            % Expected values
            imgExpctd = rand(height,width);
            dimExpctd = [ height width ];
            
            % Coefs
            ha3 = [ 1 1 ; 1 1 ]/4;
            hh3 = [ 1 -1 ; 1 -1 ]/4;
            hv3 = [ 1 1 ; -1 -1 ]/4;
            hd3 = [ 1 -1 ; -1 1 ]/4;                                    
            ha2 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',ha3);
            hh2 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',ha3);
            hv2 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',ha3);
            hd2 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',ha3);
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,4,2).',4,2).',ha2);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,4,2).',4,2).',ha2);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,4,2).',4,2).',ha2);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,4,2).',4,2).',ha2);
            ya1 = imfilter(imgExpctd,ha1,'corr','circular');
            yh1 = imfilter(imgExpctd,hh1,'corr','circular');
            yv1 = imfilter(imgExpctd,hv1,'corr','circular');
            yd1 = imfilter(imgExpctd,hd1,'corr','circular');
            yh2 = imfilter(imgExpctd,hh2,'corr','circular');
            yv2 = imfilter(imgExpctd,hv2,'corr','circular');
            yd2 = imfilter(imgExpctd,hd2,'corr','circular');            
            yh3 = imfilter(imgExpctd,hh3,'corr','circular');
            yv3 = imfilter(imgExpctd,hv3,'corr','circular');
            yd3 = imfilter(imgExpctd,hd3,'corr','circular');                        
            coefs = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ...
                yh3(:).' yv3(:).' yd3(:).' ];
            scales = repmat([ height width ],[3*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));

        end
        
        % Test for default construction
        function testLevel4ReconstructionSize32x32(testCase)
            
            nLevels = 4;
            height = 32;
            width = 32;
            
            % Expected values
            imgExpctd = rand(height,width);
            dimExpctd = [ height width ];
            
            % Coefs
            ha4 = [ 1 1 ; 1 1 ]/4;
            hh4 = [ 1 -1 ; 1 -1 ]/4;
            hv4 = [ 1 1 ; -1 -1 ]/4;
            hd4 = [ 1 -1 ; -1 1 ]/4;                                    
            ha3 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',ha4);
            hh3 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',ha4);
            hv3 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',ha4);
            hd3 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',ha4);
            ha2 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,4,2).',4,2).',ha3);
            hh2 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,4,2).',4,2).',ha3);
            hv2 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,4,2).',4,2).',ha3);
            hd2 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,4,2).',4,2).',ha3);
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,8,4).',8,4).',ha2);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,8,4).',8,4).',ha2);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,8,4).',8,4).',ha2);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,8,4).',8,4).',ha2);            
            ya1 = imfilter(imgExpctd,ha1,'corr','circular');
            yh1 = imfilter(imgExpctd,hh1,'corr','circular');
            yv1 = imfilter(imgExpctd,hv1,'corr','circular');
            yd1 = imfilter(imgExpctd,hd1,'corr','circular');
            yh2 = imfilter(imgExpctd,hh2,'corr','circular');
            yv2 = imfilter(imgExpctd,hv2,'corr','circular');
            yd2 = imfilter(imgExpctd,hd2,'corr','circular');            
            yh3 = imfilter(imgExpctd,hh3,'corr','circular');
            yv3 = imfilter(imgExpctd,hv3,'corr','circular');
            yd3 = imfilter(imgExpctd,hd3,'corr','circular');            
            yh4 = imfilter(imgExpctd,hh4,'corr','circular');
            yv4 = imfilter(imgExpctd,hv4,'corr','circular');
            yd4 = imfilter(imgExpctd,hd4,'corr','circular');                        
            coefs = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ...
                yh3(:).' yv3(:).' yd3(:).' ...
                yh4(:).' yv4(:).' yd4(:).' ];
            scales = repmat([ height width ],[3*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));

        end
        
                
        % Test for default construction
        function testLevel5ReconstructionSize64x64(testCase)
            
            nLevels = 5;
            height = 64;
            width = 64;
            
            % Expected values
            imgExpctd = rand(height,width);
            dimExpctd = [ height width ];
            
            % Coefs
            ha5 = [ 1 1 ; 1 1 ]/4;
            hh5 = [ 1 -1 ; 1 -1 ]/4;
            hv5 = [ 1 1 ; -1 -1 ]/4;
            hd5 = [ 1 -1 ; -1 1 ]/4;                                    
            ha4 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',ha5);
            hh4 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',ha5);
            hv4 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',ha5);
            hd4 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',ha5);
            ha3 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,4,2).',4,2).',ha4);
            hh3 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,4,2).',4,2).',ha4);
            hv3 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,4,2).',4,2).',ha4);
            hd3 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,4,2).',4,2).',ha4);
            ha2 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,8,4).',8,4).',ha3);
            hh2 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,8,4).',8,4).',ha3);
            hv2 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,8,4).',8,4).',ha3);
            hd2 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,8,4).',8,4).',ha3);            
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,16,8).',16,8).',ha2);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,16,8).',16,8).',ha2);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,16,8).',16,8).',ha2);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,16,8).',16,8).',ha2);                        
            ya1 = imfilter(imgExpctd,ha1,'corr','circular');
            yh1 = imfilter(imgExpctd,hh1,'corr','circular');
            yv1 = imfilter(imgExpctd,hv1,'corr','circular');
            yd1 = imfilter(imgExpctd,hd1,'corr','circular');
            yh2 = imfilter(imgExpctd,hh2,'corr','circular');
            yv2 = imfilter(imgExpctd,hv2,'corr','circular');            
            yd2 = imfilter(imgExpctd,hd2,'corr','circular');
            yh3 = imfilter(imgExpctd,hh3,'corr','circular');
            yv3 = imfilter(imgExpctd,hv3,'corr','circular');
            yd3 = imfilter(imgExpctd,hd3,'corr','circular');            
            yh4 = imfilter(imgExpctd,hh4,'corr','circular');
            yv4 = imfilter(imgExpctd,hv4,'corr','circular');
            yd4 = imfilter(imgExpctd,hd4,'corr','circular');            
            yh5 = imfilter(imgExpctd,hh5,'corr','circular');
            yv5 = imfilter(imgExpctd,hv5,'corr','circular');
            yd5 = imfilter(imgExpctd,hd5,'corr','circular');                        
            coefs = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ...
                yh3(:).' yv3(:).' yd3(:).' ...
                yh4(:).' yv4(:).' yd4(:).' ...
                yh5(:).' yv5(:).' yd5(:).' ];
            scales = repmat([ height width ], [3*nLevels+1, 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.synthesizer = UdHaarSynthesis2dSystem();
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));

        end


    end
    
end

