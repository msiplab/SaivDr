classdef UdHaarAnalysis2dSystemTestCase < matlab.unittest.TestCase
    %UdHaarAnalysis2dSystemTESTCASE Test case for UdHaarAnalysis2dSystem
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
    
    properties
        analyzer
    end
    
    methods (TestMethodTeardown)
        function deteleObject(testCase)
            delete(testCase.analyzer);
        end
    end
    
    methods (Test)

        % Test for default construction
        function testLevel1Size16x16(testCase)
            
            nLevels = 1;
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            
            % Expected values
            ha = [ 1 1 ; 1 1 ]/4;
            hh = [ 1 -1 ; 1 -1 ]/4;
            hv = [ 1 1 ; -1 -1 ]/4;
            hd = [ 1 -1 ; -1 1 ]/4;            
            ya = imfilter(srcImg,ha,'corr','circular');
            yh = imfilter(srcImg,hh,'corr','circular');
            yv = imfilter(srcImg,hv,'corr','circular');
            yd = imfilter(srcImg,hd,'corr','circular');
            coefExpctd = [ ya(:).' yh(:).' yv(:).' yd(:).' ];
            scalesExpctd = repmat([ height width ],[4 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [ coefActual, scalesActual ] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifyEqual(scalesActual, scalesExpctd);
        end

        function testLevel1Size16x32(testCase)
            
            nLevels = 1;
            height = 16;
            width = 32;
            srcImg = rand(height,width);
            
            % Expected values
            ha = [ 1 1 ; 1 1 ]/4;
            hh = [ 1 -1 ; 1 -1 ]/4;
            hv = [ 1 1 ; -1 -1 ]/4;
            hd = [ 1 -1 ; -1 1 ]/4;            
            ya = imfilter(srcImg,ha,'corr','circular');
            yh = imfilter(srcImg,hh,'corr','circular');
            yv = imfilter(srcImg,hv,'corr','circular');
            yd = imfilter(srcImg,hd,'corr','circular');
            coefExpctd = [ ya(:).' yh(:).' yv(:).' yd(:).' ];
            scalesExpctd = repmat([ height, width ],[4 1]);
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [ coefActual, scalesActual ] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifyEqual(scalesActual, scalesExpctd);
        end

        % Test for default construction
        function testLevel2Size16x16(testCase)
            
            nLevels = 2;
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            
            % Expected values
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hh2 = [ 1 -1 ; 1 -1 ]/4;
            hv2 = [ 1 1 ; -1 -1 ]/4;
            hd2 = [ 1 -1 ; -1 1 ]/4;                        
            ya1 = imfilter(srcImg,ha1,'corr','circular');
            yh1 = imfilter(srcImg,hh1,'corr','circular');
            yv1 = imfilter(srcImg,hv1,'corr','circular');
            yd1 = imfilter(srcImg,hd1,'corr','circular');
            yh2 = imfilter(srcImg,hh2,'corr','circular');
            yv2 = imfilter(srcImg,hv2,'corr','circular');
            yd2 = imfilter(srcImg,hd2,'corr','circular');            
            coefExpctd = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ];
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test for default construction
        function testLevel2Size16x32(testCase)
            
            nLevels = 2;
            height = 16;
            width = 32;
            srcImg = rand(height,width);
            
            % Expected values
            ha1 = imfilter(upsample(upsample([ 1 1 ; 1 1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hh1 = imfilter(upsample(upsample([ 1 -1 ; 1 -1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hv1 = imfilter(upsample(upsample([ 1 1 ; -1 -1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hd1 = imfilter(upsample(upsample([ 1 -1 ; -1 1 ]/4,2,1).',2,1).',[ 1 1 ; 1 1 ]/4);
            hh2 = [ 1 -1 ; 1 -1 ]/4;
            hv2 = [ 1 1 ; -1 -1 ]/4;
            hd2 = [ 1 -1 ; -1 1 ]/4;                        
            ya1 = imfilter(srcImg,ha1,'corr','circular');
            yh1 = imfilter(srcImg,hh1,'corr','circular');
            yv1 = imfilter(srcImg,hv1,'corr','circular');
            yd1 = imfilter(srcImg,hd1,'corr','circular');
            yh2 = imfilter(srcImg,hh2,'corr','circular');
            yv2 = imfilter(srcImg,hv2,'corr','circular');
            yd2 = imfilter(srcImg,hd2,'corr','circular');            
            coefExpctd = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ];
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));

        end        

        % Test for default construction
        function testLevel3Size16x16(testCase)
            
            nLevels = 3;
            height = 16;
            width = 16;
            srcImg = rand(height,width);
            
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
            ya1 = imfilter(srcImg,ha1,'corr','circular');
            yh1 = imfilter(srcImg,hh1,'corr','circular');
            yv1 = imfilter(srcImg,hv1,'corr','circular');
            yd1 = imfilter(srcImg,hd1,'corr','circular');
            yh2 = imfilter(srcImg,hh2,'corr','circular');
            yv2 = imfilter(srcImg,hv2,'corr','circular');
            yd2 = imfilter(srcImg,hd2,'corr','circular');            
            yh3 = imfilter(srcImg,hh3,'corr','circular');
            yv3 = imfilter(srcImg,hv3,'corr','circular');
            yd3 = imfilter(srcImg,hd3,'corr','circular');                        
            coefExpctd = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ...
                yh3(:).' yv3(:).' yd3(:).' ];
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));

        end
        
        % Test for default construction
        function testLevel3Size16x32(testCase)
            
            nLevels = 3;
            height = 16;
            width = 32;
            srcImg = rand(height,width);
            
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
            ya1 = imfilter(srcImg,ha1,'corr','circular');
            yh1 = imfilter(srcImg,hh1,'corr','circular');
            yv1 = imfilter(srcImg,hv1,'corr','circular');
            yd1 = imfilter(srcImg,hd1,'corr','circular');
            yh2 = imfilter(srcImg,hh2,'corr','circular');
            yv2 = imfilter(srcImg,hv2,'corr','circular');
            yd2 = imfilter(srcImg,hd2,'corr','circular');            
            yh3 = imfilter(srcImg,hh3,'corr','circular');
            yv3 = imfilter(srcImg,hv3,'corr','circular');
            yd3 = imfilter(srcImg,hd3,'corr','circular');                        
            coefExpctd = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ...
                yh3(:).' yv3(:).' yd3(:).' ];
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));

        end
        

       % Test for default construction
        function testLevel1ReconstructionSize16x16(testCase)
            
            nLevels = 1;
            height = 16;
            width = 16;
                        
            % Expected values
            imgExpctd = rand(height,width);
            dimExpctd = [ height, width ];
            nPixels = numel(imgExpctd);

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
        
            % Analysis
            coefs = step(testCase.analyzer,imgExpctd);
            ca = reshape(coefs(1:nPixels),height,width);
            ch = reshape(coefs(nPixels+1:2*nPixels),height,width);
            cv = reshape(coefs(2*nPixels+1:3*nPixels),height,width);
            cd = reshape(coefs(3*nPixels+1:4*nPixels),height,width);
        
            % Reconstruction
            ha = [ 1 1 ; 1 1 ]/4;
            hh = [ 1 -1 ; 1 -1 ]/4;
            hv = [ 1 1 ; -1 -1 ]/4;
            hd = [ 1 -1 ; -1 1 ]/4;            
            ya = circshift(imfilter(ca,ha,'conv','circular'),[1 1]);
            yh = circshift(imfilter(ch,hh,'conv','circular'),[1 1]);
            yv = circshift(imfilter(cv,hv,'conv','circular'),[1 1]);
            yd = circshift(imfilter(cd,hd,'conv','circular'),[1 1]);
        
            % Actual values
            imgActual = ya + yh + yv + yd;
            
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
            dimExpctd = [ height, width ];
            nPixels = numel(imgExpctd);

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
        
            % Analysis
            coefs = step(testCase.analyzer,imgExpctd);
            ca = reshape(coefs(1:nPixels),height,width);
            ch = reshape(coefs(nPixels+1:2*nPixels),height,width);
            cv = reshape(coefs(2*nPixels+1:3*nPixels),height,width);
            cd = reshape(coefs(3*nPixels+1:4*nPixels),height,width);
        
            % Reconstruction
            ha = [ 1 1 ; 1 1 ]/4;
            hh = [ 1 -1 ; 1 -1 ]/4;
            hv = [ 1 1 ; -1 -1 ]/4;
            hd = [ 1 -1 ; -1 1 ]/4;            
            ya = circshift(imfilter(ca,ha,'conv','circular'),[1 1]);
            yh = circshift(imfilter(ch,hh,'conv','circular'),[1 1]);
            yv = circshift(imfilter(cv,hv,'conv','circular'),[1 1]);
            yd = circshift(imfilter(cd,hd,'conv','circular'),[1 1]);
        
            % Actual values
            imgActual = ya + yh + yv + yd;
            
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
            dimExpctd = [ height, width ];
            nPixels = numel(imgExpctd);

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
        
            % Analysis
            coefs = step(testCase.analyzer,imgExpctd);
            ca1 = reshape(coefs(1:nPixels),height,width);
            ch1 = reshape(coefs(nPixels+1:2*nPixels),height,width);
            cv1 = reshape(coefs(2*nPixels+1:3*nPixels),height,width);
            cd1 = reshape(coefs(3*nPixels+1:4*nPixels),height,width);
            ch2 = reshape(coefs(4*nPixels+1:5*nPixels),height,width);
            cv2 = reshape(coefs(5*nPixels+1:6*nPixels),height,width);
            cd2 = reshape(coefs(6*nPixels+1:7*nPixels),height,width);

            
            % Reconstruction
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
        
            % Actual values
            imgActual = ya1 + yh1 + yv1 + yd1 + yh2 + yv2 + yd2;
            
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
            dimExpctd = [ height, width ];
            nPixels = numel(imgExpctd);

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
        
            % Analysis
            coefs = step(testCase.analyzer,imgExpctd);
            ca1 = reshape(coefs(1:nPixels),height,width);
            ch1 = reshape(coefs(nPixels+1:2*nPixels),height,width);
            cv1 = reshape(coefs(2*nPixels+1:3*nPixels),height,width);
            cd1 = reshape(coefs(3*nPixels+1:4*nPixels),height,width);
            ch2 = reshape(coefs(4*nPixels+1:5*nPixels),height,width);
            cv2 = reshape(coefs(5*nPixels+1:6*nPixels),height,width);
            cd2 = reshape(coefs(6*nPixels+1:7*nPixels),height,width);

            
            % Reconstruction
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
        
            % Actual values
            imgActual = ya1 + yh1 + yv1 + yd1 + yh2 + yv2 + yd2;
            
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
            dimExpctd = [ height, width ];
            nPixels = numel(imgExpctd);

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
        
            % Analysis
            coefs = step(testCase.analyzer,imgExpctd);
            ca1 = reshape(coefs(1:nPixels),height,width);
            ch1 = reshape(coefs(nPixels+1:2*nPixels),height,width);
            cv1 = reshape(coefs(2*nPixels+1:3*nPixels),height,width);
            cd1 = reshape(coefs(3*nPixels+1:4*nPixels),height,width);
            ch2 = reshape(coefs(4*nPixels+1:5*nPixels),height,width);
            cv2 = reshape(coefs(5*nPixels+1:6*nPixels),height,width);
            cd2 = reshape(coefs(6*nPixels+1:7*nPixels),height,width);
            ch3 = reshape(coefs(7*nPixels+1:8*nPixels),height,width);
            cv3 = reshape(coefs(8*nPixels+1:9*nPixels),height,width);
            cd3 = reshape(coefs(9*nPixels+1:10*nPixels),height,width);            

            % Reconstruction
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
            
            % Actual values
            imgActual = ya1 + yh1 + yv1 + yd1 + yh2 + yv2 + yd2 + yh3 + yv3 + yd3;
            
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
            dimExpctd = [ height, width ];
            nPixels = numel(imgExpctd);

            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
        
            % Analysis
            coefs = step(testCase.analyzer,imgExpctd);
            ca1 = reshape(coefs(1:nPixels),height,width);
            ch1 = reshape(coefs(nPixels+1:2*nPixels),height,width);
            cv1 = reshape(coefs(2*nPixels+1:3*nPixels),height,width);
            cd1 = reshape(coefs(3*nPixels+1:4*nPixels),height,width);
            ch2 = reshape(coefs(4*nPixels+1:5*nPixels),height,width);
            cv2 = reshape(coefs(5*nPixels+1:6*nPixels),height,width);
            cd2 = reshape(coefs(6*nPixels+1:7*nPixels),height,width);
            ch3 = reshape(coefs(7*nPixels+1:8*nPixels),height,width);
            cv3 = reshape(coefs(8*nPixels+1:9*nPixels),height,width);
            cd3 = reshape(coefs(9*nPixels+1:10*nPixels),height,width);            

            % Reconstruction
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
            
            % Actual values
            imgActual = ya1 + yh1 + yv1 + yd1 + yh2 + yv2 + yd2 + yh3 + yv3 + yd3;
            
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
            srcImg = rand(height,width);
            
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
            ya1 = imfilter(srcImg,ha1,'corr','circular');
            yh1 = imfilter(srcImg,hh1,'corr','circular');
            yv1 = imfilter(srcImg,hv1,'corr','circular');
            yd1 = imfilter(srcImg,hd1,'corr','circular');
            yh2 = imfilter(srcImg,hh2,'corr','circular');
            yv2 = imfilter(srcImg,hv2,'corr','circular');
            yd2 = imfilter(srcImg,hd2,'corr','circular');            
            yh3 = imfilter(srcImg,hh3,'corr','circular');
            yv3 = imfilter(srcImg,hv3,'corr','circular');
            yd3 = imfilter(srcImg,hd3,'corr','circular');                        
            yh4 = imfilter(srcImg,hh4,'corr','circular');
            yv4 = imfilter(srcImg,hv4,'corr','circular');
            yd4 = imfilter(srcImg,hd4,'corr','circular');                                    
            coefExpctd = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ...
                yh3(:).' yv3(:).' yd3(:).' ...
                yh4(:).' yv4(:).' yd4(:).' ];
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));

        end
        
        % Test for default construction
        function testLevel4Size32x64(testCase)
            
            nLevels = 4;
            height = 32;
            width = 64;
            srcImg = rand(height,width);
            
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
            ya1 = imfilter(srcImg,ha1,'corr','circular');
            yh1 = imfilter(srcImg,hh1,'corr','circular');
            yv1 = imfilter(srcImg,hv1,'corr','circular');
            yd1 = imfilter(srcImg,hd1,'corr','circular');
            yh2 = imfilter(srcImg,hh2,'corr','circular');
            yv2 = imfilter(srcImg,hv2,'corr','circular');
            yd2 = imfilter(srcImg,hd2,'corr','circular');            
            yh3 = imfilter(srcImg,hh3,'corr','circular');
            yv3 = imfilter(srcImg,hv3,'corr','circular');
            yd3 = imfilter(srcImg,hd3,'corr','circular');                        
            yh4 = imfilter(srcImg,hh4,'corr','circular');
            yv4 = imfilter(srcImg,hv4,'corr','circular');
            yd4 = imfilter(srcImg,hd4,'corr','circular');                                    
            coefExpctd = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ...
                yh3(:).' yv3(:).' yd3(:).' ...
                yh4(:).' yv4(:).' yd4(:).' ];
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));

        end
        
        % Test for default construction
        function testLevel5Size64x64(testCase)
            
            nLevels = 5;
            height = 64;
            width = 64;
            srcImg = rand(height,width);
            
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
            ya1 = imfilter(srcImg,ha1,'corr','circular');
            yh1 = imfilter(srcImg,hh1,'corr','circular');
            yv1 = imfilter(srcImg,hv1,'corr','circular');
            yd1 = imfilter(srcImg,hd1,'corr','circular');
            yh2 = imfilter(srcImg,hh2,'corr','circular');
            yv2 = imfilter(srcImg,hv2,'corr','circular');
            yd2 = imfilter(srcImg,hd2,'corr','circular');            
            yh3 = imfilter(srcImg,hh3,'corr','circular');
            yv3 = imfilter(srcImg,hv3,'corr','circular');
            yd3 = imfilter(srcImg,hd3,'corr','circular');            
            yh4 = imfilter(srcImg,hh4,'corr','circular');
            yv4 = imfilter(srcImg,hv4,'corr','circular');
            yd4 = imfilter(srcImg,hd4,'corr','circular');                        
            yh5 = imfilter(srcImg,hh5,'corr','circular');
            yv5 = imfilter(srcImg,hv5,'corr','circular');
            yd5 = imfilter(srcImg,hd5,'corr','circular');                                    
            coefExpctd = [ ya1(:).' yh1(:).' yv1(:).' yd1(:).' ...
                yh2(:).' yv2(:).' yd2(:).' ...
                yh3(:).' yv3(:).' yd3(:).' ...
                yh4(:).' yv4(:).' yd4(:).' ...
                yh5(:).' yv5(:).' yd5(:).' ];
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*            
            testCase.analyzer = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
            
            % Actual values
            coefActual = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));

        end
        
    end
    
end

