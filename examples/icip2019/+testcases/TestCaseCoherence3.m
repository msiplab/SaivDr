classdef TestCaseCoherence3 < matlab.unittest.TestCase
    %UDHAARDEC3TESTCASE このクラスの概要をここに記述
    %   詳細説明をここに記述
    
    properties (TestParameter)
        scl  = struct('small',0.1, 'large', 10.0); 
        sgm  = struct('small',0.1, 'large', 10.0); 
        frq  = struct('small',0.1, 'large', 10.0); 
        dim1 = struct('small',8, 'large', 32);
        dim2 = struct('small',8, 'large', 32);
        dim3 = struct('small',8, 'large', 32);        
    end
    
    methods (Test)
        function testConstruction(testCase,scl,sgm,frq)
            
            % パラメータ
            pScale = scl;
            pSigma = sgm;
            pFrq   = frq;

            % カーネル
            len = 2*round(5*pSigma)+1; % 奇数に設定 
            n = -floor(len/2):floor(len/2); 
            gc = pScale*exp(-n.^2./(2*pSigma.^2)).*cos(2*pi*pFrq*n);
            kernelExpctd = permute(gc(:),[2 3 1]);
            
            target = Coherence3(...
                'Scale',pScale,...
                'Sigma',pSigma,...
                'Frequency',pFrq);
            
            kernelActual = target.Kernel;
            
            testCase.verifyEqual(kernelActual,kernelExpctd);
            
        end
        
        function testStepForward(testCase,scl,sgm,frq,dim1,dim2,dim3)

            % パラメータ
            pScale = scl;
            pSigma = sgm;
            pFrq   = frq;

            %
            height = dim1;
            width  = dim2;
            depth  = dim3;
            src = rand(height,width,depth);
            
            % カーネル
            len = 2*round(5*pSigma)+1; % 奇数に設定
            n = -floor(len/2):floor(len/2);
            gc = pScale*exp(-n.^2./(2*pSigma.^2)).*cos(2*pi*pFrq*n);
            kernel = permute(gc(:),[2 3 1]);

            resExpctd = imfilter(src,kernel,'conv','circ');
            
            target = Coherence3(...
                'Scale',pScale,...
                'Sigma',pSigma,...
                'Frequency',pFrq);

            resActual = target.step(src,'Forward');

            % 評価
            testCase.verifySize(resActual,size(resExpctd));
            resDist = max(abs(resExpctd(:)-resActual(:))./abs(resExpctd(:)));
            testCase.verifyEqual(resActual,resExpctd,'RelTol',1e-15,...
                sprintf('%g',resDist));
        end
        
        function testStepAdjoint(testCase,scl,sgm,frq,dim1,dim2,dim3)
            
            % パラメータ
            pScale = scl;
            pSigma = sgm;
            pFrq   = frq;

            %
            height = dim1;
            width  = dim2;
            depth  = dim3;
            src = rand(height,width,depth);
            
            % カーネル
            len = 2*round(5*pSigma)+1; % 奇数に設定
            n = -floor(len/2):floor(len/2);
            gc = pScale*exp(-n.^2./(2*pSigma.^2)).*cos(2*pi*pFrq*n);
            kernel = permute(gc(:),[2 3 1]);

            resExpctd = imfilter(src,kernel,'corr','circ');
            
            target = Coherence3(...
                'Scale',pScale,...
                'Sigma',pSigma,...
                'Frequency',pFrq);

            resActual = target.step(src,'Adjoint');

            % 評価
            testCase.verifySize(resActual,size(resExpctd));
            resDist = max(abs(resExpctd(:)-resActual(:))./abs(resExpctd(:)));
            testCase.verifyEqual(resActual,resExpctd,'RelTol',1e-15,...
                sprintf('%g',resDist));
        end        
    end
end

