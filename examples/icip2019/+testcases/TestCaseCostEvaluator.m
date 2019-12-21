classdef TestCaseCostEvaluator < matlab.unittest.TestCase
    %TESTCASEPLGSOFTTHRESHOLDING このクラスの概要をここに記述
    %   詳細説明をここに記述
    
    properties
        dltFcn
        adtFcn
    end
    
    properties (TestParameter)
        mode = { 'Reflection','Linear','Signed-Quadratic','Identity'};
        dim1 = struct('small',8, 'large', 32);
        dim2 = struct('small',8, 'large', 32);
        dim3 = struct('small',8, 'large', 32);
        vrange = struct('small',[1.0 1.5], 'large',[0.5 2.0]);
    end
    
    methods (TestClassSetup)
        function addFunctions(testCase)
            kernelxy = kron([ 1 2 1 ].', [ 1 2 1 ]);
            kernelz  = permute([ 1 0 -1 ].',[ 2 3 1 ]);
            sobel3d = convn(kernelxy,kernelz)/32;
            absbl3d = abs(sobel3d);
            testCase.dltFcn = @(x) imfilter(x,sobel3d,'conv','circ');
            testCase.adtFcn = @(x) imfilter(x,absbl3d,'conv','circ');
        end
    end
    
    methods (Test)
        
        function testConstruction(testCase)
            
            % 設定
            obsExpctd = rand(8,8,8);
            outmExpctd = 'Function';
            
            % インスタンス化
            target = CostEvaluator('Observation',obsExpctd);
            
            %
            obsActual = target.Observation;
            outmActual = target.OutputMode;
            
            % 評価
            testCase.verifyEqual(outmActual,outmExpctd);
            testCase.verifySize(obsActual,size(obsExpctd));
            diff = max(abs(obsExpctd(:) - obsActual(:))./abs(obsExpctd(:)));
            testCase.verifyEqual(obsActual,obsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testStepFunction(testCase,dim1,dim2,dim3,mode,vrange)
            
            % 設定
            height = dim1;
            width  = dim2;
            depth  = dim3;
            refImg = rand(height,width,depth);
            
            % 観測信号
            phi = RefractIdx2Reflect(...
                'PhiMode','Reflection',...
                'VRange',vrange);
            
            %[refImg,scales] = adjdic.step(srcImg);
            coh3 = Coherence3();
            obsImg = coh3.step(phi.step(refImg),'Forward') ...
                + 0.1*randn(size(refImg));
            
            % 期待値
            phiapx = RefractIdx2Reflect(...
                'PhiMode',mode,...
                'VRange',vrange);
            y = coh3.step(phiapx.step(refImg),'Forward')-obsImg;
            costExpctd = (1/2)*norm(y(:),2)^2;
            
            % インスタンス生成
            target = CostEvaluator(...
                'Observation',obsImg,...
                'MeasureProcess',coh3,...
                'RefIdx2Ref',phiapx);
            %
            costActual = target.step(refImg);
            
            % 評価
            testCase.verifySize(costActual,size(costExpctd));
            diff = max(abs(costExpctd(:) - costActual(:))./abs(costExpctd(:)));
            testCase.verifyEqual(costActual,costExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testStepGradient(testCase,mode,vrange)
            
            % 設定
            delta  = 1e-4;
            height = 8;
            width  = 8;
            depth  = 8;
            srcImg = rand(height,width,depth);
            
            % 観測信号
            pScale = 1.00;
            pSigma = 8.00;
            pFrq   = 0.25;
            coh3 = Coherence3(...
                'Scale',pScale,...
                'Sigma',pSigma,...
                'Frequency',pFrq);
            
            %
            nLevels = 1;
            fwdDic = DicUdHaarRec3();
            adjDic = DicUdHaarDec3('NumLevels',nLevels);
            phi = RefractIdx2Reflect('PhiMode','Reflection');
            obsImg = coh3.step(phi.step(srcImg),'Forward') ...
                + 0.1*randn(size(srcImg));
            
            % 期待値
            phiapx = RefractIdx2Reflect(...
                'PhiMode',mode,...
                'VRange',vrange);
            costref = CostEvaluator(...
                'OutputMode','Function',...
                'Observation',obsImg,...
                'MeasureProcess',coh3,...
                'RefIdx2Ref',phiapx);
            %
            [x,info] = adjDic.step(srcImg);
            [nRows,nCols,nLays] = size(x);
            gradExpctd = zeros(size(x));
            for iLay = 1:nLays
                for iCol = 1:nCols
                    for iRow = 1:nRows
                        dx = zeros(size(x));
                        dx(iRow,iCol,iLay) = delta;
                        dltF = (...
                            costref.step(fwdDic.step(x+dx/2,info))...
                            -...
                            costref.step(fwdDic.step(x-dx/2,info))...
                            )/delta;
                        gradExpctd(iRow,iCol,iLay) = dltF;
                    end
                end
            end
            
            % インスタンス生成
            target = CostEvaluator(...
                'OutputMode','Gradient',...
                'Observation',obsImg,...
                'MeasureProcess',coh3,...
                'RefIdx2Ref',phiapx);
            
            %
            %ga = adjDic(fcn_grad_f(fwdDic(x,info),v,msrFcns,sobel3d,phiMode,vrange));
            u = fwdDic(x,info);
            gradActual = adjDic.step(target.step(u));
            
            % 評価
            testCase.verifySize(gradActual,size(gradExpctd));
            diff = max(abs(gradExpctd(:) - gradActual(:)));
            testCase.verifyEqual(gradActual,gradExpctd,'AbsTol',1e-7,sprintf('%g',diff));
            
        end
        
    end
end
