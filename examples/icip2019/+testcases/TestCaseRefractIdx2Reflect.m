classdef TestCaseRefractIdx2Reflect < matlab.unittest.TestCase
    %TESTCASEREFRACTIDX2REFL このクラスの概要をここに記述
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
            
            phimExpctd = 'Reflection';
            outmExpctd = 'Function';
            
            % インスタンス生成
            target = RefractIdx2Reflect();
            
            phimActual = target.PhiMode;
            outmActual = target.OutputMode;
            
            testCase.verifyEqual(phimActual,phimExpctd);
            testCase.verifyEqual(outmActual,outmExpctd);
            
        end
        
        function testPhiMode(testCase,mode)
            
            % インスタンス生成
            target = RefractIdx2Reflect('PhiMode',mode);
            
            modeActual = target.PhiMode;
            
            testCase.verifyEqual(modeActual,mode);
            
        end
        
        function testStepReflection(testCase,dim1,dim2,dim3)
            
            
            % 設定
            height = dim1;
            width  = dim2;
            depth  = dim3;
            phiMode   = 'Reflection';
            srcImg = rand(height,width,depth);
            
            %
            arrayDltU = testCase.dltFcn(srcImg);
            arrayAddU = testCase.adtFcn(srcImg);
            resExpctd = -(1./(arrayAddU.*arrayAddU)).*abs(arrayDltU).*arrayDltU;
            
            % インスタンス生成
            target = RefractIdx2Reflect('PhiMode',phiMode);
            
            %
            resActual = target.step(srcImg);
            
            % 評価
            testCase.verifySize(resActual,size(resExpctd));
            diff = max(abs(resExpctd(:) - resActual(:))./abs(resExpctd(:)));
            testCase.verifyEqual(resActual,resExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testStepLinear(testCase,dim1,dim2,dim3,vrange)
            
            
            % 設定
            height = dim1;
            width  = dim2;
            depth  = dim3;
            phiMode   = 'Linear';
            srcImg = rand(height,width,depth);
            
            %
            vmin   = vrange(1);
            vmax   = vrange(2);
            beta1   = 2*abs(vmax-vmin)/(vmax+vmin)^2;
            resExpctd = -beta1*testCase.dltFcn(srcImg);
            
            % インスタンス生成
            target = RefractIdx2Reflect(...
                'PhiMode',phiMode,...
                'VRange',vrange);
            
            %
            resActual = target.step(srcImg);
            
            % 評価
            testCase.verifySize(resActual,size(resExpctd));
            diff = max(abs(resExpctd(:) - resActual(:))./abs(resExpctd(:)));
            testCase.verifyEqual(resActual,resExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testStepSignedQuadratic(testCase,dim1,dim2,dim3,vrange)
            
            % 設定
            height = dim1;
            width  = dim2;
            depth  = dim3;
            phiMode   = 'Signed-Quadratic';
            srcImg = rand(height,width,depth);
            
            %
            vmin   = vrange(1);
            vmax   = vrange(2);
            beta2   = 4/(vmax+vmin)^2;
            arrayDltU = testCase.dltFcn(srcImg);
            resExpctd = -beta2*abs(arrayDltU).*arrayDltU;
            
            % インスタンス生成
            target = RefractIdx2Reflect(...
                'PhiMode',phiMode,...
                'VRange',vrange);
            
            %
            resActual = target.step(srcImg);
            
            % 評価
            testCase.verifySize(resActual,size(resExpctd));
            diff = max(abs(resExpctd(:) - resActual(:))./abs(resExpctd(:)));
            testCase.verifyEqual(resActual,resExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testStepIdentity(testCase,dim1,dim2,dim3)
            
            % 設定
            height = dim1;
            width  = dim2;
            depth  = dim3;
            phiMode   = 'Identity';
            srcImg = rand(height,width,depth);
            
            %
            resExpctd = srcImg;
            
            % インスタンス生成
            target = RefractIdx2Reflect('PhiMode',phiMode);
            
            %
            resActual = target.step(srcImg);
            
            % 評価
            testCase.verifySize(resActual,size(resExpctd));
            diff = max(abs(resExpctd(:) - resActual(:))./abs(resExpctd(:)));
            testCase.verifyEqual(resActual,resExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testStepJacobian(testCase,mode,vrange)
            
            % 設定
            height = 8;
            width  = 8;
            depth  = 16;
            vmin   = vrange(1);
            vmax   = vrange(2);
            srcImg = (vmax-vmin)*rand(height,width,depth)+vmin;
            delta = 1e-6; % 数値微分の刻み幅
            
            % 数値的微分
            nRows = numel(srcImg);
            jacobExpctd = zeros(nRows);
            phi = RefractIdx2Reflect('PhiMode',mode,'VRange',vrange);
            for iRow = 1:nRows
                du = zeros(size(srcImg));
                du(iRow) = delta;
                vecD = (phi.step(srcImg+du/2)-phi.step(srcImg-du/2))/delta;
                jacobExpctd(iRow,:) = vecD(:);
            end
            
            % インスタンス生成
            target = RefractIdx2Reflect(...
                'PhiMode',mode,...
                'VRange',vrange,...
                'OutputMode','Jacobian');
            
            % 解析的微分
            jacobActual = target.step(srcImg);
            
            % 評価
            testCase.verifySize(jacobActual,size(jacobExpctd));
            diff = max(abs(jacobExpctd(:) - jacobActual(:)));
            testCase.verifyEqual(jacobActual,jacobExpctd,'AbsTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testStepGradient(testCase,mode,vrange)
            
            % 設定
            height = 8;
            width  = 8;
            depth  = 16;
            vmin   = vrange(1);
            vmax   = vrange(2);
            delta = 1e-4; % 数値微分の刻み幅
            
            % 準備
            vrange = [vmin vmax];
            u   = (vmax-vmin)*rand(height,width,depth)+vmin;
            phi = RefractIdx2Reflect('PhiMode','Reflection');                                    
            v   = phi.step(u)+0.1*randn(size(u));

              
            % 数値的勾配
            phiapx = RefractIdx2Reflect(...
                'PhiMode',mode,...
                'VRange',vrange,...
                'OutputMode','Function');                        
            [nRows,nCols,nLays] = size(u);
            gradExpctd = zeros(size(u));
            for iLay = 1:nLays
                for iCol = 1:nCols
                    for iRow = 1:nRows
                        du = zeros(size(u));
                        du(iRow,iCol,iLay) = delta;
                        % y = Φ(u)-r 
                        y1 = phiapx.step(u+du/2)-v;
                        y2 = phiapx.step(u-du/2)-v;                        
                        % f = (1/2)||y||_2^2                        
                        f1 = (1/2)*norm(y1(:),2)^2;                        
                        f2 = (1/2)*norm(y2(:),2)^2;                                                
                        %
                        dltF = (f1-f2)/delta;
                        gradExpctd(iRow,iCol,iLay) = dltF;
                    end
                end
            end
            
            % インスタンス生成
            target = RefractIdx2Reflect(...
                'PhiMode',mode,...
                'VRange',vrange,...
                'OutputMode','Gradient');
            
            % 解析的勾配
            r = phiapx.step(u)-v;
            gradActual = target.step(u,r);
            
            % 評価
            testCase.verifySize(gradActual,size(gradExpctd));
            diff = max(abs(gradExpctd(:) - gradActual(:)));
            testCase.verifyEqual(gradActual,gradExpctd,'AbsTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testStepCloneGradient(testCase,mode,vrange)
            
            % 設定
            height = 8;
            width  = 8;
            depth  = 16;
            vmin   = vrange(1);
            vmax   = vrange(2);
            delta = 1e-4; % 数値微分の刻み幅
            
            % 期待値
            phiModeExpctd = mode;
            vrangeExpctd = vrange;
            outModeExpctd = 'Gradient';
            numInputsExpctd = 2;
            
            % 準備
            vrange = [vmin vmax];
            u   = (vmax-vmin)*rand(height,width,depth)+vmin;
            phi = RefractIdx2Reflect('PhiMode','Reflection');                                    
            v   = phi.step(u)+0.1*randn(size(u));
              
            % 数値的勾配
            phiapx = RefractIdx2Reflect(...
                'PhiMode',mode,...
                'VRange',vrange,...
                'OutputMode','Function');                        
            [nRows,nCols,nLays] = size(u);
            gradExpctd = zeros(size(u));
            for iLay = 1:nLays
                for iCol = 1:nCols
                    for iRow = 1:nRows
                        du = zeros(size(u));
                        du(iRow,iCol,iLay) = delta;
                        % y = Φ(u)-r 
                        y1 = phiapx.step(u+du/2)-v;
                        y2 = phiapx.step(u-du/2)-v;                        
                        % f = (1/2)||y||_2^2                        
                        f1 = (1/2)*norm(y1(:),2)^2;                        
                        f2 = (1/2)*norm(y2(:),2)^2;                                                
                        %
                        dltF = (f1-f2)/delta;
                        gradExpctd(iRow,iCol,iLay) = dltF;
                    end
                end
            end
            
            % インスタンス生成
            target = clone(phiapx);
            target.release();
            target.OutputMode = 'Gradient';
            
            phiModeActual = target.PhiMode;
            vrangeActual  = target.VRange;
            outModeActual = target.OutputMode;
            
            % 解析的勾配
            r = phiapx.step(u)-v;
            gradActual = target.step(u,r);
            
            % 評価
            testCase.verifyEqual(phiModeActual,phiModeExpctd);
            testCase.verifyEqual(vrangeActual,vrangeExpctd);
            testCase.verifyEqual(outModeActual,outModeExpctd);
            testCase.verifySize(gradActual,size(gradExpctd));
            diff = max(abs(gradExpctd(:) - gradActual(:)));
            testCase.verifyEqual(gradActual,gradExpctd,'AbsTol',1e-7,sprintf('%g',diff));
            
        end
        
    end
    
end
