classdef DictionaryLearning
    
    properties
        orgImg
        nsolt
        StageCount
        Angles
        Mus
        ErrPerPix
        NumberOfChannels = 8
        NumberOfTreeLevels = 4
    end
    
    properties (Constant)
        CropSize = [64 64]
        NumberOfCoefs = 1500
        MaxStageCount = 50
        PreviousDictionaryFile = ''
        PreviousStageCount = 0
    end
    
    methods
        function obj = DictionaryLearning()
            % Prepare a source image
            srcImg = imread('18_ibushi.normal.png');
            srcImg = im2double(srcImg);
            obj.orgImg = (srcImg(:,:,1) + 1i*srcImg(:,:,2))/sqrt(2);
            
%             width  = 64; % Width
%             height = 64; % Height
%             px     = 64;  % Horizontal position of cropping
%             py     = 64;  % Vertical position of cropping
%             obj.orgImg = im2double(srcImg(py:py+height-1,px:px+width-1,:));
            
            % Parameters for NSOLT
            nDec    = [2 2]; % Decimation factor
            nChs    = obj.NumberOfChannels; % # of channels
            nOrd    = [2 2]; % Polyphase order
            nVm     = 0;     % # of vanishing moments
            
            obj.nsolt = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrd,...
                'NumberOfVanishingMoments',nVm);
            
            angs = get(obj.nsolt,'Angles');
            obj.Angles = zeros(length(angs),obj.MaxStageCount);
            
            mus = get(obj.nsolt,'Mus');
            sizeOfMus = size(mus);
            obj.Mus = zeros(sizeOfMus(1),sizeOfMus(2),obj.MaxStageCount);
            for idx = 1:obj.MaxStageCount
                obj.Angles(:,idx) = angs;
                obj.Mus(:,:,idx) = mus;
            end
            
            if ~strcmp(obj.PreviousDictionaryFile,'')
                s = load(obj.PreviousDictionaryFile,'dictionary');
                prevDic = s.dictionary;
                obj.nsolt = clone(prevDic.nsolt);
                
                idx = obj.PreviousStageCount;
                set(obj.nsolt,'Angles',prevDic.Angles(:,idx));
                set(obj.nsolt,'Mus',prevDic.Mus(:,:,idx));
            end
        end
        
        function update(obj)
            
            filename = ['dictionaries/','dic',datestr(now,'yyyymmdd_HHMMSS')];
            
            obj.ErrPerPix = zeros(obj.MaxStageCount,1);
            
            nch = get(obj.nsolt,'NumberOfChannels');
            angs = get(obj.nsolt,'Angles');
            angs = angs(nch+1:end);
            
            opt = optimoptions(@lsqnonlin,...
                'Algorithm','trust-region-reflective',...
                'Display','iter-detailed',...
                'FiniteDifferenceStepSize',1e0*sqrt(eps),...
                'FunctionTolerance', 1e-8,...
                'OptimalityTolerance',1e-6,...
                'StepTolerance',1e-8,...
                'TypicalX',pi*1e0*ones(size(angs)),...
                'UseParallel',true);
            
            %             opt = optimoptions(@lsqnonlin,...
            %                 'Algorithm','trust-region-reflective',...
            %                 'Display','iter-detailed',...
            %                 'Jacobian','on',...
            %                 'MaxFunctionEvaluations',obj.MaxFunctionEvaluations,...
            %                 'MaxIterations',obj.MaxIterations,...
            %                 'StepTolerance',1e-8,...
            %                 'UseParallel',true);
            
            for idx = 1:obj.MaxStageCount
                obj.StageCount = idx;
                fprintf('StageCount = %d\n',obj.StageCount);
                
                % coefficients optimization
                fprintf('start coefficients optimization stage.\n');
                
                %p = floor((size(obj.orgImg)-obj.CropSize).*rand(1,2));
                px     = 64;  % Horizontal position of cropping
                py     = 64;  % Vertical position of cropping
                cropImg = im2double(obj.orgImg(py:py+obj.CropSize(1)-1,px:px+obj.CropSize(2)-1,:));
%                 cropImg = obj.orgImg(p(1):p(1)+obj.CropSize(1)-1,p(2):p(2)+obj.CropSize(2)-1);
%                 obsImg = imnoise(real(cropImg),'gaussian',0,(40/255)^2)...
%                     + 1i*imnoise(imag(cropImg),'gaussian',0,(40/255)^2);
                obsImg = cropImg;
                
                import saivdr.dictionary.nsoltx.*
                import saivdr.sparserep.*
                analyzer = NsoltAnalysis2dSystem('LpPuFb2d',obj.nsolt);
                synthesizer = NsoltSynthesis2dSystem('LpPuFb2d',obj.nsolt);
                
                iht = IterativeHardThresholding(...
                    'NumberOfTreeLevels',obj.NumberOfTreeLevels,...
                    'Synthesizer',synthesizer,'AdjOfSynthesizer',analyzer);
                [~,coefvec,scales] = step(iht,obsImg,obj.NumberOfCoefs);
                fprintf('end coefficients optimization stage.\n');
                %dictionary update
                fprintf('start dictionary update stage.\n');
                nch = obj.NumberOfChannels;
                %                 mus = get(obj.nsolt,'Mus');
                %                 mus = 2*(rand(size(mus)) >= 0.5) - 1;
                %                 obj.Mus(:,:,idx) = mus;
                %                 set(obj.nsolt,'Mus',mus);
                
                %preangs = obj.Angles(:,idx);
                preangs = get(obj.nsolt,'Angles');
                %preangs = pi*(2*rand(size(preangs))-ones(size(preangs)));
                preangs = preangs(nch+1:end);
                objFunc = getObjFunc(obj,coefvec,scales,cropImg);
                
                lb = [];
                ub = [];
                %lb = -pi*ones(size(preangs));
                %ub = pi*ones(size(preangs));
                obj.Angles(nch+1:end,idx) = lsqnonlin(objFunc,preangs,lb,ub,opt);
                obj.ErrPerPix(idx) = norm(objFunc(obj.Angles(nch+1:end,idx)));
                set(obj.nsolt,'Angles',obj.Angles(:,idx));
                fprintf('end dictionary update stage\n');
                
                % save this DictionaryLearning object;
                dictionary = obj;
                save(filename,'dictionary');
            end
        end
        
        function func = getObjFunc(obj,coefvec,scales,cropImg)
            ns = clone(obj.nsolt);
            function [value,J] = objFunc(angs)
                import saivdr.dictionary.nsoltx.*
                release(ns);
                
                set(ns,'Angles',[zeros(obj.NumberOfChannels,1);angs]);
                synthesizer = NsoltSynthesis2dSystem('LpPuFb2d',ns);
                
                diff = cropImg - step(synthesizer,coefvec,scales);
                value = [real(diff(:)),imag(diff(:))];
                
                if nargout > 1
                    Jcmpx = complex(zeros(length(value),length(angs)));
                    for idx = 1:length(angs)
                        dangs = angs;
                        dangs(idx) = angs(idx)+pi/2;
                        set(ns,'Angles',[zeros(obj.NumberOfChannels,1);dangs]);
                        synthesizer = NsoltSynthesis2dSystem('LpPuFb2d',ns);
                        dimg = -step(synthesizer,coefvec,scales);
                        Jcmpx(:,idx) = dimg(:);
                    end
                    J = [real(Jcmpx);imag(Jcmpx)];
                    fprintf('max(J) = %f, min(J) = %f\n',max(J(:)),min(J(:)));
                end
            end
            func = @objFunc;
        end
    end
end