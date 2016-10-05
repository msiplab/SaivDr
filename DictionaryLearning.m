classdef DictionaryLearning
    
    properties
        orgImg
        nsolt
        StageCount
        Angles
        Mus
        ErrPerPix
        NumberOfChannels = 6
    end
    
    properties (Constant)
        NumberOfCoefs = 24000
        MaxStageCount = 30
        MaxFunctionEvaluations = 4400
        MaxIterations = 1000
    end
    
    methods
        function obj = DictionaryLearning()
            % Prepare a source image
            srcImg = imread('18_ibushi.normal.png');
            srcImg = im2double(srcImg);
            srcImg = srcImg(:,:,1) + 1i*srcImg(:,:,2);
            
            width  = 256; % Width
            height = 256; % Height
            px     = 64;  % Horizontal position of cropping
            py     = 64;  % Vertical position of cropping
            obj.orgImg = im2double(srcImg(py:py+height-1,px:px+width-1,:));
            
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
        end
        
        function update(obj)
            
            obj.ErrPerPix = zeros(obj.MaxStageCount,1);
            
            nch = get(obj.nsolt,'NumberOfChannels');
            angs = get(obj.nsolt,'Angles');
            angs = angs(nch+1:end);
            
            opt = optimoptions(@lsqnonlin,...
                'Algorithm','trust-region-reflective',...
                'Display','iter-detailed',...
                'DiffMaxChange',2*pi,...
                'FiniteDifferenceStepSize',1e+2*sqrt(eps),...
                'MaxFunctionEvaluations',obj.MaxFunctionEvaluations,...
                'MaxIterations',obj.MaxIterations,...
                'OptimalityTolerance',1e-6,...
                'StepTolerance',1e-6,...     
                'TypicalX',pi/2*1e0*ones(size(angs)),...
                'UseParallel',true);
            
            for idx = 1:obj.MaxStageCount
                obj.StageCount = idx;
                fprintf('StageCount = %d\n',obj.StageCount);
                % coefficients optimization
                fprintf('start coefficients optimization stage.\n');
                import saivdr.dictionary.nsoltx.*
                import saivdr.sparserep.*
                analyzer = NsoltAnalysis2dSystem('LpPuFb2d',obj.nsolt);
                synthesizer = NsoltSynthesis2dSystem('LpPuFb2d',obj.nsolt);
                
                iht = IterativeHardThresholding(...
                    'Synthesizer',synthesizer,'AdjOfSynthesizer',analyzer);
                [~,coefvec,scales] = step(iht,obj.orgImg,obj.NumberOfCoefs);
                fprintf('end coefficients optimization stage.\n');
                %dictionary update
                fprintf('start dictionary update stage.\n');
                nch = obj.NumberOfChannels;
                mus = get(obj.nsolt,'Mus');
                mus = 2*(rand(size(mus)) >= 0.5) - 1;
                obj.Mus(:,:,idx) = mus;
                set(obj.nsolt,'Mus',mus);
                
                preangs = get(obj.nsolt,'Angles');
                preangs = pi*(2*rand(size(preangs))-ones(size(preangs)));
                preangs = preangs(nch+1:end);
                objFunc = getObjFunc(obj,coefvec,scales);
                
                lb = -pi*ones(size(preangs));
                ub = pi*ones(size(preangs));
                obj.Angles(nch+1:end,idx) = lsqnonlin(objFunc,preangs,lb,ub,opt);
                obj.ErrPerPix(idx) = norm(objFunc(obj.Angles(nch+1:end,idx)));
                set(obj.nsolt,'Angles',obj.Angles(:,idx));
                fprintf('end dictionary update stage\n');
                atmimshow(obj.nsolt);
                
                % save this DictionaryLearning object;
                filename = ['dictionaries/','dic',datestr(now,'yyyymmdd_HHMMSS')];
                dictionary = obj;
                save(filename,'dictionary');
            end
        end

        function func = getObjFunc(obj,coefvec,scales)
            function value = objFunc(angs)
                release(obj.nsolt);
                angs = [zeros(obj.NumberOfChannels,1);angs];
                set(obj.nsolt,'Angles',angs);
                synthesizer = saivdr.dictionary.nsoltx.NsoltSynthesis2dSystem('LpPuFb2d',obj.nsolt);
                
                diff = obj.orgImg - step(synthesizer,coefvec,scales);
                value = [real(diff(:)),imag(diff(:))];
            end
            func = @objFunc;
        end
        
        function viewSparsity(obj,index)
            import saivdr.dictionary.nsoltx.*
            release(obj.nsolt);
            set(obj.nsolt,'Angles',obj.Angles(:,index));
            analyzer = NsoltAnalysis2dSystem('LpPuFb2d',obj.nsolt);
            [coefvec,~] = step(analyzer,obj.orgImg,1);
            absCoef = sort(abs(coefvec),'descend');
            range = 0:0.1:100;
            nCoef = numel(absCoef);
            E = sum(absCoef);
            cdf = arrayfun(@(x) sum(absCoef(1:floor(x/100*nCoef))),range)/E*100;
            plot(range,cdf);
            grid on
        end
        
    end
end