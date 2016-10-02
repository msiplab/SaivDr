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
        NumberOfCoefs = 20000
        MaxStageCount = 30
%        MaxFunctionEvaluations = 100000
%        MaxIterations = 3000
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
            
            opt = optimoptions(@lsqnonlin,...
                'Algorithm','trust-region-reflective',...
                'Display','iter-detailed',...
                'DiffMaxChange',2*pi,...
                'UseParallel',true);
%                 'MaxFunctionEvaluations',obj.MaxFunctionEvaluations,...
%                 'MaxIterations',obj.MaxIterations,...
%                 'OptimalityTolerance',1e-8,...
%                 'StepTolerance',1e-7,...     
            for idx = 1:obj.MaxStageCount
                obj.StageCount = idx;
                % coefficients optimization
                import saivdr.dictionary.nsoltx.*
                import saivdr.sparserep.*
                analyzer = NsoltAnalysis2dSystem('LpPuFb2d',obj.nsolt);
                synthesizer = NsoltSynthesis2dSystem('LpPuFb2d',obj.nsolt);
                
                iht = IterativeHardThresholding(...
                    'Synthesizer',synthesizer,'AdjOfSynthesizer',analyzer);
                [~,coefvec,scales] = step(iht,obj.orgImg,obj.NumberOfCoefs);
                
                %dictionary update
                nch = obj.NumberOfChannels;
                preangs = get(obj.nsolt,'Angles');
                preangs = zeros(size(preangs));
                preangs = preangs(nch+1:end);
                objFunc = getObjFunc(obj,coefvec,scales);
                %obj.Angles(nch+1:end,idx) = fminunc(objFunc,preangs,opt);
                
                lb = -pi*ones(size(preangs));
                ub = pi*ones(size(preangs));
                obj.Angles(nch+1:end,idx) = lsqnonlin(objFunc,preangs,lb,ub,opt);
                obj.ErrPerPix(idx) = norm(objFunc(obj.Angles(nch+1:end,idx)));
                set(obj.nsolt,'Angles',obj.Angles(:,idx));
                
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
            absCoef = abs(coefvec);
            range = 0:0.001:2.5;
            cdf = arrayfun(@(x) sum(absCoef <= x),range)/numel(absCoef)*100;
            %fprintf('[StageCount = %d] A number of null coefficients is %d (%.2f%%)\n',index,hoge,100*hoge/numel(absCoef));
            plot(range,cdf);
            grid on
        end
        
    end
end