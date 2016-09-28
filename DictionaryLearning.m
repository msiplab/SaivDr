classdef DictionaryLearning
    
    properties
        orgImg
        nsolt
        StageCount
        Angles = []
        ErrPerPix = []
    end
    
    properties (Constant)
        NumberOfCoefs = 20000
        MaxStageCount = 20
        MaxFunctionEvaluations = 100000
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
            nChs    = 6; % # of channels
            nOrd    = [2 2]; % Polyphase order
            nVm     = 0;     % # of vanishing moments
            
            obj.nsolt = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrd,...
                'NumberOfVanishingMoments',nVm);
            
        end
        
        function update(obj)
            
            angs = get(obj.nsolt,'Angles');
            obj.Angles = zeros(length(angs),obj.MaxStageCount);
            obj.ErrPerPix = zeros(obj.MaxStageCount);
            
            opt = optimoptions(@fminunc,...
                'Display','iter-detailed',...
                'Algorithm','quasi-newton',...
                'UseParallel',true,...
                'GradObj','off',...
                'DiffMaxChange',2*pi,...
                'OptimalityTolerance',1e-8,...
                'StepTolerance',1e-7,...
                'MaxFunctionEvaluations',obj.MaxFunctionEvaluations,...
                'MaxIterations',obj.MaxIterations);
            
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
                preangs = get(obj.nsolt,'Angles');
                objFunc = getObjFunc(obj,coefvec,scales);
                obj.Angles(:,idx) = fminunc(objFunc,preangs,opt);
                obj.ErrPerPix(idx) = objFunc(obj.Angles(:,idx));
                set(obj.nsolt,'Angles',obj.Angles(:,idx));
                
                atmimshow(obj.nsolt);
            end
        end
        
        % 目的関数を戻り値とするクロージャ
        function func = getObjFunc(obj,coefvec,scales)
            function value = objFunc(angs)
                release(obj.nsolt);
                set(obj.nsolt,'Angles',angs);
                synthesizer = saivdr.dictionary.nsoltx.NsoltSynthesis2dSystem('LpPuFb2d',obj.nsolt);
                
                diff = obj.orgImg - step(synthesizer,coefvec,scales);
                sf = 10e6; % scaling factor
                value = sf*sum(abs(diff(:)).^2)/numel(diff);
            end
            func = @objFunc;
        end
    end
end
