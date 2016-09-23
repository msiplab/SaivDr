classdef DictionaryLearning
    
    properties
        orgImg
        nsolt
        NumberOfCoefs = 10000
        StageCount
        MaxStageCount = 10
        MaxFunctionEvaluations = 100000
        MaxIterations = 1000
        Angles = []
    end
    
    methods
        function obj = DictionaryLearning()
            % Prepare a source image
            srcImg = imread('peppers.png');
            srcImg = rgb2gray(srcImg);
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
            
            opt = optimoptions(@fminunc,...
                'Display','iter-detailed',...
                'Algorithm','quasi-newton',...
                'UseParallel',true,...
                'GradObj','off',...
                'DiffMaxChange',2*pi,...
                'StepTolerance',10e-8,...
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
                value = sum(abs(diff(:)).^2)/numel(diff);
            end
            func = @objFunc;
        end
        
%         function func = getObjFuncWithGrad(obj,coefvec,scales)
%             function [value, grad] = objFunc(angs)
%                 value = getObjFunc(obj,coefvec,scales);
%                 grad = zeros(size(angs));
%                 for idx = 1:length(angs)
%                     dangs = angs;
%                     dangs(idx) = angs(idx)+pi/2;
%                     release(obj.nsolt);
%                     set(obj.nsolt,'Angles',dangs);
%                     dsynth = saivdr.dictionary.nsoltx.NsoltSynthesis2dSystem('LpPuFb2d',obj.nsolt);
%                     
%                     tmp1 = conj(diff).*step(dsynth,coefs,scales);
%                     grad(idx) = -2*real(sum(tmp1(:)))/numel(tmp1);
%                 end
%             end
%             func = @objFunc;
%         end
    end
end
