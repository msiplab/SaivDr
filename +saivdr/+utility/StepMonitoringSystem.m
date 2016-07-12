classdef StepMonitoringSystem < matlab.System %#codegen
    %STEPMONITORINGSYSTEM Monitor and evaluate step results
    %
    % SVN identifier:
    % $Id: StepMonitoringSystem.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627    
    %
    
    properties (Nontunable)
        %
        ImageFigureHandle
        PlotFigureHandle
        %
        EvaluationType = 'uint8'
    end

    properties
        SourceImage
        ObservedImage
    end
    
    properties (Hidden, Transient)
        EvaluationTypeSet = ...
            matlab.system.StringSet(...
            {'uint8','int16','uint16','single','double'});
    end
    
    properties (Nontunable, PositiveInteger)
        MaxIter     = 1000;
    end
    
    properties (Nontunable, Logical)
        IsMSE  = false
        IsPSNR = false
        IsSSIM = false
        IsVisible = false
        IsVerbose = false
        IsPlotPSNR = false
        IsConversionToEvaluationType = true
    end

    properties(Hidden)
        PeakValue = 1
    end

    properties (SetAccess = private, GetAccess = public)
        nItr
        MSEs
        PSNRs
        SSIMs
    end
    
    properties (Access = private)
        hSrcImg
        hObsImg
        hResImg
        hPlotPsnr
    end
    
    methods
       
        function obj = StepMonitoringSystem(varargin)
            setProperties(obj,nargin,varargin{:})
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.nItr  = obj.nItr;
            s.hResImg = obj.hResImg;
            s.hPlotPsnr = obj.hPlotPsnr;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.nItr  = s.nItr;
            obj.hResImg = s.hResImg;
            obj.hPlotPsnr = s.hPlotPsnr;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function flag = isInactiveSubPropertyImpl(obj,propertyName)
            if strcmp(propertyName,'ImageFigureHandle')
                flag = ~obj.IsVisible;
            elseif strcmp(propertyName,'IsPlotPSNR')
                flag = ~obj.IsPSNR;
            elseif strcmp(propertyName,'PlotFigureHandle')
                flag = ~obj.IsPlotPSNR;
            elseif strcmp(propertyName,'PeakValue')
                flag = obj.IsConversionToEvaluationType;
            else
                flag = false;
            end
        end
        
        function validateInputsImpl(obj,varargin)
            resImg = varargin{1};
            if obj.IsSSIM && size(resImg,3) > 1
                  id  = 'SaivDr:IndexOutOfBoundsException';
                  msg = 'SSIM is available only for grayscale image.';
                  me  = MException(id, msg);
                  throw(me);
            end
        end
        
        function processTunedPropertiesImpl(obj)
            if obj.IsConversionToEvaluationType            
                if ~isempty(obj.SourceImage)
                    obj.SourceImage = convEvalType_(obj,obj.SourceImage);
                end
                %
                if ~isempty(obj.ObservedImage)
                    obj.ObservedImage= convEvalType_(obj,obj.ObservedImage);
                end
            end
        end
        
        function setupImpl(obj,varargin)

            if obj.IsConversionToEvaluationType
                % Peak value
                if strcmp(obj.EvaluationType,'uint8') || ...
                        strcmp(obj.EvaluationType,'uint16') || ...
                            strcmp(obj.EvaluationType,'int16')
                elseif  strcmp(obj.EvaluationType,'single') || ...
                        strcmp(obj.EvaluationType,'double')
                    obj.PeakValue = 1;
                else
                    error('SaivDr: Invalid data type was given.')
                end 
                % Data type conversion
                if ~isempty(obj.SourceImage)
                    obj.SourceImage   = convEvalType_(obj,obj.SourceImage);
                end
                if ~isempty(obj.ObservedImage)
                    obj.ObservedImage = convEvalType_(obj,obj.ObservedImage);
                end
            end
            
            if obj.IsSSIM && verLessThan('images','9.0');
                % Download ssim_index.m if it is absent
                saivdrRoot = getenv('SAIVDR_ROOT');
                if exist(sprintf('%s/ssim_index.m',saivdrRoot),'file') ~= 2
                    if labindex == 1
                        currentLocation = pwd;
                        cd(saivdrRoot);
                        ssimcode = ...
                            urlread('https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m');
                        fid = fopen('ssim_index.m','w');
                        fwrite(fid,ssimcode);
                        fprintf('ssim_index.m was downloaded.\n');
                        cd(currentLocation);
                    end
                    labBarrier
                    rehash
                end
            end
            if obj.IsVerbose
                fprintf('\n')
            end
            if obj.IsVisible
                if isempty(obj.ImageFigureHandle)
                    obj.ImageFigureHandle = figure;
                end
                nPics = 3;
                if isempty(obj.SourceImage)
                    nPics = nPics - 1;
                end
                if isempty(obj.ObservedImage)
                    nPics = nPics - 1;
                end
                figure(obj.ImageFigureHandle);
                % Figure handle for source image
                iPic = 1;
                if ~isempty(obj.SourceImage)
                    subplot(1,nPics,iPic)
                    if size(obj.SourceImage,3) < 4 
                        obj.hSrcImg = imshow(obj.SourceImage);
                    else
                        obj.hSrcImg = imshow(obj.SourceImage(:,:,ceil(end/2)));
                    end
                    set(obj.hSrcImg,'UserData','Source');
                    title('Source')
                    iPic = iPic + 1; 
                end
                % Figure handle for observed image
                if ~isempty(obj.ObservedImage)
                    subplot(1,nPics,iPic)
                    if size(obj.ObservedImage,3) < 4
                        obj.hObsImg = imshow(obj.ObservedImage);
                    else
                        obj.hObsImg = imshow(obj.ObservedImage(:,:,ceil(end/2)));
                    end
                    set(obj.hObsImg,'UserData','Observed');
                    title('Observed')
                    iPic = iPic + 1;
                end
                % Figure handle for result image
                if obj.IsConversionToEvaluationType
                    resImg = convEvalType_(obj,varargin{1});
                else
                    resImg = varargin{1};
                end
                subplot(1,nPics,iPic)
                if size(resImg,3) < 4
                    obj.hResImg = imshow(resImg);
                else
                    obj.hResImg = imshow(resImg(:,:,ceil(end/2)));
                end
                set(obj.hResImg,'UserData','Result');
                title('Result (nItr =    0)');                                
            end
            if obj.IsPlotPSNR
                if isempty(obj.PlotFigureHandle)
                    obj.PlotFigureHandle = figure;
                end
                figure(obj.PlotFigureHandle)
                obj.hPlotPsnr = plot(0);
                xlabel('#Iteration')
                ylabel('PSNR [dB]')
            end
        end
        
        function resetImpl(obj)
            obj.nItr = 0;
            if obj.IsMSE
                obj.MSEs = zeros(1,obj.MaxIter);
            end
            if obj.IsPSNR
                obj.PSNRs = zeros(1,obj.MaxIter);
            end
            if obj.IsSSIM
                obj.SSIMs = zeros(1,obj.MaxIter);
            end            
        end
                    
        function varargout = stepImpl(obj,varargin)
            obj.nItr = obj.nItr + 1;
            iOut = 0;
            %
            if obj.IsConversionToEvaluationType            
                resImg = convEvalType_(obj,varargin{1});
            else
                resImg = varargin{1};
            end
            %
            if obj.IsVerbose
                fprintf('(% 4d) ',obj.nItr)
            end
            % MSE
            if obj.IsMSE
                iOut = iOut + 1;
                obj.MSEs(obj.nItr)   = mse_(obj,resImg);
                varargout{iOut} = obj.MSEs;
                if obj.IsVerbose
                    fprintf(' MSE = %6.4g ',obj.MSEs(obj.nItr))
                end
            end
            % PSNR
            if obj.IsPSNR
                iOut = iOut + 1;
                obj.PSNRs(obj.nItr)  = psnr_(obj,resImg);
                varargout{iOut} = obj.PSNRs;
                if obj.IsVerbose
                    fprintf(' PSNR = %6.2f [dB] ',obj.PSNRs(obj.nItr))
                end                
                if obj.IsPlotPSNR
                    set(obj.hPlotPsnr,...
                        'XData',1:obj.nItr,...
                        'YData',obj.PSNRs(1:obj.nItr));
                    drawnow
                end
            end            
            % SSIM
            if obj.IsSSIM
                iOut = iOut + 1;
                obj.SSIMs(obj.nItr)  = ssim_(obj,resImg);
                varargout{iOut} = obj.SSIMs;                
                if obj.IsVerbose
                    fprintf(' SSIM = %6.3f ',obj.SSIMs(obj.nItr))
                end                                
            end
            %
            if obj.IsVerbose
                fprintf('\n')
            end
            %
            if obj.IsVisible
                if ~isempty(obj.SourceImage)
                    if size(obj.SourceImage,3) > 4
                        srcImg = obj.SourceImage(:,:,ceil(end/2));
                    else
                        srcImg = obj.SourceImage;
                    end
                    set(obj.hSrcImg,'CData',srcImg);
                end
                %
                if ~isempty(obj.ObservedImage)                
                    if size(resImg,3) > 4
                        obsImg = obj.ObservedImage(:,:,ceil(end/2));
                    else
                        obsImg = obj.ObservedImage;
                    end
                    set(obj.hObsImg,'CData',obsImg);
                end
                %
                if size(resImg,3) > 4
                    resImg = resImg(:,:,ceil(end/2));
                end
                set(obj.hResImg,'CData',resImg);
                title(get(obj.hResImg,'Parent'),...
                    sprintf('Result (nItr = % 4d)',obj.nItr))
                drawnow
            end
            
        end
        
        function N = getNumInputsImpl(~)
            N = 1;
        end
        
        function N = getNumOutputsImpl(obj)
            N = 0;
            if obj.IsMSE
                N = N + 1;
            end
            if obj.IsPSNR
                N = N + 1;
            end    
            if obj.IsSSIM
                N = N + 1;
            end              
        end
        
    end
    
    methods (Access = private)
    
        function outimg = convEvalType_(obj,inimg)
            if strcmp(obj.EvaluationType,'uint8')
                outimg = im2uint8(inimg);
            elseif strcmp(obj.EvaluationType,'uint16')
                outimg = im2uint16(inimg);
            elseif strcmp(obj.EvaluationType,'int16')
                outimg = im2int16(inimg);
            elseif  strcmp(obj.EvaluationType,'single')
                outimg = im2single(inimg);
            elseif  strcmp(obj.EvaluationType,'double')
                outimg = im2double(inimg);
            else
                outimg = inimg;
            end
        end
        
        function value = mse_(obj,resImg)
            srcImg = obj.SourceImage;
%             value = sum((double(srcImg(:))-double(resImg(:))).^2)...
%                 /numel(srcImg);
            value = sum(abs(double(srcImg(:))-double(resImg(:))).^2);
        end
        
        function value = psnr_(obj,resImg)
            srcImg = obj.SourceImage;
            if verLessThan('images','9.0') || ...
                    ~obj.IsConversionToEvaluationType 
                value = 10*log10(abs(obj.PeakValue)^2/mse_(obj,resImg));
            else
                value = psnr(srcImg,resImg);
            end
        end
        
        function value = ssim_(obj,resImg)
            srcImg = obj.SourceImage;
            if verLessThan('images','9.0');
                value = ssim_index(resImg,srcImg);
            else
                value = ssim(resImg,srcImg);
            end
        end
        
    end
    
end
