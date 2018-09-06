classdef VolumetricDataVisualizerTestCase < matlab.unittest.TestCase
    %VOLUMETRICDATAVISUALIZERTESTCASE Test cases for VolumetricDataVisualizer
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2018, Shogo MURAMATSU
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
        TestFigure
    end
    
    properties (TestParameter)
        dim1 = struct('even',64,'odd',65);
        dim2 = struct('even',64,'odd',65);
        dim3 = struct('even',64,'odd',65);
        slice  = { 'XY', 'YZ' };
        vrange = struct('negpos',[-1 1],'nonneg',[0 1]);
        scale = struct('half',0.5,'one',1,'two',2);
    end
    
    methods(TestMethodSetup)
        function createFigure(testCase)
            % comment
            testCase.TestFigure = figure;
        end
    end
    
    methods(TestMethodTeardown)
        function closeFigure(testCase)
            close(testCase.TestFigure)
        end
    end
    
    methods (Test)
        
        function testConstruction(testCase)
            
            % Expected
            expctdSlisePlane = 'XY';
            
            % Instantiation
            import saivdr.utility.*
            target = VolumetricDataVisualizer();
            
            % Actual 
            actualSlicePlane = target.SlicePlane;
            
            % Evaluation
            testCase.verifyEqual(actualSlicePlane,expctdSlisePlane)
            
        end
        
        function testStep(testCase,dim1,dim2,dim3)
            
            % Parameters
            depth   = dim1; % Depth
            height  = dim2; % Height
            width   = dim3; % Width
            phtm = phantom('Modified Shepp-Logan',min(height,depth));
            if depth > height
                phtm = padarray(phtm,[0 (depth-height)],0,'post');
            elseif height > depth
                phtm = padarray(phtm,[(height-depth) 0],0,'post');
            end
            testCase.verifySize(phtm,[height depth]);
            sliceYZ = permute(phtm,[1 3 2]);
            uSrc = 0.5*repmat(sliceYZ,[1 width 1]) + 1;
            
            % Expected
            vmin = 0;
            vmax = 1;
            scale_ = 1;
            expctdCData = uSrc(:,:,round(depth/2));
            expctdCData = (expctdCData-vmin)/(vmax-vmin);
            if vmin < 0
                expctdCData = scale_*(expctdCData - 0.5)+0.5;
            else
                expctdCData = scale_*expctdCData;
            end
            %expctdCData(expctdCData<0) = 0;
            %expctdCData(expctdCData>1) = 1;            
            
            % Instantiation
            import saivdr.utility.*            
            hImg = imshow(zeros(size(expctdCData)));
            target = VolumetricDataVisualizer('ImageObject',hImg);
            
            % Actual 
            target.step(uSrc);
            actualCData = hImg.CData;
            actualXLim  = hImg.Parent.XLim;
            actualYLim  = hImg.Parent.YLim;
            
            % Evaluation
            testCase.verifySize(actualCData,size(expctdCData));
            testCase.verifyEqual(actualXLim(2)-actualXLim(1),size(expctdCData,2));
            testCase.verifyEqual(actualYLim(2)-actualYLim(1),size(expctdCData,1));
            testCase.verifyEqual(actualCData,expctdCData,'AbsTol',1e-6);
        end
        
        function testStepSlicePlane(testCase,dim1,dim2,dim3,slice)
            
            % Parameters
            depth   = dim1; % Depth
            height  = dim2; % Height
            width   = dim3; % Width
            phtm = phantom('Modified Shepp-Logan',min(height,depth));
            if depth > height
                phtm = padarray(phtm,[0 (depth-height)],0,'post');
            elseif height > depth
                phtm = padarray(phtm,[(height-depth) 0],0,'post');
            end
            testCase.verifySize(phtm,[height depth]);
            sliceYZ = permute(phtm,[1 3 2]);
            uSrc = 0.5*repmat(sliceYZ,[1 width 1]) + 1;
            
            % Expected
            vmin = 0;
            vmax = 1;
            scale_ = 1;
            expctdSlicePlane = slice;
            if strcmp(expctdSlicePlane,'XY')
                expctdCData = uSrc(:,:,round(depth/2));
            else
                expctdCData = squeeze(uSrc(:,round(width/2),:));
            end
            expctdCData = (expctdCData-vmin)/(vmax-vmin);
            if vmin < 0
                expctdCData = scale_*(expctdCData - 0.5)+0.5;
            else
                expctdCData = scale_*expctdCData;
            end
            %expctdCData(expctdCData<0) = 0;
            %expctdCData(expctdCData>1) = 1;
            
            % Instantiation
            import saivdr.utility.*            
            hImg = imshow(zeros(size(expctdCData)));
            target = VolumetricDataVisualizer(...
                'ImageObject',hImg,...
                'SlicePlane',slice);
            
            % Actual 
            target.step(uSrc);
            actualCData = hImg.CData;
            actualSlicePlane = target.SlicePlane;
            actualXLim  = hImg.Parent.XLim;
            actualYLim  = hImg.Parent.YLim;
            
            % Evaluation
            testCase.verifyEqual(actualSlicePlane,expctdSlicePlane);
            testCase.verifySize(actualCData,size(expctdCData));
            testCase.verifyEqual(actualXLim(2)-actualXLim(1),size(expctdCData,2));
            testCase.verifyEqual(actualYLim(2)-actualYLim(1),size(expctdCData,1));
            testCase.verifyEqual(actualCData,expctdCData,'AbsTol',1e-6);
        end
        
         function testStepVRangeScale(testCase,slice,vrange,scale)
            
            % Parameters
            depth   = 64; % Depth
            height  = 64; % Height
            width   = 64; % Width
            phtm = phantom('Modified Shepp-Logan',min(height,depth));
            testCase.verifySize(phtm,[height depth]);
            sliceYZ = permute(phtm,[1 3 2]);
            uSrc = 0.5*repmat(sliceYZ,[1 width 1]) + 1;
            
            % Expected
            vmin = vrange(1);
            vmax = vrange(2);
            expctdSlicePlane = slice;
            if strcmp(expctdSlicePlane,'XY')
                expctdCData = uSrc(:,:,round(depth/2));
            else
                expctdCData = squeeze(uSrc(:,round(width/2),:));
            end
            expctdCData = (expctdCData-vmin)/(vmax-vmin);
            if vmin < 0
                expctdCData = scale*(expctdCData - 0.5)+0.5;
            else
                expctdCData = scale*expctdCData;
            end
            %expctdCData(expctdCData<0) = 0;
            %expctdCData(expctdCData>1) = 1;
            
            % Instantiation
            import saivdr.utility.*            
            target = VolumetricDataVisualizer(...
                'SlicePlane',slice,...
                'VRange',vrange,...
                'Scale',scale);
            
            % Actual 
            hImg = target.step(uSrc);
            actualCData = hImg.CData;
            actualSlicePlane = target.SlicePlane;
            actualXLim  = hImg.Parent.XLim;
            actualYLim  = hImg.Parent.YLim;            
            
            % Evaluation
            testCase.verifyEqual(actualSlicePlane,expctdSlicePlane);
            testCase.verifySize(actualCData,size(expctdCData));
            testCase.verifyEqual(actualXLim(2)-actualXLim(1),size(expctdCData,2));
            testCase.verifyEqual(actualYLim(2)-actualYLim(1),size(expctdCData,1));            
            testCase.verifyEqual(actualCData,expctdCData,'AbsTol',1e-6);
         end
        
         function testStepTexture3d(testCase)
            
            % Parameters
            texture = '3D';
            depth   = 16; % Depth
            height  = 16; % Height
            width   = 16; % Width
            phtm = phantom('Modified Shepp-Logan',min(height,depth));
            if depth > height
                phtm = padarray(phtm,[0 (depth-height)],0,'post');
            elseif height > depth
                phtm = padarray(phtm,[(height-depth) 0],0,'post');
            end
            testCase.verifySize(phtm,[height depth]);
            sliceYZ = permute(phtm,[1 3 2]);
            uSrc = 0.5*repmat(sliceYZ,[1 width 1]) + 1;
            
            % Expected
            %expctdCData = uSrc;
            %%{
            vmin = 0;
            vmax = 1;
            %expctdSlicePlane = slice;
            %if strcmp(expctdSlicePlane,'XY')
            %    expctdCData = uSrc(:,:,round(depth/2));
            %else
            %    expctdCData = squeeze(uSrc(:,round(width/2),:));
            %end
            expctdCData = (uSrc-vmin)/(vmax-vmin);
            %expctdCData(expctdCData<0) = 0;
            %expctdCData(expctdCData>1) = 1;
            %%}
            
            % Instantiation
            import saivdr.utility.*            
            target = VolumetricDataVisualizer(...
                'Texture',texture);
            
            % íl
            hVol = target.step(uSrc);
            actualCData = getappdata(hVol,'CData');
            
            % åüèÿ
            testCase.verifySize(actualCData,size(expctdCData));
            testCase.verifyEqual(actualCData,expctdCData,'AbsTol',1e-6);
            
         end
         
    end
end
