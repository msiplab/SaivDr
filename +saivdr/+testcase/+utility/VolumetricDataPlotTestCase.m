classdef VolumetricDataPlotTestCase < matlab.unittest.TestCase
    %PLGSOFTTHRESHOLDINGTESTCASE Test cases for VolumetricDataPlot
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
        dir  = { 'X', 'Y', 'Z' };
        scale1 = struct('half',0.5,'one',1,'two',2);
        scale2 = struct('half',0.5,'one',1,'two',2);        
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
            expctdDirection = 'Z';
            
            % Instantiation
            import saivdr.utility.*
            target = VolumetricDataPlot();
            
            % Actual 
            actualDirection = target.Direction;
            
            % Evaluation
            testCase.verifyEqual(actualDirection,expctdDirection)
            
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
            y = squeeze(uSrc(round(height/2),round(width/2),:));
            expctdYData = y(:).';
            expctdNumPlots = 1;
            
            % Instantiation
            import saivdr.utility.*            
            hPlot = plot(zeros(size(expctdYData)));
            target = VolumetricDataPlot('PlotObjects',{ hPlot });
            
            % Actual 
            hPlot = target.step(uSrc);
            actualYData = hPlot.YData;
            actualNumPlots = target.NumPlots;
            
            % Evaluation
            testCase.verifyEqual(actualNumPlots,expctdNumPlots);
            testCase.verifySize(actualYData,size(expctdYData));
            testCase.verifyEqual(actualYData,expctdYData,'AbsTol',1e-6);
        end
        
        function testStepDirection(testCase,dim1,dim2,dim3,dir)
            
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
            dirY = permute(phtm,[1 3 2]);
            uSrc = 0.5*repmat(dirY,[1 width 1]) + 1;
            
            % Expected
            expctdDirection = dir;
            if strcmp(expctdDirection,'X')
                y = squeeze(uSrc(round(height/2),:,round(depth/2)));
            elseif strcmp(expctdDirection,'Y')
                y = squeeze(uSrc(:,round(width/2),round(depth/2)));
            else
                y = squeeze(uSrc(round(height/2),round(width/2),:));
            end
            expctdYData = y(:).';
            expctdNumPlots = 1;
            
            % Instantiation
            import saivdr.utility.*            
            hPlot = plot(zeros(size(expctdYData)));
            target = VolumetricDataPlot(...
                'PlotObjects',{ hPlot },...
                'Direction',dir);
            
            % Actual 
            hPlot = target.step(uSrc);
            actualYData = hPlot.YData;
            actualDirection = target.Direction;
            actualNumPlots = target.NumPlots;
            
            % Evaluation
            testCase.verifyEqual(actualNumPlots,expctdNumPlots);
            testCase.verifyEqual(actualDirection,expctdDirection);
            testCase.verifySize(actualYData,size(expctdYData));
            testCase.verifyEqual(actualYData,expctdYData,'AbsTol',1e-6);
        end
        
        function testStepScales(testCase,dir,scale1)
            
            % Parameters
            depth   = 64; % Depth
            height  = 64; % Height
            width   = 64; % Width
            phtm = phantom('Modified Shepp-Logan',min(height,depth));
            testCase.verifySize(phtm,[height depth]);
            dirY = permute(phtm,[1 3 2]);
            uSrc = 0.5*repmat(dirY,[1 width 1]) + 1;
            
            % Expected
            expctdDirection = dir;
            if strcmp(expctdDirection,'X')
                y = squeeze(uSrc(round(height/2),:,round(depth/2)));
            elseif strcmp(expctdDirection,'Y')
                y = squeeze(uSrc(:,round(width/2),round(depth/2)));
            else
                y = squeeze(uSrc(round(height/2),round(width/2),:));
            end
            expctdYData = scale1*y(:).';
            
            % Instantiation
            import saivdr.utility.*            
            target = VolumetricDataPlot(...
                'Direction',dir,...
                'Scales',scale1);
            
            % Actual 
            hPlot = target.step(uSrc);
            actualYData = hPlot.YData;
            actualDirection = target.Direction;
            
            % Evaluation
            testCase.verifyEqual(actualDirection,expctdDirection);
            testCase.verifySize(actualYData,size(expctdYData));
            testCase.verifyEqual(actualYData,expctdYData,'AbsTol',1e-6);
        end
        
        function testStepDual(testCase,dim1,dim2,dim3,scale1,scale2)
            
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
            uSrc1 = 0.5*repmat(sliceYZ,[1 width 1]) + 1;
            uSrc2 = 0.5*repmat(sliceYZ,[1 width 1]);
            
            % Expected
            y1 = squeeze(uSrc1(round(height/2),round(width/2),:));
            expctdYData1 = scale1*y1(:).';
            y2 = squeeze(uSrc2(round(height/2),round(width/2),:));
            expctdYData2 = scale2*y2(:).';
            
            % Instantiation
            import saivdr.utility.*            
            target = VolumetricDataPlot(...
                'NumPlots',2,...
                'Scales',[scale1 scale2]);
            
            % Actual 
            [hPlot1,hPlot2] = target.step(uSrc1,uSrc2);
            actualYData1 = hPlot1.YData;
            actualYData2 = hPlot2.YData;
            
            % Evaluation
            testCase.verifySize(actualYData1,size(expctdYData1));
            testCase.verifySize(actualYData2,size(expctdYData2));            
            testCase.verifyEqual(actualYData1,expctdYData1,'AbsTol',1e-6);
            testCase.verifyEqual(actualYData2,expctdYData2,'AbsTol',1e-6);            
        end
        
    end
end
