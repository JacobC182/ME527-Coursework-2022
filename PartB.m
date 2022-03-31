%Jacob Currie - 201718558 - ME527 Coursework
%Part B: Solution - Surrogate Approach
%----
%THIS SCRIPT IS SCULPTED FROM:
%Dr. Edmondo Minisci's "Simplified Kriging Surrogate Optimisation"
%Example script - available from the class resources on MyPlace
%And its supporting functions "update_database.m"
%----
%This script also uses the DACE library - for building surrogates
%"DACE - A Matlab kriging toolbox" - https://omicron.dk/dace.html
%----

tic;
rng('default') %set random seed for repeatable results
rng(123)

colours = [255 0 0; 255 100 0; 255 170 0; 168 180 0; 60 255 0; 0 255 200; 0 130 255; 0 0 255; 160 0 255; 255 0 120];
dots = ['.','*','+','x','s','d','^','v','p','h']; %Plotting

ndim=6; ub = [10, 50, 200, 1000, 5000, 20000]; %no. of variables and true upper bounds for scaling

FITNESSFCN=@(x) [fg_1(x);fg_2(x)]; %Combining F1 and F2 Surrogates for multiobjective function

outputF = {};
outputX = outputF;
outputFeval = [];

nRuns = 10;

for parloop = 1:nRuns
 %start timer
clearvars -except ndim ub outputF outputX parloop colours dots FITNESSFCN nRuns outputFeval %initial setup - clearing
close all
clc

TRUE_F=@(x) AuxModel(x .*ub); %real function handle

theta = [2 2 2 2 2 2]; %INITIAL theta guess
lob = ones(ndim, 1) .* 1e-6; upb = ones(ndim, 1) .* 30; %Kriging initial values/bounds
MaxEval = 300; %MAXIMUM NUMBER OF TRUE FUNCTION EVALUATIONS ALLOWED
%% DOE
nn=18; %number of initial sample points
xKept=lhsdesign(nn,ndim,'criterion','maximin','iterations',30); % Generate initial sample points with latin hypercube

%% Evaluation of the two objectives
yKept1=[];
yKept2=[];
for i=1:nn %sampling initial points on true function
    [f]=TRUE_F(xKept(i,:));
    yKept1=[yKept1; f(1)];
    yKept2=[yKept2; f(2)];
end

addpath('.\dace') %Importing DACE library

[dmodel1, perf1] = dacefit(xKept,yKept1, @regpoly0, @corrgauss, theta, lob, upb); %Creating initial Kriging model for Objective 1
[dmodel2, perf2] = dacefit(xKept,yKept2, @regpoly0, @corrgauss, theta, lob, upb); %Creatinf for Objective 2
dmodel10=dmodel1;
dmodel20=dmodel2;

%% Definition of the surrogate model that should be passed to optimiser
fg_1=@(x) predictor(x,dmodel1);
fg_2=@(x) predictor(x,dmodel2);
FITNESSFCN=@(x) [fg_1(x);fg_2(x)]; %Combining F1 and F2 Surrogates for multiobjective function

ng=0;
ngmax=0;

while size(yKept1,1)< (MaxEval - nn) && ng<430
    %% Setting of the GA options
    PopSize=120;
    X0=lhsdesign(PopSize,ndim,'criterion','maximin','iterations',30);
    naddMax=3;   
    naddZ=0; %Resetting "zero-add" counter - CONVERGENCE CONTROLLING VARIABLE - counts the number of times no new points have been added to the surrogate collection
    ngmax=ngmax+5; %incremement generations
    %% Surrogate based Optimisation loop
    while naddZ<=2 && size(yKept1,1)< (MaxEval - nn)
        ng=ng+1; %'PlotFcn',@gaplotpareto
        optionsGA = optimoptions('gamultiobj','PopulationSize',PopSize,'MaxGenerations',ngmax,'InitialPopulationMatrix',X0,'Display','none','UseParallel',false);
        [X,FVAL,EXITFLAG,OUTPUT,POPULATION,SCORE] = gamultiobj(FITNESSFCN,ndim,[],[],[],[],zeros(1,ndim),ones(1,ndim),[],optionsGA);
        X0=POPULATION;
        update_database %CALLING update database routine to evaluate true function points and add to surrogate training sample collection
        
        if nadd>0 %Create new surrogates if there are valuable points to add to the surrogate fitting
            theta1 = dmodel1.theta;
            theta2 = dmodel2.theta;
            [dmodel1, perf1] = dacefit(xKept,yKept1, @regpoly0, @corrgauss, theta1, lob, upb);
            [dmodel2, perf2] = dacefit(xKept,yKept2, @regpoly0, @corrgauss, theta2, lob, upb);
             
            fg_1=@(x) predictor(x,dmodel1);
            fg_2=@(x) predictor(x,dmodel2);
            FITNESSFCN=@(x) [fg_1(x);fg_2(x)];
        end
        if nadd==0 %Increment zero-add counter variable of zero points added to surrogate
            naddZ=naddZ+1;
        else
            naddZ=0;
        end
        disp([size(yKept1, 1) nadd naddZ ng ngmax parloop])
    end    
    
end

%store outputs from 1-10 run    
outputF{parloop} = FVAL;
outputX{parloop} = X;
outputFeval(parloop) = size(yKept1, 1);

end %end for

close all;
figure(1); xlabel('F-1');ylabel('F-2');title('10x Pareto Fronts'); hold on; %Creating final figure for pareto fronts
toc; %stop timer

for i = 1:nRuns
    figure(1); scatter(outputF{i}(:,1),outputF{i}(:,2),35,colours(i,:)./255, dots(i)); %Plotting front
end

disp('PART B - SURROGATE OPTIMISATION ROUTINE FINISHED');
