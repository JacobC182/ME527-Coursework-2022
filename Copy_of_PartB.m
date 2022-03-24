%Jacob Currie - 201718558 - ME527 Coursework
%Part B: Solution - Surrogate Approach
%---------------INITAL SETUP-----------------------------------------------
tic; %start timer
clear all %#ok<CLALL>     %clearing
close all; clc;
addpath('.\dace'); %Importing DACE library
rng('default'); rng(12345); %make results repeatable but still stochastic
colors = [255 0 0; 255 100 0; 255 170 0; 168 180 0; 60 255 0; 0 255 200; 0 130 255; 0 0 255; 160 0 255; 255 0 120];
dots = ['.','*','+','x','s','d','^','v','p','h']; %Plotting
%Bounds
lb = zeros(1,6); %true lower bounds
ub = [10, 50, 200, 1000, 5000, 20000]; %true upper bounds
%Function Handle
Fn = @(x) AuxModel(x.*ub);  %INPUT SCALED [0-1], OUTPUT NOT SCALED
%Kriging Surrogate variables
lb_K = lb + 1e-5; %lower bounds with "close to zero" tolerance for surrogate
ub_K = ones(1,6); %upper bounds fixed for input scaling
theta = [1, 1, 1, 1, 1, 1]; %Initial Theta per-variable values for kriging
%---------------CREATING SURROGATE-----------------------------------------
%number of training samples for initial surrogate
n = 12;
xTrain1 = lhsdesign(n, 6, 'iterations',5); xTrain2 = xTrain1; %Sampling using latin hypercube (initial surrogate)
yTrain = zeros(n, 2);

for i = 1:n; yTrain(i,:) = Fn(xTrain1(i,:)); end %Getting solutions for LHS samples
yTrain1 = yTrain(:,1); yTrain2 = yTrain(:,2);
%Creating first surrogate
[dmodel1, perf] = dacefit(xTrain1,yTrain1, @regpoly1, @corrgauss, theta, lb_K, ub_K);
[dmodel2, perf] = dacefit(xTrain2,yTrain2, @regpoly1, @corrgauss, theta, lb_K, ub_K);
%---------------OPTIMISING SURROGATE - GA----------------------------------
FnS1 = @(x) predictor(x, dmodel1); %Surrogate function handle
FnS2 = @(x) predictor(x, dmodel2); %Surrogate function handle
p = 20; g = 200; r = floor((300 - n)/p); %Surrogate Optimisation options - p=population, g=generations, r=number of iterations for surrogate improvement
optionsGA = optimoptions('gamultiobj', 'TolFun',1e-4, 'PopulationSize',p, 'MaxGenerations',g, 'Display','off');

for i = 1:r %Iteratively improving surrogate
    [x1, f1, eFlag, outInfo, popFinal1, popScore] = gamultiobj(FnS1, 6, [], [], [], [], lb_K, ub_K, optionsGA);
    [x2, f2, eFlag, outInfo, popFinal2, popScore] = gamultiobj(FnS2, 6, [], [], [], [], lb_K, ub_K, optionsGA);%GA to find optimal front of surrogate

    popFinal1 = unique(popFinal1, "rows"); %Removing duplicates from pareto points - DACEFIT does not like duplicate points!!
    popFinal2 = unique(popFinal2, "rows");

    xTrain1 = cat(1, xTrain1, popFinal1); 
    xTrain2 = cat(1, xTrain2, popFinal2); 

    for j = 1:size(popFinal1, 1)%Adding new pareto points to list of surrogate training samples (line above also)
        tt = Fn(popFinal1(j,:));
        yTrain1 = cat(1, yTrain1, tt(1));
    end

    for j = 1:size(popFinal2, 1)%Adding new pareto points to list of surrogate training samples (line above also)
        tt = Fn(popFinal2(j,:));
        yTrain2 = cat(1, yTrain2, tt(2));
    end

    [dmodel1, perf] = dacefit(xTrain1,yTrain1, @regpoly1, @corrgauss, theta, lb_K, ub_K); %Creating new surrogate with updated training points
    [dmodel2, perf] = dacefit(xTrain2,yTrain2, @regpoly1, @corrgauss, theta, lb_K, ub_K);
end
fEvalTotal = n + (r*p); %Calculating total function evaluations - does NOT take into account reduction due to line 35!!
disp("Total Problem Function Evaluations: " + num2str(fEvalTotal));
%---------------OPTIMISATION USING SURROGATE - GA--------------------------
FnS_Final = @(x) [FnS1(x), FnS2(x)];

maxEval = 6000; %Genetic algorithm options for final optimisation
popsize = 200;
maxgen = round(maxEval/popsize);
ftol = 1e-5;
optionsGA = optimoptions('gamultiobj', 'TolFun',ftol, 'PopulationSize',popsize, 'MaxGenerations',maxgen, 'Display','off');%, 'PlotFcn',@gaplotpareto);

figure(1); xlabel('F-1');ylabel('F-2');title('10x Pareto Fronts'); hold on; %Creating final figure for pareto fronts
%GA optimisation routine - repeating 10 times
for i = 1:10
    [x, f, eFlag, outInfo, popFinal, popScore] = gamultiobj(FnS_Final, 6, [], [], [], [], lb_K, ub_K, optionsGA); %Finding pareto front with GA
    figure(1);
    scatter(f(:,1),f(:,2),35,colors(i,:)./255, dots(i)); %Plotting front
end
disp('PART B - SURROGATE OPTIMISATION ROUTINE FINISHED'); toc; %stop timer