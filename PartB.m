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
n = 10;
xTrain = lhsdesign(n, 6, 'iterations',5); %Sampling using latin hypercube (initial surrogate)
yTrain = zeros(n, 2);
for i = 1:n; yTrain(i,:) = Fn(xTrain(i,:)); end %Getting solutions for LHS samples
%Creating first surrogate
[dmodel, perf] = dacefit(xTrain,yTrain, @regpoly1, @corrspline, theta, lb_K, ub_K);
%---------------OPTIMISING SURROGATE - GA----------------------------------
FnS = @(x) predictor(x, dmodel); %Surrogate function handle
p = 10; g = 2000; r = floor((300 - n)/p); %Surrogate Optimisation options - p=population, g=generations, r=number of iterations for surrogate improvement
optionsGA = optimoptions('gamultiobj', 'TolFun',1e-4, 'PopulationSize',p, 'MaxGenerations',g, 'Display','off');

for i = 1:r %Iteratively improving surrogate
    [x, f, eFlag, outInfo, popFinal, popScore] = gamultiobj(FnS, 6, [], [], [], [], lb_K, ub_K, optionsGA); %GA to find optimal front of surrogate
    popFinal = unique(popFinal, "rows"); %Removing duplicates from pareto points - DACEFIT does not like duplicate points!!
    xTrain = cat(1, xTrain, popFinal); 
    for j = 1:size(popFinal, 1)%Adding new pareto points to list of surrogate training samples (line above also)
        yTrain = cat(1, yTrain, Fn(popFinal(j,:)));
    end
    [dmodel, perf] = dacefit(xTrain,yTrain, @regpoly1, @corrspline, theta, lb_K, ub_K); %Creating new surrogate with updated training points
end
fEvalTotal = n + (r*p); %Calculating total function evaluations - does NOT take into account reduction due to line 35!!
disp("Total Problem Function Evaluations: " + num2str(fEvalTotal));
%---------------OPTIMISATION USING SURROGATE - GA--------------------------
maxEval = 3000; %Genetic algorithm options for final optimisation
popsize = 150;
maxgen = round(maxEval/popsize);
ftol = 1e-5;
optionsGA = optimoptions('gamultiobj', 'TolFun',ftol, 'PopulationSize',popsize, 'MaxGenerations',maxgen, 'Display','off');%, 'PlotFcn',@gaplotpareto);

figure(1); xlabel('F-1');ylabel('F-2');title('10x Pareto Fronts'); hold on; %Creating final figure for pareto fronts
%GA optimisation routine - repeating 10 times
for i = 1:10
    [x, f, eFlag, outInfo, popFinal, popScore] = gamultiobj(FnS, 6, [], [], [], [], lb_K, ub_K, optionsGA); %Finding pareto front with GA
    figure(1);
    scatter(f(:,1),f(:,2),35,colors(i,:)./255, dots(i)); %Plotting front
end
disp('PART B - SURROGATE OPTIMISATION ROUTINE FINISHED'); toc; %stop timer