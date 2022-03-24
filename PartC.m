%Jacob Currie - 201718558 - ME527 Coursework
%Part C: Solution - Surrogate Approach - EXPENSIVE FUNCTION
%---------------PARALLEL POOL SETUP----------------------------------------
tic %start timer
nProcess = 10; %number of process workers to instantiate for parpool - with current settings only 10 are needed for maximum speed ~2.5hours
parpool('local', nProcess);
%---------------INITAL SETUP-----------------------------------------------
clear all %#ok<CLALL>     %clearing
close all; clc;
addpath('.\dace'); %Importing DACE library
rng('default'); rng(12345); %make results repeatable but still stochastic
%Bounds
lb = zeros(1,6); %true lower bounds
ub = [10, 50, 200, 1000, 5000, 20000]; %true upper bounds
%Function Handle
Fn = @(x) ExpModel(x.*ub);  %INPUT SCALED [0-1], OUTPUT NOT SCALED
%Kriging Surrogate variables
lb_K = lb + 1e-5; %lower bounds with "close to zero" tolerance for surrogate
ub_K = ones(1,6); %upper bounds fixed for input scaling
theta = [1, 1, 1, 1, 1, 1]; %Initial Theta per-variable values
%---------------CREATING SURROGATE-----------------------------------------
%number of training samples for initial surrogate
n = 10;
xTrain = lhsdesign(n, 6, 'iterations',5); %Sampling using latin hypercube (initial surrogate)
yTrain = zeros(n, 2);
parfor i = 1:n; yTrain(i,:) = Fn(xTrain(i,:)); end %Getting EXPENSIVE solutions for LHS samples
%Creating first surrogate
[dmodel, perf] = dacefit(xTrain,yTrain, @regpoly1, @corrspline, theta, lb_K, ub_K);
%---------------OPTIMISING SURROGATE - GA----------------------------------
FnS = @(x) predictor(x, dmodel); %Surrogate function handle
p = 10; g = 2000; r = floor((300 - n)/p); %Surrogate Optimisation options - p=population, g=generations, r=number of iterations for surrogate improvement
optionsGA = optimoptions('gamultiobj', 'TolFun',1e-4, 'PopulationSize',p, 'MaxGenerations',g, 'Display','off');

for i = 1:r %Iteratively improving surrogate
    [~,~,~,~,popFinal,~] = gamultiobj(FnS, 6, [], [], [], [], lb_K, ub_K, optionsGA); %GA to find optimal front of surrogate
    popFinal = unique(popFinal, "rows"); %Removing duplicates from pareto points - DACEFIT does not like duplicate points!!
    xTrain = cat(1, xTrain, popFinal);  %Adding new points to surrogate training points list
    yTrainNew = zeros(size(popFinal, 1), 2);%Placeholder variable for parallel
    parfor j = 1:size(popFinal, 1)%Sampling new points found from surrogate on expensive function
        yTrainNew(j,:) = Fn(popFinal(j,:));
    end
    yTrain = cat(1, yTrain, yTrainNew); %Adding new points to surrogate training points list
    [dmodel, perf] = dacefit(xTrain,yTrain, @regpoly1, @corrspline, theta, lb_K, ub_K); %Creating new surrogate with updated training points
end
fEvalTotal = n + (r*p); %Calculating total function evaluations - does NOT take into account reduction due to line 38!!
disp("Total Problem Function Evaluations: " + num2str(fEvalTotal));
%---------------OPTIMISATION USING SURROGATE - GA--------------------------
maxEval = 3000; %Genetic algorithm options for final optimisation
popsize = 150;
maxgen = round(maxEval/popsize);
ftol = 1e-5;
optionsGA = optimoptions('gamultiobj', 'TolFun',ftol, 'PopulationSize',popsize, 'MaxGenerations',maxgen, 'Display','off');%, 'PlotFcn',@gaplotpareto);
%GA optimisation routine
[x, f, eFlag, outInfo, popFinal, popScore] = gamultiobj(FnS, 6, [], [], [], [], lb_K, ub_K, optionsGA); %Finding pareto front with GA
figure(1); xlabel('F-1');ylabel('F-2');title('Pareto Fronts'); %Creating final figure for pareto fronts
scatter(f(:,1),f(:,2),35); %Plotting front
fTest = zeros(size(f, 1), 2);
parfor i = 1:size(f, 1)
    fTest(i, :) = Fn(x(i, :)); %Sampling expensive function at solutions found from true front
end
hold on; scatter(fTest(:,1), f(:,2),35); legend('Surrogate', 'Expensive Function'); %Plotting true front
disp('PART C [EXPENSIVE FUNCTION] - SURROGATE OPTIMISATION ROUTINE FINISHED'); toc; %stop timer