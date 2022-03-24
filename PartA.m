%Jacob Currie - 201718558 - ME527 Coursework
%Part A: Solution - Non-surrogate Approach - Genetic Algorithm
%---------------INITAL SETUP-----------------------------------------------
tic %start timer
clear all %#ok<CLALL>     %clearing
close all ;clc;
rng('default'); rng(12345); %make results repeatable but still stochastic
colors = [255 0 0; 255 100 0; 255 170 0; 168 180 0; 60 255 0; 0 255 200; 0 130 255; 0 0 255; 160 0 255; 255 0 120];
dots = ['.','*','+','x','s','d','^','v','p','h']; %Plotting
maxEval = 50000; %Maximum allowable function evaluations
%Function Handle
Fn = @(x) AuxModel(x);
%Bounds
lb = zeros(1,6); ub = [10, 50, 200, 1000, 5000, 20000]; %NOTE - scaling is not used

popsize = 150; %GA optimisation options
maxgen = round(maxEval/popsize);
ftol = 1e-5;
optionsGA = optimoptions('gamultiobj', 'TolFun',ftol, 'PopulationSize',popsize, 'MaxGenerations',maxgen, 'PlotFcn',@gaplotpareto, 'MutationFcn','mutationpower', 'Display','off');

figure(1); %Creating final figure for pareto fronts
xlabel('F-1');ylabel('F-2');title('10x Pareto Fronts'); hold on;
%GA optimisation routine - repeating 10 times
for i = 1:10
    [x, f, eFlag, outInfo, popFinal, popScore] = gamultiobj(Fn, 6, [], [], [], [], lb, ub, optionsGA); %Finding pareto front with GA
    figure(1);
    scatter(f(:,1),f(:,2),35,colors(i,:)./255, dots(i)); %Plotting front
end
disp('PART A - NON-SURROGATE OPTIMISATION ROUTINE FINISHED');
toc %stop timer