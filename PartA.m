%Jacob Currie - 201718558 - ME527 Coursework
%Part A: Solution - Non-surrogate Approach - Genetic Algorithm
tic; %start timer
clear all %#ok<CLALL>     %clearing
close all; clc;
rng('default'); rng(1); %make results repeatable but still stochastic
colours = [255 0 0; 255 100 0; 255 170 0; 168 180 0; 60 255 0; 0 255 200; 0 130 255; 0 0 255; 160 0 255; 255 0 120];
dots = ['.','*','+','x','s','d','^','v','p','h']; %Plotting
maxEval = 50000; %Maximum allowable function evaluations
%Bounds
lb = zeros(1,6); ub = ones(1,6);
%Function Handle
ub_Scale = [10, 50, 200, 1000, 5000, 20000]; %true upper bounds for scaling
Fn = @(x) AuxModel(x .*ub_Scale);

popsize = 50; %GA optimisation options
maxgen = round(maxEval/popsize); ftol = 1e-5;
optionsGA = optimoptions('gamultiobj', 'TolFun',ftol, 'PopulationSize',popsize, 'MaxGenerations',maxgen, 'PlotFcn',@gaplotpareto, 'Display','off');

figure(1); xlabel('F-1');ylabel('F-2');title('10x Pareto Fronts'); hold on; %Creating final figure for pareto fronts
%GA optimisation routine - repeating 10 times
for i = 1:10
    [x, f, eFlag, outInfo, popFinal, popScore] = gamultiobj(Fn, 6, [], [], [], [], lb, ub, optionsGA); %Finding pareto front with GA
    figure(1); scatter(f(:,1),f(:,2),35,colours(i,:)./255, dots(i)); %Plotting front
end
disp('PART A - NON-SURROGATE OPTIMISATION ROUTINE FINISHED'); toc; %stop timer