%Jacob Currie - 201718558 - ME527 Coursework
%Part B: Solution - Surrogate Approach
clear
close all
clc
rng('default');
rng(12345)
colors = [255 0 0; 255 100 0; 255 170 0; 168 180 0; 60 255 0; 0 255 200; 0 130 255; 0 0 255; 160 0 255; 255 0 120];
dots = ['.','*','+','x','s','d','^','v','p','h'];
%Bounds
lb = zeros(1,6);
ub = [10, 50, 200, 1000, 5000, 20000];
%Function Handle
Fn = @(x) AuxModel(x.*ub);

%number of training samples
n = 150;

xTrain = lhsdesign(n, 6, 'iterations',5);
fTrain = zeros(length(xTrain),2);
for i = 1:length(xTrain)
    fTrain(i,:) = Fn(xTrain(i,:));
end

hLayers = 3;

net = fitnet(hLayers, 'trainbfg');
net = train(net, xTrain', fTrain');

maxEval = 300-n;
popsize = 10;
maxgen = round(maxEval/popsize);
ftol = 1e-5;
optionsGA = optimoptions('gamultiobj', 'TolFun',ftol, 'PopulationSize',popsize, 'MaxGenerations',maxgen, 'PlotFcn',@gaplotpareto);

FnS = @(x) net(x');
hold off
figure(1);
xlabel('F-1');ylabel('F-2');title('10x Pareto Fronts');
hold on
%GA optimisation routine
for i = 1:10
    [x, f, eFlag, outInfo, popFinal, popScore] = gamultiobj(FnS, 6, [], [], [], [], lb, ub, optionsGA);
    figure(1);
    scatter(f(:,1),f(:,2),35,colors(i,:)./255, dots(i));
end

