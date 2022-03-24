clear all
close all
clc

rng('default')
rng(220)

ndim=2;
TRUE_F=@(x) mask_EM1_EXq1(x,0);

%% Optimisation wi
optionsGA = optimoptions('gamultiobj','PopulationSize',500,'PlotFcn',@gaplotpareto,'MaxGenerations',500,'Display','none');
[X,FVAL,EXITFLAG,OUTPUT,POPULATION,SCORE] = gamultiobj(TRUE_F,ndim,[],[],[],[],zeros(1,ndim),ones(1,ndim),[],optionsGA);

close all

%% DOE
nn=7;
xKept=lhsdesign(nn,ndim,'criterion','maximin','iterations',30);


hold on
plot(xKept(:,1)*2*pi,xKept(:,2)*20,'o');

%% Evaluation of the two objectives
yKept1=[];
yKept2=[];
for i=1:nn
    [f]=TRUE_F(xKept(i,:));
    yKept1=[yKept1; f(1)];
    yKept2=[yKept2; f(2)];
end



addpath('.\dace')
theta = [10 10]; lob = [1e-1 1e-1]; upb = [25 25];
[dmodel1, perf1] = dacefit(xKept,yKept1, @regpoly0, @corrgauss, theta, lob, upb);
[dmodel2, perf2] = dacefit(xKept,yKept2, @regpoly0, @corrgauss, theta, lob, upb);
dmodel10=dmodel1;
dmodel20=dmodel2;

%% Definition of the surrogate model that should be passed to optimiser
fg_1=@(x) predictor(x,dmodel1);
fg_2=@(x) predictor(x,dmodel2);
FITNESSFCN=@(x) [fg_1(x);fg_2(x)];


D=(0:.1:2*pi);
U=(0:.1:20);
[Xr,Yr] = meshgrid(D./2/pi,U./20);                                
for i=1:size(Xr,1)
    for j=1:size(Xr,2)
        FF=FITNESSFCN([Xr(i,j),Yr(i,j)]);
        yEM1(i,j)=FF(1);
        yEXq1(i,j)=FF(2);
    end
end
figure(5)
contour(Xr*2*pi,Yr*20,yEM1)
hold on
plot(xKept(:,1)*2*pi,xKept(:,2)*20,'o');
grid;title('Kriging')
figure(6)
contour(Xr*2*pi,Yr*20,yEXq1)
hold on
plot(xKept(:,1)*2*pi,xKept(:,2)*20,'o');
grid;title('Kriging')
ng=0;
ngmax=0;

while size(yKept1,1)<600 && ng<200
    %% Setting of the GA options
    PopSize=5000;
    X0=lhsdesign(PopSize,ndim,'criterion','maximin','iterations',30);
    optionsGA = optimoptions('gamultiobj','PopulationSize',PopSize,'PlotFcn',@gaplotpareto,'InitialPopulationMatrix',X0);
    naddMax=6;
    naddZ=0;
    ngmax=ngmax+5;
    %% Surrogate based Optimisation loop
    while naddZ<=2 && size(yKept1,1)<600
        ng=ng+1;
        optionsGA = optimoptions('gamultiobj','PopulationSize',PopSize,'PlotFcn',@gaplotpareto,'MaxGenerations',ngmax,'InitialPopulationMatrix',X0,'Display','none');
        [X,FVAL,EXITFLAG,OUTPUT,POPULATION,SCORE] = gamultiobj(FITNESSFCN,ndim,[],[],[],[],zeros(1,ndim),ones(1,ndim),[],optionsGA);
        X0=POPULATION;
        update_database
        
        if nadd>0
            [dmodel1, perf1] = dacefit(xKept,yKept1, @regpoly0, @corrgauss, theta, lob, upb);
            [dmodel2, perf2] = dacefit(xKept,yKept2, @regpoly0, @corrgauss, theta, lob, upb);
            
            
            fg_1=@(x) predictor(x,dmodel1);
            fg_2=@(x) predictor(x,dmodel2);
            FITNESSFCN=@(x) [fg_1(x);fg_2(x)];
        end
        if nadd==0
            naddZ=naddZ+1;
        else
            naddZ=0;
        end
        disp([size(yKept1) nadd naddZ])
    end    
    
end
