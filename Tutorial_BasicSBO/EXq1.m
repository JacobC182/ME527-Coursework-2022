function [f,g]=EXq1(x)
global fv
global gv
f=[];g=[];
%x(1)=(0:.1:2*pi);
%x(2)=(0:.1:20);
f=(x(1)-3).^2+3*(x(2)-1).^2+2; 
if nargout > 1 % gradient required
    g = [2*(x(1)-3);
         6*(x(2)-1)];
end

fv=[fv;f x];

% gv=[gv;f x g'];