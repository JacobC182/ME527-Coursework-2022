function [f,g]=EM1(x)
global fv
%x(1)=(0:.1:2*pi);
%x(2)=(0:.1:20);
f=[];g=[];
f=(x(2)-3.*x(1)).*sin(x(2))+(x(1)-2).^2; 
if nargout > 1 % gradient required
    g = [2*x(1)-3*sin(x(2))-4;
        (x(2)-3.*x(1)).*cos(x(2))+sin(x(2))];
end

fv=[fv;f x];
