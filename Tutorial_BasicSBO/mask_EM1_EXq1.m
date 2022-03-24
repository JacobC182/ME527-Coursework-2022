function FF=mask_EM1_EXq1(xa,name,ndim)
ll=[0    0   
    2*pi 20 ];
x=xa.*(ll(2,:)-ll(1,:))+ll(1,:);
[f1]=EXq1(x);
[f2]=EM1(x);
FF=[f1 f2];

