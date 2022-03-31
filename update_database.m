nadd=0;
    try
        [min1,imin1]=min(FVAL(:,1)); %Find minimum position of F1
        [min2,imin2]=min(FVAL(:,2)); %Find minimum position of F2
        
        %%CHECKING OF F1-------------------------
        xcheck=X(imin1,:); %Get X solution of F1 minimum
        DV=[];
        
        for id=1:size(xKept,1)
            DV(id)=norm(xKept(id,:)-xcheck,2); %Calculate magnitude difference between minX1 and all other X1
        end
        if min(DV)>0.1
            xKept=[xKept; xcheck];
            [f]=TRUE_F(xKept(end,:));
            yKept1=[yKept1; f(1)];
            yKept2=[yKept2; f(2)];
            nadd=nadd+1;
        end
        
        %%CHECKING OF F2-------------------------
        xcheck=X(imin2,:);
        DV=[];

        for id=1:size(xKept,1)
            DV(id)=norm(xKept(id,:)-xcheck,2); %Calculate magnitude difference between minX1 and all other X2
        end
        if min(DV)>0.1
            xKept=[xKept; xcheck];
            [f]=TRUE_F(xKept(end,:));
            yKept1=[yKept1; f(1)];
            yKept2=[yKept2; f(2)];
            nadd=nadd+1;
        end
    catch
        
    end
    
    try %CHECKING OF NEW FOUND FRONT
        for ifront=1:size(X,1)
            xcheck=X(ifront,:);
            DV=[];
            for id=1:size(xKept,1)
                DV(id)=norm(xKept(id,:)-xcheck,2);
            end
            if min(DV)>0.1
                xKept=[xKept; xcheck];
                [f]=TRUE_F(xKept(end,:));
                yKept1=[yKept1; f(1)];
                yKept2=[yKept2; f(2)];
                nadd=nadd+1;
            end
            if nadd>=naddMax
                break
            end
        end
    catch
        
    end

    if nadd<3 %CHECKING EVERY POPULATION MEMBER
        RN=randperm(size(POPULATION,1));
        for ifront=1:size(POPULATION,1)
            xcheck=POPULATION(RN(ifront),:);
            DV=[];
            for id=1:size(xKept,1)
                DV(id)=norm(xKept(id,:)-xcheck,2);
            end
            if min(DV)>0.1
                xKept=[xKept; xcheck];
                [f]=TRUE_F(xKept(end,:));
                yKept1=[yKept1; f(1)];
                yKept2=[yKept2; f(2)];
                nadd=nadd+1;
            end
            if nadd>=naddMax
                break
            end
        end
    end