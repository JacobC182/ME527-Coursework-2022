function mse = MSE(predicted, true)
%Compute MSE (Mean Square Error) between predicted and true list
if length(predicted) ==  length(true)
    mse = 0;
    for i = 1:length(predicted)
        mse = mse + (true(i) - predicted(i))^2;
    end
    mse = mse .* (1/length(predicted));
else
    mse = nan;
end