function label = SemismoothNewtonPredict(model, data)
%% Predict on data by the trained L2-loss L2-regularized linear SVM model
% last modified by Juan Yin and Qingna Li 2020/08/09 
% If you have comments, please contact qnl@bit.edu.cn

%Input:
% model: the trained model, including weight vector and cost parameter
% data: d-by-m data matrix 
%         (m is number of data points and d is dimension of one data point)
%
%Output:
% label: m-by-1 predicted labels vector for the data
%         (each label is either -1 or 1)

[~ ,m] = size(data);
data = [data ;ones(1,m)];
y_predict = data'*model.weight;
label = sign(y_predict);