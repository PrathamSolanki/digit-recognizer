%-----------------Initialization-------------------------------------------
clear ; close all; clc; history -c

%-------------------Loading Data-------------------------------------------
fprintf("\n\nLoading and Pre-Processing data....\n\n");
train_data = csvread('../../../Data/train.csv');
test_data = csvread('../../../Data/test.csv');

%------------------Setting up the parameters--------------------------------------
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
%m = size(train_data,1);
y = train_data(:,1);

%-------------------Formatting Data-------------------------------------------
train_data(:,1) = [];
train_data(1,:) = [];
test_data(1,:) = [];
y(1,:) = [];
X = train_data;
m = size(X,1);

%-----------------Training Logistic Regression-----------------------------
fprintf("\n\nTraining Logistic Regression....\n\n");
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

%----------------Predicting------------------------------------------------
fprintf("\n\nMaking Predictions on the Training Set....\n\n");
pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%----------------Predicting on test data-----------------------------------
fprintf("\n\nMaking Predictions on the Test Set....\n\n");
pred = predictOneVsAll(all_theta, test_data);

%-----------------Writing output to file-----------------------------------
fprintf("\n\nWriting Predictions to file (submission.csv)....\n\n");
ImageId = 1:1:size(pred);
output = [ImageId' pred];
csvwrite ('submission.csv', output);