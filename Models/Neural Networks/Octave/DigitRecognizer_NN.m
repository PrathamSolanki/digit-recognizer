%-----------------Initialization-------------------------------------------
clear ; close all; clc; history -c

%-------------------Loading Data-------------------------------------------
fprintf("\n\nLoading and Pre-Processing data....\n\n");
train_data = csvread('../../../Data/train.csv');
test_data = csvread('../../../Data/test.csv');

%-------------------Formatting Data-------------------------------------------
y = train_data(:,1);
train_data(:,1) = [];
train_data(1,:) = [];
test_data(1,:) = [];
y(1,:) = [];
X = train_data;
m = size(X,1);

%-------------------Setting the parameters------------------------------------
fprintf("\n\nSetting the Neural Network parameters....\n\n");
input_layer_size  = 784;  % 784 pixel values
hidden_layer1_size = 250;   % 250 hidden units
hidden_layer2_size = 250;   % 250 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% ================ Initializing Pameters ================
fprintf('\n\nInitializing the Neural Network Parameters ...\n\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
initial_Theta3 = randInitializeWeights(hidden_layer2_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

%% =================== Training NN ===================
fprintf('\n\nTraining Neural Network... \n\n')
options = optimset('MaxIter', 1000);
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1, Theta2 and Theta3 back from nn_params
%Theta1 = reshape(nn_params(1:19625),25,785);

%Theta2 = reshape(nn_params(19626:20275),25,26);

%Theta3 = reshape(nn_params(20276:20535),10,26);

temp1 = hidden_layer1_size * (input_layer_size + 1);
Theta1 = reshape(nn_params(1:temp1),hidden_layer1_size, (input_layer_size + 1));

temp2 = hidden_layer1_size*(hidden_layer1_size+1);
Theta2 = reshape(nn_params((temp1+1):(temp1+temp2)),hidden_layer1_size,hidden_layer1_size+1);

temp3 = (num_labels*hidden_layer1_size+1);
Theta3 = reshape(nn_params((temp1+temp2+1):end),num_labels,hidden_layer1_size+1);

%% ================= Implement Predict =================
fprintf("\n\nMaking Predictions on the Training Set....\n\n");
pred = predict(Theta1, Theta2, Theta3, X);

for i = 1:m,
  if (pred(i) == 10)
  	pred(i) = 0;
  endif
end

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% ================= Predict Test Data =================
fprintf("\n\nMaking Predictions on the Test Set....\n\n");
pred = predict(Theta1, Theta2, Theta3, test_data);

for i = 1:size(pred),
  if (pred(i) == 10)
  	pred(i) = 0;
  endif
end

%-----------------Writing output to file-----------------------------------
fprintf("\n\nWriting Predictions to file (submission.csv)....\n\n");
ImageId = 1:1:size(pred);
output = [ImageId' pred];
csvwrite ('submission.csv', output);