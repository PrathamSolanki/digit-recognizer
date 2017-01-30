function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a 
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer1_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1,Theta2 and Theta3,the weight matrices
% for our 3 layer neural network
%Theta1 = reshape(nn_params(1:19625),25,785);

%Theta2 = reshape(nn_params(19626:20275),25,26);

%Theta3 = reshape(nn_params(20276:20535),10,26);

temp1 = hidden_layer1_size * (input_layer_size + 1);
Theta1 = reshape(nn_params(1:temp1),hidden_layer1_size, (input_layer_size + 1));

temp2 = hidden_layer1_size*(hidden_layer1_size+1);
Theta2 = reshape(nn_params((temp1+1):(temp1+temp2)),hidden_layer1_size,hidden_layer1_size+1);

temp3 = (num_labels*hidden_layer1_size+1);
Theta3 = reshape(nn_params((temp1+temp2+1):end),num_labels,hidden_layer1_size+1);

% Setup some useful variables
m = size(X, 1);
         
% Need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

% -------------------------Cost function Without Regularization-----------------------
x = X';

yk = zeros(num_labels, m);
y_ind = y;
for i = 1:m,
  if (y_ind(i) == 0)
  	y_ind(i) = 10;
  endif
  yk(y_ind(i),i) = 1;
end

a1 = [ones(1,m);x];

z2 = Theta1*a1;
a2 = sigmoid(z2);

a2 = [ones(1,m);a2];
z3 = Theta2*a2;
a3 = sigmoid(z3);

a3 = [ones(1,m);a3];
z4 = Theta3*a3;
a4 = sigmoid(z4);
hx = a4;

term1 = (-yk) .* log(hx);
term2 = (1-yk) .* log(1-hx);
J = sum(sum((term1 - term2)));
J = (1/m)*J;

% -----------------------Regularized Cost Function--------------------------------------

term1 = (-yk) .* log(hx);
term2 = (1-yk) .* log(1-hx);
J = sum(sum((term1 - term2)));
J = (1/m)*J;

%j1 = size(Theta1,1);
%k1 = size(Theta1,2);

t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));
t3 = Theta3(:,2:size(Theta3,2));

regularizationFactor = (lambda/(2*m)) * (sum(sum((t1).^2)) + sum(sum((t2).^2)) + sum(sum((t3).^2)));
J = J + regularizationFactor;

% ------------------------Back-Propagation without Regularization-------------------------------------

t3 = Theta3(:,2:size(Theta3,2));
t2 = Theta2(:,2:size(Theta2,2));
y = yk;
delta4 = a4 - y;

delta3 = ((t3')*delta4).*(sigmoidGradient(z3));
delta2 = ((t2')*delta3).*(sigmoidGradient(z2));

Theta1_grad = (1/m)*(delta2*(a1'));
Theta2_grad = (1/m)*(delta3*(a2'));
Theta3_grad = (1/m)*(delta4*(a3'));

% ------------------------Back-Propagation with Regularization-------------------------------------

for j = 2:size(Theta1_grad, 2)
    Theta1_grad(:,j) = Theta1_grad(:,j) + (lambda/m)*Theta1(:,j);
end

for j = 2:size(Theta2_grad, 2)
    Theta2_grad(:,j) = Theta2_grad(:,j) + (lambda/m)*Theta2(:,j);
end

for j = 2:size(Theta3_grad, 2)
    Theta3_grad(:,j) = Theta3_grad(:,j) + (lambda/m)*Theta3(:,j);
end

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];

end