function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Computes cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ===========================================================================

x = X';
y_transpose = y';

z = (theta')*x;
hx = sigmoid(z);

v1 = log(hx)*(-y);
v2 = log(1-hx)*(1-y);
term1 = (1/m)*(v1-v2);
term2 = 0;
for j = 2:size(theta)
    term2 = term2 + (lambda/(2*m))*((theta(j))^2);
end
J = term1 + term2;

grad(1) = sum((1/m)*(hx-y_transpose).*x(1,:));
for j = 2:size(grad)
    grad(j) = sum((1/m)*(hx-y_transpose).*x(j,:)) + ((lambda/m)*theta(j));
end 

% ==============================================================================

grad = grad(:);

end
