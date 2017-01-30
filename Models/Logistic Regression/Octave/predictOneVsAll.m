function p = predictOneVsAll(all_theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters all_theta
%   p = PREDICT(all_theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(all_theta'*x) >= 0.5, predict 1)

m = size(X, 1);
num_labels = size(all_theta, 1);

% Need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the data matrix
test_data = [ones(m, 1) X];
x = X';

% =========================================================================

for i=1:m,
  for j=1:num_labels,
    %pred(j)=X(i,:)*all_theta(j,:)';
    pred(j) = all_theta(j,:)*x(:,i);
  end;
  [val p(i)]=max(pred, [], 2);
end;

% =========================================================================


end
