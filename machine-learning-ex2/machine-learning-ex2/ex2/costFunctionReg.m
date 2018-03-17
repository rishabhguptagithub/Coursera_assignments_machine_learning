function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h=X*theta;s=0;
for j=2:size(theta)
  s=s+(theta(j)^2);
end
  J=(1/m)*(-y'*log(sigmoid(h))-(1-y)'*log(ones(size(h))-sigmoid(h)))+((lambda/(2*m))*s);
g=(X'*(sigmoid(h)-y))/m;
grad(1)=g(1);
for i=2:size(theta)
  grad(i)=g(i)+((lambda/m)*theta(i));
end


% =============================================================

end