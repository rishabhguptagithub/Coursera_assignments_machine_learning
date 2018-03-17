function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X=[ones(m,1) X];

del2=zeros(size(Theta2));
del1=zeros(size(Theta1));

for i=1:m
  a1=X(i,:);
  a2=sigmoid(Theta1*a1');
  a2=[1;a2];
  a3=sigmoid(Theta2*a2);
  
  
  
  
 c=zeros(num_labels,1);
  for l=1:num_labels
    c(l)=l;
  end
  if y(i)==0
    y(i)=10;
  end
  d=(c==y(i));
  
  for j=1:num_labels
    J=J+(((-d(j)*log(a3(j)))-((1-d(j))*(log(1-a3(j)))))/m);
  end
  
    
    
  
  

d3=a3-d;
d2=((Theta2)'*d3).*(a2.*(ones(size(a2))-a2));
d2=d2(2:hidden_layer_size+1);
del2=del2+(d3*(a2)');
del1=del1+(d2*(a1));
end

Q1=Theta1;
Q1(:,1)=0;
Q2=Theta2;
Q2(:,1)=0;
Theta1_grad=(del1+(lambda*Q1))/m;

Theta2_grad=(del2+(lambda*Q2))/m;


s1=0;s2=0;
  for p=1:hidden_layer_size
    for o=1:input_layer_size
      s1=s1+((Theta1(p,o+1))^2);
    end
  end
  for p=1:num_labels
    for o=1:hidden_layer_size
      s2=s2+((Theta2(p,o+1))^2);
    end
  end
  J=J+((lambda/(2*m))*(s1+s2));












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
