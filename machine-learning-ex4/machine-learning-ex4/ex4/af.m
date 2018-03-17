
[x map]=rgb2ind(i);
 y=ind2gray(x,map);
  y=double(y);
   y=(255.0*(y/max(max(y))));
    h=[1;y(:)]';
     h1 = sigmoid(h* Theta1');
      h2 = sigmoid([1 h1] * Theta2')
   