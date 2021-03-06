<b>Description</b>

<p>This repository contains a number of python functions for implementing a 2 dimensional linear classifier. It includes functions for importing data, training model, and making predictions. One of the main and most interesting parts of this repository is its visualization of how batch and stochastic gradient descent models learn the parameters of a linear model. Currently, this repository is only in a very basic form with only batch and stochastic gradient descent for model training. It is my intention however to continue development on this repository to include more classifier models and training algorithms</p>

<b>Model Training Visualization</b>

Both the stochastic and batch gradient descent algorithms implemented in this repository have within them step-by-step plot/graph updating of the boundary decision graphs. This shows how each iterative step effects the decision boundary and how both techniques differ in the way they learn models. An example of a decision boundary graph can be seen below.

 <img src="https://cloud.githubusercontent.com/assets/11066939/10028693/d6a69c12-613b-11e5-92ba-d0b3de7d103c.png" alt="Batch Descent Decision Boundary Plot" width="570" height="430">
 
  <img src="https://cloud.githubusercontent.com/assets/11066939/10028691/d6a0e86c-613b-11e5-831a-3c550c7e567b.png" alt="Stochastic Descent Decision Boundary Plot" width="570" height="430">
 
