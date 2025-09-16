<!DOCTYPE html>
<html lang="en">

<body>
<img width="1446" height="1292" alt="image" src="https://github.com/user-attachments/assets/9195d97a-044b-42e6-9f7d-a7036522a41e" />


<h1>Gradient Descent Optimiser and Neural Network (Tensor-based Backpropagation)</h1>

<p>
This project contains a Python implementation of a <strong>gradient descent optimiser</strong>, 
a <strong>fully custom neural network framework</strong>, and a 
<strong>generalised Jacobian class</strong> for handling tensor derivatives.  
The network is trained on a simple example: fitting a sine wave using multilayer perceptrons.
</p>

<hr>

<h2>Features</h2>

<h3>1. <code>gradient_descent.py</code></h3>
<p>Implements the optimiser:</p>

<ul>
  <li>
    <strong><code>class GradientDescent(np.ndarray)</code></strong>  
    <ul>
      <li>Preserves the shape of optimisation variables.</li>
      <li>Stores metadata (<code>gradient_current</code>, <code>gradient_previous</code>, <code>gradient_step</code>, <code>minima</code>).</li>
      <li>Supports momentum and noise injection.</li>
    </ul>
  </li>
  <li>
    <strong><code>gradient_descent_step(...)</code></strong>  
    Performs one update step:
    <ul>
      <li>Descent or ascent.</li>
      <li>Learning rate control.</li>
      <li>Momentum.</li>
      <li>Optional Gaussian noise at fixed intervals.</li>
    </ul>
  </li>
  <li>
    <strong><code>gradient_descent_from_function(...)</code></strong>  
    Optimises any function with a known gradient:
    <ul>
      <li>Multiple random restarts.</li>
      <li>Gradient clipping.</li>
      <li>Convergence based on gradient norm.</li>
      <li>Maximisation as well as minimisation.</li>
    </ul>
  </li>
</ul>

<hr>

<h3>2. <code>neural_network.py</code></h3>
<p>Implements a <strong>general neural network pipeline</strong>:</p>

<ul>
  <li><strong>Activation functions</strong>: <code>sigmoid</code>, <code>relu</code>, <code>tanh</code> (with derivatives), accessed via the <code>ActivationFunctions</code> registry.</li>
  <li>
    <strong><code>class TensorJacobian(np.ndarray)</code></strong>  
    <ul>
      <li>Generalised Jacobian supporting arbitrary tensor derivatives.</li>
      <li>Stores numerator and denominator tensor types.</li>
      <li>Overloads multiplication to compose tensor derivatives via Einstein summation.</li>
      <li>Handles higher-order tensor calculus.</li>
    </ul>
  </li>
  <li><strong>Forward propagation</strong>: arbitrary depth and width defined via <code>layer_node_nums</code>.</li>
  <li><strong>Backward propagation (from tensor calculus)</strong>:  
    <ul>
      <li>Derives full Jacobians for each layer.</li>
      <li>Computes gradients with respect to weights and biases (<code>DEDWs</code>, <code>DEDbs</code>).</li>
      <li>Supports multiple hidden layers with arbitrary sizes.</li>
    </ul>
  </li>
  <li><strong>Loss function</strong>: column-wise mean squared error  
    <pre>
E(Y, Yhat) = (1 / (2m)) * Σ ||yᵢ - ŷᵢ||²
    </pre>
    with derivative provided.
  </li>
</ul>

<hr>

<h3>3. <code>neural_network_test.py</code></h3>
<p>Demonstrates training the network to fit a sine wave:</p>

<ul>
  <li><strong>Dataset</strong>: random samples of sin(x) between 0 and 2π, rescaled to [0, 1].</li>
  <li><strong>Model</strong>: multilayer perceptron, e.g. <code>[1, 10, 10, 1]</code>.</li>
  <li><strong>Training loop</strong>:  
    <ul>
      <li>Gradient descent updates using the custom optimiser.</li>
      <li>Forward pass on both training and curve points.</li>
      <li>Plots: the true sine curve, training points, and network predictions at intervals.</li>
    </ul>
  </li>
</ul>

<hr>

<h2>Example Run</h2>

<pre>
python neural_network_test.py
</pre>

<p>This will:</p>
<ul>
  <li>Initialise a random neural network.</li>
  <li>Fit it to noisy sine samples.</li>
  <li>Plot the true sine, training points, and iterative predictions.</li>
</ul>

<hr>

<h2>Tests</h2>

<p><code>test_gradient_descent.py</code> contains pytest-based unit tests:</p>
<ul>
  <li>Quartic function minimisation.</li>
  <li>Simple quadratic minimum.</li>
  <li>Higher-dimensional exponential minimisation.</li>
  <li>Gradient ascent demonstration (maximisation).</li>
</ul>

<p>Tolerance for all tests: <strong>1e-8</strong>.</p>

<pre>
pytest
</pre>

<hr>

<h2>Summary</h2>

<p>This project demonstrates:</p>
<ul>
  <li>Subclassing NumPy arrays to store optimisation metadata.</li>
  <li>Implementing gradient descent with momentum and noise.</li>
  <li>Deriving a neural network training pipeline from first principles of tensor calculus.</li>
  <li>Creating a generalised Jacobian (<code>TensorJacobian</code>) that supports higher-order tensor derivatives.</li>
  <li>Training a custom neural network on a sine wave dataset.</li>
</ul>

</body>
</html>
