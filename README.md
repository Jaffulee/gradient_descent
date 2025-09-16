<!DOCTYPE html>
<html lang="en">

<body>
<img width="1446" height="1292" alt="image" src="https://github.com/user-attachments/assets/9195d97a-044b-42e6-9f7d-a7036522a41e" />

<h1>Gradient Descent Optimiser and Neural Network (Tensor-based Backpropagation)</h1>

<p>
This project contains a Python implementation of a <strong>gradient descent optimiser</strong>, 
a <strong>fully custom neural network framework</strong>, and a 
<strong>generalised Jacobian class</strong> for handling tensor derivatives.  
The framework is built from scratch using <code>NumPy</code>, with tensor-calculus-based backpropagation, 
and demonstrates training on toy datasets such as <strong>fitting a sine wave</strong> and a <strong>helix projection</strong>.
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
      <li>Learning rate control.</li>
      <li>Descent or ascent.</li>
      <li>Momentum support.</li>
      <li>Optional Gaussian noise at fixed intervals.</li>
    </ul>
  </li>
  <li>
    <strong><code>gradient_descent_from_function(...)</code></strong>  
    Optimises any function with a known gradient:
    <ul>
      <li>Multiple random restarts.</li>
      <li>Gradient clipping.</li>
      <li>Convergence checks (gradient norm).</li>
      <li>Maximisation as well as minimisation.</li>
    </ul>
  </li>
</ul>

<hr>

<h3>2. <code>neural_network.py</code></h3>
<p>Implements a <strong>general neural network pipeline</strong>:</p>

<ul>
  <li><strong>Activation functions</strong>: <code>sigmoid</code>, <code>relu</code>, <code>tanh</code>, <code>softmax</code>, <code>identity</code>.  
    Each with corresponding derivatives, accessible via the <code>ActivationFunctions</code> registry.
  </li>
  <li>
    <strong><code>class TensorJacobian(np.ndarray)</code></strong>  
    <ul>
      <li>Generalised Jacobian supporting arbitrary tensor derivatives.</li>
      <li>Stores numerator and denominator tensor types.</li>
      <li>Overloads multiplication to compose tensor derivatives via Einstein summation (<code>einsum</code>).</li>
      <li>Handles higher-order tensor calculus automatically.</li>
    </ul>
  </li>
  <li><strong>Forward propagation</strong>: arbitrary depth and width defined via <code>layer_node_nums</code>.</li>
  <li><strong>Backward propagation</strong>:  
    <ul>
      <li>Derived from full tensor calculus (Jacobian compositions).</li>
      <li>Computes gradients with respect to weights and biases (<code>DEDWs</code>, <code>DEDbs</code>).</li>
      <li>Supports multiple hidden layers with arbitrary sizes.</li>
    </ul>
  </li>
  <li><strong>Loss function</strong>: column-wise mean squared error  
    <pre>
E(Y, Yhat) = (1 / (2m)) * Σ ||yᵢ - ŷᵢ||²
    </pre>
    with derivative included.
  </li>
</ul>

<hr>

<h3>3. <code>neural_network_test.py</code></h3>
<p>Demonstrates training the network on two toy problems:</p>

<ul>
  <li><strong>Sine wave fitting</strong>:  
    <ul>
      <li>Random samples of sin(x) between 0 and 2π, rescaled to [0, 1].</li>
      <li>Network: multilayer perceptron, e.g. <code>[1, 6, 1]</code>.</li>
      <li>Plots:
        <ul>
          <li>True sine curve (black).</li>
          <li>Training samples (red points, smaller markers for clarity).</li>
          <li>Intermediate predictions every 10% of training (low opacity lines).</li>
          <li>Final prediction (green line).</li>
        </ul>
      </li>
      <li>Also plots training loss over iterations.</li>
    </ul>
  </li>
  <li><strong>Helix projection fitting</strong>:  
    <ul>
      <li>Function: f(t) = [t cos t, t sin t].</li>
      <li>Network: <code>[1, 10, 2]</code>.</li>
      <li>Plots:
        <ul>
          <li>True XY helix projection (black).</li>
          <li>Training samples (red points).</li>
          <li>Intermediate predictions every 10% of training (low opacity lines).</li>
          <li>Final NN prediction (green dashed line).</li>
        </ul>
      </li>
      <li>Also plots training loss.</li>
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
  <li>Initialise random neural networks.</li>
  <li>Train on sine and helix datasets.</li>
  <li>Output progress logs (iterations + loss).</li>
  <li>Generate plots for true functions, training data, predictions, and training losses.</li>
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
  <li>Training a custom neural network on both <strong>sine wave regression</strong> and <strong>helix projection</strong>.</li>
  <li>Visualising training with snapshots of predictions and loss curves.</li>
</ul>

</body>
</html>
