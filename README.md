# Gradient Descent Optimiser (with NumPy subclassing)

This project contains a small Python implementation of a gradient descent optimiser, together with unit tests.  
It demonstrates how to subclass `numpy.ndarray` to carry optimisation metadata and how to run optimisation against symbolic/numeric functions.

---

## Files

### `gradient_descent.py`
Implements the optimiser:

- **`class GradientDescent(np.ndarray)`**  
  A subclass of NumPy arrays that represents optimisation parameters.  
  It stores:
  - `gradient_current` – the current gradient vector.
  - `gradient_previous` – the previous gradient vector.
  - `gradient_step` – the current iteration count.
  - `minima` – a list for storing candidate minima.

- **`gradient_descent_step(...)`**  
  Performs one gradient descent (or ascent) step with:
  - learning rate,
  - momentum,
  - optional noise injection,
  - support for both minimisation and maximisation.

- **`gradient_descent_from_function(...)`**  
  Runs optimisation on a provided function and its gradient.  
  Allows multiple random restarts (`num_attempts`) and returns:
  - the minimum (or maximum) found,
  - the corresponding function value.

- **Manual test block (`if __name__ == "__main__":`)**  
  Contains example runs with simple analytic functions.  
  These are kept for illustration but are considered obsolete since pytest tests are provided separately.

---

### `test_gradient_descent.py`
Contains **pytest-based unit tests**.  
Each test checks that the function value returned by the optimiser matches the known minimum/maximum within tolerance `1e-8`.

Tests included:
- **`test_quartic_minimum`**: A quartic function with minimum at (1, -1).
- **`test_simple_quadratic`**: A quadratic with minimum at (0, 1).
- **`test_exponential_target`**: A higher-dimensional function involving powers of 2 with minimum at (8, 0, 0, 8, 5).
- **`test_maximisation`**: A function with a known maximum at x = 6 (demonstrating gradient ascent mode).

Helper function:
- **`assert_value_equal(...)`**  
  Compares the returned minimum value against the true value of the function at the known extremum, enforcing tolerance of `1e-8`.
