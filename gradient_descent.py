import numpy as np
from typing import Callable, List, Tuple, Union, Any


class GradientDescent(np.ndarray):
    """
    A subclass of numpy.ndarray that stores additional information
    useful for gradient descent optimization, such as current and previous
    gradients, iteration count, and a history of minima.

    Attributes
    ----------
    gradient_current : np.ndarray
        The current gradient vector (same shape as the input array).
    gradient_previous : np.ndarray
        The previous gradient vector.
    gradient_step : int
        Counter for the number of gradient descent steps taken.
    minima : list
        A list to store discovered minima during optimization.
    """

    gradient_current: np.ndarray
    gradient_previous: np.ndarray
    gradient_step: int
    minima: List[Any]

    def __new__(cls, input_array: Union[np.ndarray, List[float]]) -> "GradientDescent":
        """
        Create a new GradientDescent object from an input array.

        Parameters
        ----------
        input_array : array-like
            The initial values for the optimization variables.

        Returns
        -------
        GradientDescent
            An instance of GradientDescent.
        """
        arr = np.asarray(input_array).reshape(-1)
        obj = arr.view(cls)

        obj.gradient_current = np.zeros_like(arr, dtype=float)
        obj.gradient_previous = np.zeros_like(arr, dtype=float)
        obj.gradient_step = 0
        obj.minima = []
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        """
        Finalize creation of the array, ensuring attributes are preserved
        when new views or slices are created.

        Parameters
        ----------
        obj : Any
            The object being viewed or sliced from.
        """
        if obj is None:
            return
        self.gradient_current = getattr(obj, 'gradient_current', None)
        self.gradient_previous = getattr(obj, 'gradient_previous', None)
        self.gradient_step = getattr(obj, 'gradient_step', None)
        self.minima = getattr(obj, 'minima', None)

    def gradient_descent_step(
        self,
        gradient_current: np.ndarray,
        descent: bool = True,
        rate_parameter: float = 0.05,
        previous_rate_parameter: float = 0.001,
        momentum_parameter: float = 0.75,
        noise_parameter: float = 0.001,
        noise_interval: int = 10000,
        noise_sigma: float = 1.0,
    ) -> "GradientDescent":
        """
        Perform one gradient descent update step.

        Parameters
        ----------
        gradient_current : np.ndarray
            The current gradient vector.
        descent : bool
            Whether the algorithm is gradient descent (True) or ascent (False) (default True).
        rate_parameter : float, optional
            Step size for the current gradient (default 0.05).
        previous_rate_parameter : float, optional
            Weight for the previous gradient (default 0.001).
        momentum_parameter : float, optional
            Momentum coefficient (default 0.75).
        noise_parameter : float, optional
            Scale factor for random noise injection (default 0.001).
        noise_interval : int, optional
            Interval of steps at which noise is added (default 10000).
        noise_sigma : float, optional
            Standard deviation of the Gaussian noise (default 1.0).

        Returns
        -------
        GradientDescent
            Updated parameters after one gradient descent step.
        """
        if noise_parameter == 0:
            noise = 0
        else:
            if (self.gradient_step + 1) % noise_interval == 0:
                noise = np.random.normal(0, noise_sigma, self.size)
            else:
                noise = 0
        if descent:
            descent_flag = 1
        else:
            descent_flag = -1
        x = self - descent_flag * rate_parameter * gradient_current - descent_flag * previous_rate_parameter * self.gradient_previous
        x = x + momentum_parameter * (x - self) + noise_parameter * noise
        x = GradientDescent(x)
        x.gradient_previous = gradient_current
        x.gradient_step = self.gradient_step + 1
        return x


def gradient_descent_from_function(
    dim: int,
    gradf: Callable[..., np.ndarray],
    f: Callable[..., float],
    num_attempts: int = 5,
    xinit_radius: float = 10,
    error_flag: float = 1e-8,
    max_iter: int = int(1e6),
    verbose: bool = False,
    descent: bool = True,
    rate_parameter: float = 0.05,
    previous_rate_parameter: float = 0.001,
    momentum_parameter: float = 0.75,
    noise_parameter: float = 0.001,
    noise_interval: int = 10000,
    noise_sigma: float = 1.0,
) -> Tuple[GradientDescent, float]:
    """
    Run gradient descent optimization on a given function.

    Parameters
    ----------
    dim : int
        Dimension of the input space.
    gradf : Callable
        Function that computes the gradient of f.
    f : Callable
        Objective function to minimize.
    num_attempts : int, optional
        Number of random initializations (default 5).
    xinit_radius : float, optional
        Range for uniform random initialization (default 10).
    error_flag : float, optional
        Convergence tolerance for gradient norm (default 1e-8).
    max_iter : int, optional
        Maximum iterations allowed (default 1e6).
    verbose : bool, optional
        Whether to print progress information (default False).
    rate_parameter : float, optional
        Learning rate (default 0.05).
    previous_rate_parameter : float, optional
        Weight for previous gradient (default 0.001).
    momentum_parameter : float, optional
        Momentum coefficient (default 0.75).
    noise_parameter : float, optional
        Scale of injected noise (default 0.001).
    noise_interval : int, optional
        Interval between noise injections (default 10000).
    noise_sigma : float, optional
        Standard deviation of injected Gaussian noise (default 1.0).

    Returns
    -------
    minima : GradientDescent
        The position of the minimum found.
    minima_value : float
        The value of the function at the minimum.
    """
    minima: List[GradientDescent] = []
    for _ in range(num_attempts):
        xinit = np.random.uniform(-xinit_radius, xinit_radius, dim)
        x = GradientDescent(xinit)

        error = 1.0
        try:
            while error > error_flag and x.gradient_step < max_iter:
                gradfx = gradf(*x)
                grad_norm = np.linalg.norm(gradfx)
                if grad_norm > 1.0:  # gradient clipping
                    gradfx = gradfx / grad_norm
                xnew = x.gradient_descent_step(
                    gradfx,
                    descent = descent,
                    rate_parameter=rate_parameter,
                    previous_rate_parameter=previous_rate_parameter,
                    momentum_parameter=momentum_parameter,
                    noise_parameter=noise_parameter,
                    noise_interval=noise_interval,
                    noise_sigma=noise_sigma,
                )
                error = grad_norm
                x = xnew
            if verbose:
                print("x =", x, ", error =", error, ", iter =", x.gradient_step)
            minima.append(x)
        except Exception as e:
            print(e)
            continue
    
        minima_values = [f(*x) for x in minima]

        if descent:
            minima_pos = int(np.argmin(minima_values))
        else:
            minima_pos = int(np.argmax(minima_values))

        minimum_value = minima_values[minima_pos]
        minimum = minima[minima_pos]

    return minimum, minimum_value


if __name__ == '__main__':
    def evaluate_error_grad_desc(minimum, minimum_grad, tol = 1e-8):
        error = np.linalg.norm(minimum-minimum_grad)
        print(error)
        result = error<tol
        print(result)
        return result

    # Test
    def f0(x,y):
        return (x-1)**4 + (y+1)**4 + 1    

    def gradf0(x,y):
        return np.array([4*((x-1)**3), 4*((y+1)**3)])
    
    test_results = []
    
    minima0, minima_value0 = [1,-1], 1
    minima, minima_value = gradient_descent_from_function(2, gradf0, f0, rate_parameter=0.7, previous_rate_parameter = 0.1, momentum_parameter=0.7, verbose=True)
    print(minima, minima_value)
    print(minima0,minima_value0)
    test_result = evaluate_error_grad_desc(minima_value0,minima_value)
    print(test_result)
    test_results.append(test_result)


    import sympy as sp
    x1,x2,x3,x4,x5 = sp.symbols('x1 x2 x3 x4 x5', real=True)

    # Test
    args1 = [x1,x2]
    f1 = sp.Matrix([(x1+x2 - 1)**2 + (x1-x2 + 1)**2])
    gradf1 = f1.jacobian(args1)

    f1 = sp.lambdify(args1,f1,"numpy")
    gradf1 = sp.lambdify(args1,gradf1,"numpy")

    minima1, minima_value1 = [0,1], 0
    minima, minima_value = gradient_descent_from_function(2,gradf1,f1, verbose=True)
    print(minima, minima_value)
    print(minima1,minima_value1)
    test_result = evaluate_error_grad_desc(minima_value1,minima_value)
    print(test_result)
    test_results.append(test_result)

    # Test
    args2 = [x1,x2,x3,x4,x5]
    f2 = sp.Matrix([(2**x1-2**8)**2 +
                    (2**x2-2**0)**2 +
                    (2**x3-2**0)**2 +
                    (2**x4-2**8)**2 +
                    (2**x5-2**5)**2
                    +80085
                    ])
    gradf2 = f2.jacobian(args2)

    f2 = sp.lambdify(args2,f2,"numpy")
    gradf2 = sp.lambdify(args2,gradf2,"numpy")

    minima2, minima_value2 = [8,0,0,8,5], 80085
    minima, minima_value = gradient_descent_from_function(5, gradf2, f2, rate_parameter=0.0005, previous_rate_parameter = 0.00001, momentum_parameter=0.95, noise_parameter=1, noise_interval=100000, noise_sigma=2, verbose=True)
    print(minima, minima_value)
    print(minima2,minima_value2)
    test_result = evaluate_error_grad_desc(minima_value2,minima_value)
    print(test_result)
    test_results.append(test_result)

    # Test
    args3 = [x1]
    f3 = sp.Matrix([-((x1-6)**2)*(((2**x1)+1)**2)+6])
    gradf3 = f3.jacobian(args3)

    f3 = sp.lambdify(args3,f3,"numpy")
    gradf3 = sp.lambdify(args3,gradf3,"numpy")

    minima3, minima_value3 = [6], 6
    minima, minima_value = gradient_descent_from_function(1, gradf3, f3 , num_attempts=15, xinit_radius=15, descent=False, rate_parameter=0.00005, previous_rate_parameter = 0.00001, momentum_parameter=0.97, noise_parameter= 0.000001, noise_interval=50000,verbose=True)
    print(minima, minima_value)
    print(minima3,minima_value3)
    test_result = evaluate_error_grad_desc(minima_value3,minima_value)
    print(test_result)
    test_results.append(test_result)

    print("\nTest results:")
    print(list(enumerate(test_results)))
