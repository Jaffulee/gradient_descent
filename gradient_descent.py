import numpy as np

class GradientDescent(np.ndarray):
    def __new__(cls, input_array):
        # Ensure it's a NumPy array, and force 1D
        arr = np.asarray(input_array).reshape(-1)
        obj = arr.view(cls)

        # Initialize attributes as arrays of zeros
        obj.gradient_current = np.zeros_like(arr, dtype=float)
        obj.gradient_previous = np.zeros_like(arr, dtype=float)
        obj.gradient_step = 0
        obj.minima = []
        return obj

    def __array_finalize__(self, obj):
        if obj is None: 
            return
        # Copy attributes from the source object if they exist
        self.gradient_current = getattr(obj, 'gradient_current', None)
        self.gradient_previous = getattr(obj, 'gradient_previous', None)
        self.gradient_step = getattr(obj, 'gradient_step', None)
        self.minima = getattr(obj, 'minima', None)

    def gradient_descent_step(self, gradient_current, rate_parameter=0.05, previous_rate_parameter = 0.001, momentum_parameter=0.4, noise_parameter=0.001, noise_interval = 1000, noise_sigma=1):
        if noise_parameter == 0:
            noise = 0
        else:
            if (self.gradient_step + 1)% noise_interval == 0:
                # noise_parameter = noise_parameter*(np.sin(noise_wavelength * self.gradient_step / (2 * np.pi))**5)
                noise = np.random.normal(0, noise_sigma, self.size)
            else:
                noise = 0
        x = self - rate_parameter*gradient_current - previous_rate_parameter*self.gradient_previous
        x = x + momentum_parameter*(x-self) + noise_parameter*noise
        x = GradientDescent(x)
        x.gradient_previous = gradient_current
        x.gradient_step = self.gradient_step + 1
        return x
    
def gradient_descent_from_function(dim, gradf, f, num_attempts=5, xinit_radius=10, error_flag = 1e-8, max_iter = 1e6, verbose=False,
                                   rate_parameter=0.05, previous_rate_parameter = 0.001, momentum_parameter=0.4, noise_parameter=0.001, noise_interval = 1000, noise_sigma=1):
    minima = []
    for iter in range(num_attempts):
        xinit = np.random.uniform(-xinit_radius,xinit_radius,dim)
        x = GradientDescent(xinit)

        error = 1
        try:
            while error > error_flag and x.gradient_step < max_iter:
                # print(x)
                gradfx = gradf(*x)
                grad_norm = np.linalg.norm(gradfx)
                if grad_norm > 1.0:   # threshold
                    gradfx = gradfx / grad_norm
                xnew = x.gradient_descent_step(gradfx, rate_parameter=rate_parameter, previous_rate_parameter = previous_rate_parameter, momentum_parameter=momentum_parameter, noise_parameter=noise_parameter, noise_interval=noise_interval, noise_sigma=noise_sigma)
                # error = np.linalg.norm(x-xnew)
                error = grad_norm
                x = xnew
            if verbose:
                print("x =", x, ", error =", error, ", iter =", x.gradient_step)
            minima.append(x)
        except Exception as e:
            print(e)
            continue

    minima_values = [f(*x) for x in minima]
    minima_pos = np.argmin(minima_values)
    minima_value = minima_values[minima_pos]
    minima = minima[minima_pos]

    return minima, minima_value


if __name__ == '__main__':
    def evaluate_error_grad_desc(minimum, minimum_grad, tol = 1e-8):
        error = np.linalg.norm(minimum-minimum_grad)
        print(error)
        result = error<tol
        print(result)
        return result

    def f0(x,y):
        return (x-1)**4 + (y+1)**4 + 1    

    def gradf0(x,y):
        return np.array([4*((x-1)**3), 4*((y+1)**3)])
    
    test_results = []
    
    minima0, minima_value0 = [1,-1], 1
    minima, minima_value = gradient_descent_from_function(2,gradf0,f0,verbose=True)
    print(minima, minima_value)
    print(minima0,minima_value0)
    test_result = evaluate_error_grad_desc(minima_value0,minima_value)
    print(test_result)
    test_results.append(test_result)


    import sympy as sp
    x1,x2,x3,x4,x5 = sp.symbols('x1 x2 x3 x4 x5', real=True)

    args1 = [x1,x2]
    f1 = sp.Matrix([(x1+x2 - 1)**2 + (x1-x2 + 1)**2])
    gradf1 = f1.jacobian(args1)

    f1 = sp.lambdify(args1,f1,"numpy")
    gradf1 = sp.lambdify(args1,gradf1,"numpy")

    minima1, minima_value1 = [0,1], 0
    minima, minima_value = gradient_descent_from_function(2,gradf1,f1,verbose=True)
    print(minima, minima_value)
    print(minima1,minima_value1)
    test_result = evaluate_error_grad_desc(minima_value1,minima_value)
    print(test_result)
    test_results.append(test_result)

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
    minima, minima_value = gradient_descent_from_function(5, gradf2, f2, rate_parameter=0.0005, previous_rate_parameter = 0.00001, momentum_parameter=0.9, noise_parameter=0.00001, noise_interval= 100000, verbose=True)
    print(minima, minima_value)
    print(minima2,minima_value2)
    test_result = evaluate_error_grad_desc(minima_value2,minima_value)
    print(test_result)
    test_results.append(test_result)

    print("\nTest results:")
    print(list(enumerate(test_results)))












