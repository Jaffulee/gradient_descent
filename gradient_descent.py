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

    def gradient_descent_step(self, gradient_current, rate_parameter=0.05, momentum_parameter=0.01, noise_parameter=0.01, noise_interval = 100, noise_sigma=1):
        if noise_parameter == 0:
            noise = 0
        else:
            if (self.gradient_step + 1)% noise_interval == 0:
                # noise_parameter = noise_parameter*(np.sin(noise_wavelength * self.gradient_step / (2 * np.pi))**5)
                noise = np.random.normal(0, noise_sigma, self.size)
            else:
                noise = 0

        x = self - rate_parameter*gradient_current - momentum_parameter*self.gradient_previous + noise_parameter*noise
        x = GradientDescent(x)
        x.gradient_previous = gradient_current
        x.gradient_step = self.gradient_step + 1
        return x
    


def f(x,y):
    return (np.sin(x)*x-1)**4 + (np.cos(y)*y+1)**4 + 1    

def gradf(x,y):
    return np.array([4*(np.cos(x)*x + np.sin(x))*((np.sin(x)-1)**3), 4*(-np.sin(y)*y + np.cos(y))*((np.cos(y)+1)**3)])



max_iter = 100000
minima = []
for iter in range(10):
    xinit = np.random.uniform(-10,10,2)
    x = GradientDescent(xinit)
    error = 1
    while error > 1e-8 and x.gradient_step < max_iter:
        # print(x)
        gradfx = gradf(*x)
        grad_norm = np.linalg.norm(gradfx)
        if grad_norm > 1.0:   # threshold
            gradfx = gradfx / grad_norm
        xnew = x.gradient_descent_step(gradfx)
        # error = np.linalg.norm(x-xnew)
        error = grad_norm
        x = xnew
    print(x,error)
    minima.append(x)

minima_values = [f(*x) for x in minima]
minima_pos = np.argmin(minima_values)
minima_value = minima_values[minima_pos]
minima = minima[minima_pos]
print(minima, minima_value)


