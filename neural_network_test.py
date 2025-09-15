import neural_network as nn
import gradient_descent as gd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Number of random points
    n_points = 60

    # Generate random x values between 0 and 2π
    x_random = np.random.uniform(0, 2*np.pi, n_points)
    y_random = (np.sin(x_random)+1)/2

    # Plot the sine curve for reference
    x_curve = np.linspace(0, 2*np.pi, 400)
    y_curve = (np.sin(x_curve)+1)/2

    # Reshape inputs
    Xhat = x_random.reshape(1, -1)  # training inputs
    Yhat = y_random.reshape(1, -1)  # training outputs
    X = x_curve.reshape(1, -1)      # evaluation inputs

    # Initialize network
    layer_node_nums = [1, 10, 1]
    Ws, bs = nn.init_weights(layer_node_nums, 1)

    # Training loop
    plt.figure(figsize=(8, 5))
    plt.plot(x_curve, y_curve, label="sin(x)", color="blue")
    plt.scatter(x_random, y_random, color="red", zorder=5, label="Training points")

    As, Zs = nn.forward_propagate(Xhat, Ws, bs, activation_function='sigmoid')

    iterations = 100000
    for it in range(iterations):
        
        DEDWs, DEDbs = nn.back_propagate(Xhat, Yhat, As, Zs, Ws, bs, activation_function='sigmoid')

        # Update weights with gradient descent
        Ws_gd = [gd.GradientDescent(W) for W in Ws]
        bs_gd = [gd.GradientDescent(b) for b in bs]
        Ws = [W.gradient_descent_step(DEDWs[i]) for i, W in enumerate(Ws_gd)]
        bs = [b.gradient_descent_step(DEDbs[i]) for i, b in enumerate(bs_gd)]

        # Forward pass on both training and curve points
        As, Zs = nn.forward_propagate(Xhat, Ws, bs, 'sigmoid')
        A_curve, _ = nn.forward_propagate(X, Ws, bs, 'sigmoid')

        # Plot prediction for current iteration
        if it%(iterations//10)==0:
            # print(As[-1])
            print(f"Iteration {it+1}")
            plt.plot(x_curve, A_curve[-1].flatten(), label=f"Iteration {it}", alpha=0.6)

    print(f"Iteration {it}")
    plt.plot(x_curve, A_curve[-1].flatten(), label=f"Iteration {it}", alpha=0.6)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Neural Network fitting sin(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
