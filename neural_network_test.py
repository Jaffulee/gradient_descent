import neural_network as nn
import gradient_descent as gd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # ==========================================================
    # Test 1: Fit sin(x) scaled into [0,1]
    # ==========================================================
    print("\n=== Training on sin(x) ===")

    # Training data
    n_points = 60
    x_random = np.random.uniform(0, 2*np.pi, n_points)
    y_random = (np.sin(x_random) + 1) / 2

    # Evaluation grid
    x_curve = np.linspace(0, 2*np.pi, 400)
    y_curve = (np.sin(x_curve) + 1) / 2

    # Reshape inputs
    X_train = x_random.reshape(1, -1)
    Y_train = y_random.reshape(1, -1)
    X_eval = x_curve.reshape(1, -1)
    Y_eval = y_curve.reshape(1, -1)

    # Initialize network
    layer_node_nums = [1, 6, 1]
    Ws, bs = nn.init_weights(layer_node_nums, radius=1)

    # Training setup
    activations = {
        'final_activation_function': 'sigmoid', # Values between 0 and 1
        'activation_function': 'sigmoid'
    }
    As, Zs = nn.forward_propagate(X_train, Ws, bs, **activations)

    iterations = 100000
    losses = []
    snapshots = {}

    for it in range(iterations):
        # Backpropagation
        DEDWs, DEDbs = nn.back_propagate(X_train, Y_train, As, Zs, Ws, bs, **activations)

        # Gradient descent step
        Ws_gd = [gd.GradientDescent(W) for W in Ws]
        bs_gd = [gd.GradientDescent(b) for b in bs]
        Ws = [W.gradient_descent_step(DEDWs[i], rate_parameter=0.6) for i, W in enumerate(Ws_gd)]
        bs = [b.gradient_descent_step(DEDbs[i], rate_parameter=0.95) for i, b in enumerate(bs_gd)]

        # Forward pass
        As, Zs = nn.forward_propagate(X_train, Ws, bs, **activations)
        Y_pred = As[-1]

        # Track loss
        loss = nn.columnwise_mse(Y_pred, Y_train)
        losses.append(loss)

        # Save snapshots for plotting
        if it % (iterations // 10) == 0:
            print(f"Iteration {it+1}/{iterations}, Loss = {loss:.6f}")
            A_curve, _ = nn.forward_propagate(X_eval, Ws, bs, **activations)
            snapshots[it+1] = A_curve[-1].flatten()

    # Final prediction
    A_curve, _ = nn.forward_propagate(X_eval, Ws, bs, **activations)
    Y_curve_pred = A_curve[-1]

    # Plot fit
    plt.figure(figsize=(8, 5))
    plt.plot(x_curve, y_curve, label="True sin(x)", color="black")  # black true curve
    plt.scatter(x_random, y_random, color="red", s=15, zorder=5, label="Training points")  # smaller red circles

    # Plot snapshots with low opacity
    for it, y_snapshot in snapshots.items():
        plt.plot(x_curve, y_snapshot, alpha=0.3, label=f"Iter {it}")

    # Plot final
    plt.plot(x_curve, Y_curve_pred.flatten(), label="Final NN Prediction", color="green")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Neural Network fitting sin(x)")
    plt.legend()
    plt.grid(True)
    #plt.show()

    # Plot training loss
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss (sin test)")
    plt.grid(True)
    #plt.show()


    # ==========================================================
    # Test 2: Fit helix projection f(t) = [t cos t, t sin t]
    # ==========================================================
    print("\n=== Training on helix projection f(t) = [t cos t, t sin t] ===")

    # Training data
    n_points = 200
    t_random = np.random.uniform(0, 2*np.pi, n_points)
    x_random = t_random * np.cos(t_random)
    y_random = t_random * np.sin(t_random)

    # Reshape inputs/outputs
    T_train = t_random.reshape(1, -1)
    Y_train = np.vstack([x_random, y_random])

    # Evaluation grid
    t_curve = np.linspace(0, 2*np.pi, 400)
    X_curve = t_curve * np.cos(t_curve)
    Y_curve = t_curve * np.sin(t_curve)
    T_eval = t_curve.reshape(1, -1)
    Y_eval = np.vstack([X_curve, Y_curve])

    # Initialize network
    layer_node_nums = [1, 10, 2]
    Ws, bs = nn.init_weights(layer_node_nums, radius=0.5)

    # Training setup
    activations = {
        'final_activation_function': 'identity',
        'activation_function': 'tanh'
    }
    As, Zs = nn.forward_propagate(T_train, Ws, bs, **activations)

    iterations = 1000
    losses = []
    snapshots = {}

    for it in range(iterations):
        # Backpropagation
        DEDWs, DEDbs = nn.back_propagate(T_train, Y_train, As, Zs, Ws, bs, **activations)

        # Gradient descent step
        Ws_gd = [gd.GradientDescent(W) for W in Ws]
        bs_gd = [gd.GradientDescent(b) for b in bs]
        Ws = [W.gradient_descent_step(DEDWs[i], rate_parameter=0.05, momentum_parameter=0.2) for i, W in enumerate(Ws_gd)]
        bs = [b.gradient_descent_step(DEDbs[i], rate_parameter=0.05, momentum_parameter=0.2) for i, b in enumerate(bs_gd)]

        # Forward pass
        As, Zs = nn.forward_propagate(T_train, Ws, bs, **activations)
        Y_pred = As[-1]

        # Track loss
        loss = nn.columnwise_mse(Y_pred, Y_train)
        losses.append(loss)

        # Save snapshots
        if it % (iterations // 10) == 0:
            print(f"Iteration {it+1}/{iterations}, Loss = {loss:.6f}")
            A_curve, _ = nn.forward_propagate(T_eval, Ws, bs, **activations)
            snapshots[it+1] = A_curve[-1]

    # Final prediction
    A_curve, _ = nn.forward_propagate(T_eval, Ws, bs, **activations)
    Y_curve_pred = A_curve[-1]

    # Plot XY projection
    plt.figure(figsize=(7, 7))
    plt.plot(X_curve, Y_curve, label="True helix projection", color="black")  # black curve
    plt.scatter(x_random, y_random, color="red", s=15, alpha=0.6, label="Training points")  # smaller red points

    # Plot snapshots with low opacity
    for it, Y_snapshot in snapshots.items():
        plt.plot(Y_snapshot[0], Y_snapshot[1], alpha=0.3, label=f"Iter {it}")

    # Final prediction
    plt.plot(Y_curve_pred[0], Y_curve_pred[1], label="Final NN Prediction", color="green", linestyle="--")
    plt.xlabel("x = t cos t")
    plt.ylabel("y = t sin t")
    plt.title("Neural Network fitting helix projection")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    #plt.show()

    # Plot training loss
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss (helix test)")
    plt.grid(True)
    
    
    plt.show()
