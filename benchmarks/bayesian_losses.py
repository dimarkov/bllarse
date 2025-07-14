import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import time
import matplotlib.pyplot as plt
from jaxtyping import Array
from functools import partial

from bllarse.losses.bayesian import IBProbit, MultinomialPolyaGamma

def generate_data(key, n_samples, n_features, n_classes):
    """Generate synthetic data for multinomial logistic regression."""
    key, subkey = jr.split(key)
    params = jr.normal(subkey, (n_features + 1, n_classes))
    W = params[:-1]
    b = params[-1]
    
    key, subkey = jr.split(key)
    X = jr.normal(subkey, (n_samples, n_features))
    
    logits = X @ W + b
    probs = jax.nn.softmax(logits, axis=-1)
    
    key, subkey = jr.split(key)
    y = jr.categorical(subkey, logits=logits, axis=-1)
    
    return X, y, probs, (W, b)

def benchmark_loss_vs_iterations(key):
    n_samples, n_features, n_classes = 1000, 10, 3
    X, y, *_ = generate_data(key, n_samples, n_features, n_classes)

    # Initialize models
    key, subkey = jr.split(key)
    ib_probit = IBProbit(n_features, n_classes, key=subkey)
    key, subkey = jr.split(key)
    mpg = MultinomialPolyaGamma(n_features, n_classes, key=subkey)

    n_iters_range = jnp.arange(1, 32)[::2]
    losses = {
        "IBProbit_0": [], "IBProbit_1": [], "IBProbit_2": [], "IBProbit_3": [],
        "MPG_0": [], "MPG_1": []
    }

    for n_iters in n_iters_range:
        # Update and compute loss for IBProbit
        for loss_type in range(4):
            updated_ib_probit = ib_probit.update(X, y, num_iters=n_iters)
            loss = updated_ib_probit(X, y, loss_type=loss_type).mean()
            losses[f"IBProbit_{loss_type}"].append(loss)

        # Update and compute loss for MultinomialPolyaGamma
        for loss_type in range(2):
            updated_mpg = mpg.update(X, y, num_iters=n_iters)
            loss = updated_mpg(X, y, loss_type=loss_type).mean()
            losses[f"MPG_{loss_type}"].append(loss)

    # Plotting
    plt.figure(figsize=(12, 8))
    for name, loss_values in losses.items():
        plt.plot(n_iters_range, loss_values, label=name)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.title("Loss vs. Number of Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_vs_iterations.png")
    plt.close()

def benchmark_runtimes(key):
    n_classes = 5
    
    # Runtime vs. Data Size
    n_features = 20
    n_samples_range = jnp.arange(100, 1100, 100)
    runtimes_data = {"IBProbit": [], "MPG": []}

    for n_samples in n_samples_range:
        X, y, *_ = generate_data(key, n_samples, n_features, n_classes)
        
        key, subkey = jr.split(key)
        ib_probit = IBProbit(n_features, n_classes, key=subkey)
        key, subkey = jr.split(key)
        mpg = MultinomialPolyaGamma(n_features, n_classes, key=subkey)

        start_time = time.time()
        ib_probit = ib_probit.update(X, y)
        jax.block_until_ready(ib_probit)
        runtimes_data["IBProbit"].append(time.time() - start_time)

        start_time = time.time()
        mpg = mpg.update(X, y)
        jax.block_until_ready(mpg)
        runtimes_data["MPG"].append(time.time() - start_time)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for name, rt in runtimes_data.items():
        plt.plot(n_samples_range, rt, label=name)
    plt.xlabel("Data Size")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs. Data Size")
    plt.legend()
    plt.grid(True)

    # Runtime vs. Feature Size
    n_samples = 500
    n_features_range = jnp.arange(10, 110, 10)
    runtimes_features = {"IBProbit": [], "MPG": []}

    for n_features in n_features_range:
        X, y, *_ = generate_data(key, n_samples, n_features, n_classes)
        
        key, subkey = jr.split(key)
        ib_probit = IBProbit(n_features, n_classes, key=subkey)
        key, subkey = jr.split(key)
        mpg = MultinomialPolyaGamma(n_features, n_classes, key=subkey)

        start_time = time.time()
        ib_probit = ib_probit.update(X, y)
        jax.block_until_ready(ib_probit)
        runtimes_features["IBProbit"].append(time.time() - start_time)

        start_time = time.time()
        mpg = mpg.update(X, y)
        jax.block_until_ready(mpg)
        runtimes_features["MPG"].append(time.time() - start_time)

    plt.subplot(1, 2, 2)
    for name, rt in runtimes_features.items():
        plt.plot(n_features_range, rt, label=name)
    plt.xlabel("Feature Size")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs. Feature Size")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("runtimes.png")
    plt.close()

def benchmark_accuracy(key):
    n_samples, n_features, n_classes = 10_000, 20, 5
    key, subkey = jr.split(key)
    X_, y_, probs_train, (W_true, b_true) = generate_data(subkey, n_samples + 1000, n_features, n_classes)
    X_train = X_[:n_samples]
    y_train = y_[:n_samples]
    X_test, y_test = X_[n_samples:], y_[n_samples:]
    
    # --- Bayesian Models ---
    key, subkey = jr.split(key)
    ib_probit = IBProbit(n_features, n_classes, key=subkey)
    updated_ib_probit = ib_probit.update(X_train, y_train)
    _, logits_ib = updated_ib_probit(X_test, y_test, with_logits=True)
    acc_ib = (jnp.argmax(logits_ib, -1) == y_test).mean()
    print(f"IBProbit Accuracy: {acc_ib:.4f}")

    key, subkey = jr.split(key)
    mpg = MultinomialPolyaGamma(n_features, n_classes, key=subkey)
    updated_mpg = mpg.update(X_train, y_train)
    _, logits_mpg = updated_mpg(X_test, y_test, with_logits=True)
    acc_mpg = (jnp.argmax(logits_mpg, -1) == y_test).mean()
    print(f"MultinomialPolyaGamma Accuracy: {acc_mpg:.4f}")

    # --- Gradient-based Multinomial Regression ---
    class MultinomialRegression(eqx.Module):
        weights: Array
        bias: Array

        def __init__(self, n_features, n_classes, key):
            self.weights = jr.normal(key, (n_features, n_classes)) * 0.01
            self.bias = jnp.zeros(n_classes)

        def __call__(self, x):
            return x @ self.weights + self.bias

    @eqx.filter_jit
    def loss_fn(model, x, y):
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @eqx.filter_jit
    def accuracy_fn(model, x, y):
        pred_y = model(x)
        return (jnp.argmax(pred_y, -1) == y).mean()

    key, subkey = jr.split(key)
    model = MultinomialRegression(n_features, n_classes, subkey)
    optimizer = optax.adamw(1e-2, weight_decay=1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for step in range(201):
        model, opt_state, loss = make_step(model, opt_state, X_train, y_train)
        if step % 50 == 0:
            acc = accuracy_fn(model, X_train, y_train)
            print(f"Step {step}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    final_acc = accuracy_fn(model, X_test, y_test)
    print(f"Gradient-based Multinomial Regression Accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main_key = jr.PRNGKey(42)
    
    print("--- Benchmarking Loss vs. Iterations ---")
    benchmark_loss_vs_iterations(main_key)
    print("Plot saved to loss_vs_iterations.png")

    print("\n--- Benchmarking Runtimes ---")
    benchmark_runtimes(main_key)
    print("Plot saved to runtimes.png")

    print("\n--- Benchmarking Accuracy ---")
    benchmark_accuracy(main_key)
