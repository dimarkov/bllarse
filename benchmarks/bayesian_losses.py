import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import time
import matplotlib.pyplot as plt
from jaxtyping import Array
from functools import partial

from bllarse.losses.bayesian import IBProbit, IBPolyaGamma, MultinomialPolyaGamma

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
    
    return X, y.astype(jnp.float32), probs, (W, b)

def benchmark_loss_vs_iterations(key):
    n_samples, n_features, n_classes = 1000, 100, 50
    X, y, *_ = generate_data(key, n_samples, n_features, n_classes)

    # Initialize models
    key, subkey = jr.split(key)
    ib_probit = IBProbit(n_features, n_classes, key=subkey)
    key, subkey = jr.split(key)
    ib_pg = IBPolyaGamma(n_features, n_classes, key=subkey)
    key, subkey = jr.split(key)
    mpg = MultinomialPolyaGamma(n_features, n_classes, key=subkey)

    n_iters_range = jnp.arange(1, 128)[::4]
    losses = {
        "IBProbit_0": [], "IBProbit_1": [], "IBProbit_2": [], "IBProbit_3": [],
        "IBPG_0": [], "IBPG_1": [], "IBPG_2": [], "IBPG_3": [],
        "MPG_0": [], "MPG_1": []
    }

    for n_iters in n_iters_range:
        for loss_type in range(4):
            # Update and compute loss for IBModels
            loss = ib_probit.update(X, y, num_iters=n_iters)(X, y, loss_type=loss_type).mean()
            losses[f"IBProbit_{loss_type}"].append(loss)
            
            loss = ib_pg.update(X, y, num_iters=n_iters)(X, y, loss_type=loss_type).mean()
            losses[f"IBPG_{loss_type}"].append(loss)

            # Update and compute loss for MultinomialPolyaGamma
            if loss_type < 2:
                loss = mpg.update(X, y, num_iters=n_iters)(X, y, loss_type=loss_type).mean()
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
    n_classes = 10
    n_runs = 5
    
    # Runtime vs. Data Size
    n_features = 100
    n_samples_range = jnp.arange(100, 1100, 100)
    runtimes_data = {"IBProbit": [], "IBPG": [], "MPG": []}

    ib_probit = IBProbit(n_features, n_classes, key=key)
    ib_pg = IBPolyaGamma(n_features, n_classes, key=key)
    mpg = MultinomialPolyaGamma(n_features, n_classes, key=key)

    for n_samples in n_samples_range:
        X, y, *_ = generate_data(key, n_samples, n_features, n_classes)

        func = eqx.filter_jit(ib_probit.update)
        func(X, y)
        start_time = time.time()
        for _ in range(n_runs):
            jax.block_until_ready(func(X, y).eta)
        runtimes_data["IBProbit"].append( (time.time() - start_time) / n_runs)

        func = eqx.filter_jit(ib_pg.update)
        func(X, y)
        start_time = time.time()
        for _ in range(n_runs):
            jax.block_until_ready(func(X, y).mu)
        runtimes_data["IBPG"].append( (time.time() - start_time) / n_runs )

        func = eqx.filter_jit(mpg.update)
        func(X, y)
        start_time = time.time()
        for _ in range(n_runs):
            jax.block_until_ready(func(X, y).mu)
        runtimes_data["MPG"].append( (time.time() - start_time) / n_runs)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    for name, rt in runtimes_data.items():
        plt.plot(n_samples_range, rt, label=name)
    plt.xlabel("Data Size")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs. Data Size")
    plt.legend()
    plt.grid(True)

    # Runtime vs. Feature Size
    n_samples = 1024
    n_features_range = jnp.arange(100, 800, 100)
    runtimes_features = {"IBProbit": [], "IBPG": [], "MPG": []}

    for n_features in n_features_range:
        X, y, *_ = generate_data(key, n_samples, n_features, n_classes)
        
        ib_probit = IBProbit(n_features, n_classes, key=key)
        ib_pg = IBPolyaGamma(n_features, n_classes, key=key)
        mpg = MultinomialPolyaGamma(n_features, n_classes, key=key)

        func = eqx.filter_jit(ib_probit.update)
        func(X, y)
        start_time = time.time()
        for _ in range(n_runs):
            jax.block_until_ready(func(X, y).eta)
        runtimes_features["IBProbit"].append((time.time() - start_time) / n_runs)

        func = eqx.filter_jit(ib_pg.update)
        func(X, y)
        start_time = time.time()
        for _ in range(n_runs):
            jax.block_until_ready(func(X, y).mu)
        runtimes_features["IBPG"].append( (time.time() - start_time) / n_runs)

        func = eqx.filter_jit(mpg.update)
        func(X, y)
        start_time = time.time()
        for _ in range(n_runs):
            jax.block_until_ready( func(X, y).mu )
        runtimes_features["MPG"].append( (time.time() - start_time) / n_runs)

    plt.subplot(1, 3, 2)
    for name, rt in runtimes_features.items():
        plt.plot(n_features_range, rt, label=name)
    plt.xlabel("Feature Size")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs. Feature Size")
    plt.legend()
    plt.grid(True)

    # Runtime vs. Number of Classes
    n_samples = 1024
    n_features = 768
    n_classes_range = jnp.arange(10, 101, 20)
    runtimes_classes = {"IBProbit": [], "IBPG": [], "MPG": []}

    for n_classes in n_classes_range:
        X, y, *_ = generate_data(key, n_samples, n_features, n_classes)
        
        ib_probit = IBProbit(n_features, n_classes, key=key)
        ib_pg = IBPolyaGamma(n_features, n_classes, key=key)
        mpg = MultinomialPolyaGamma(n_features, n_classes, key=key)

        func = eqx.filter_jit(ib_probit.update)
        func(X, y)
        start_time = time.time()
        for _ in range(n_runs):
            jax.block_until_ready(func(X, y).eta)
        runtimes_classes["IBProbit"].append((time.time() - start_time) / n_runs)

        func = eqx.filter_jit(ib_pg.update)
        func(X, y)
        start_time = time.time()
        for _ in range(n_runs):
            jax.block_until_ready(func(X, y).mu)
        runtimes_classes["IBPG"].append( (time.time() - start_time) / n_runs)

        func = eqx.filter_jit(mpg.update)
        func(X, y)
        start_time = time.time()
        for _ in range(n_runs):
            jax.block_until_ready( func(X, y).mu )
        runtimes_classes["MPG"].append( (time.time() - start_time) / n_runs)

    plt.subplot(1, 3, 3)
    for name, rt in runtimes_classes.items():
        plt.plot(n_classes_range, rt, label=name)
    plt.xlabel("Number of Classes")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs. Number of Classes")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("learning_runtimes.png")
    plt.close()


def benchmark_gradient_runtimes(key):
    n_classes = 10
    n_runs = 5

    n_features_range = range(100, 801, 100)
    loss_types = {
        "IBProbit": [1, 2, 3],
        "IBPG": [1, 2, 3],
        "MPG": [0]
    }
    
    runtimes = {f"{model}_{loss_type}": [] for model in loss_types for loss_type in loss_types[model]}
    runtimes['NN'] = []
    batches = [256]
    for n_samples in batches:
        for n_features in n_features_range:
            X, y, *_ = generate_data(key, n_samples, n_features, n_classes)
            
            # IBProbit
            ib_probit = IBProbit(n_features, n_classes, key=key)
            for loss_type in loss_types["IBProbit"]:
                def loss_fn_probit(x):
                    return ib_probit(x, y, loss_type=loss_type).mean()
                
                grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn_probit))
                grad_fn(X) # JIT compilation
                start_time = time.time()
                for _ in range(n_runs):
                    jax.block_until_ready(grad_fn(X))
                runtimes[f"IBProbit_{loss_type}"].append(1000 * (time.time() - start_time) / n_runs)
                print(n_features, f"IBProbit_{loss_type}", runtimes[f"IBProbit_{loss_type}"][-1])

            # IBPG
            ib_pg = IBPolyaGamma(n_features, n_classes, key=key)
            for loss_type in loss_types["IBPG"]:
                def loss_fn_ibpg(x):
                    return ib_pg(x, y, loss_type=loss_type).mean()

                grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn_ibpg))
                grad_fn(X) # JIT compilation
                start_time = time.time()
                for _ in range(n_runs):
                    jax.block_until_ready(grad_fn(X))
                runtimes[f"IBPG_{loss_type}"].append(1000 * (time.time() - start_time) / n_runs)
                print(n_features, f"IBPG_{loss_type}", runtimes[f"IBPG_{loss_type}"][-1])

            # MPG
            mpg = MultinomialPolyaGamma(n_features, n_classes, key=key)
            for loss_type in loss_types["MPG"]:
                def loss_fn_mpg(x):
                    return mpg(x, y, loss_type=loss_type).mean()
                
                grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn_mpg))
                grad_fn(X) # JIT compilation
                start_time = time.time()
                for _ in range(n_runs):
                    jax.block_until_ready(grad_fn(X))
                runtimes[f"MPG_{loss_type}"].append(1000 * (time.time() - start_time) / n_runs)
                print(n_features, f"MPG_{loss_type}", runtimes[f"MPG_{loss_type}"][-1])

            # Linear layer
            nn_lin = eqx.nn.Linear(in_features=n_features, out_features=n_classes, key=key)
            def loss_lin(x):
                logits = jax.vmap(nn_lin)(x)
                return optax.safe_softmax_cross_entropy(logits, jax.nn.one_hot(y, n_classes)).mean()

            grad_fn = jax.jit(jax.value_and_grad(loss_lin))
            grad_fn(X)
            start_time = time.time()
            for _ in range(n_runs):
                jax.block_until_ready(grad_fn(X))
            runtimes[f"NN"].append(1000 * (time.time() - start_time) / n_runs)
            print(n_features, f"NN", runtimes[f"NN"][-1])


    plt.figure(figsize=(12, 8))
    for name, rt in runtimes.items():
        plt.plot(n_features_range, rt, label=name)
    
    plt.xlabel("Feature Size")
    plt.ylabel("Gradient Runtime (ms)")
    plt.title("Gradient Runtime vs. Feature Size for different loss types")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gradient_runtimes.png")
    plt.close()


def benchmark_accuracy(key):
    n_samples, n_features, n_classes = 10_000, 200, 100
    key, subkey = jr.split(key)
    X_, y_, probs_train, (W_true, b_true) = generate_data(subkey, n_samples + 1000, n_features, n_classes)
    X_train = X_[:n_samples]
    y_train = y_[:n_samples]
    X_test, y_test = X_[n_samples:], y_[n_samples:]
    
    # --- Bayesian Models ---
    key, subkey = jr.split(key)
    ib_probit = IBProbit(n_features, n_classes, key=subkey)
    updated_ib_probit = eqx.filter_jit(ib_probit.update)(X_train, y_train, num_iters=128)
    accs = []
    for loss_type in [0, 1, 2]:
        _, logits_ib = updated_ib_probit(X_test, y_test, with_logits=True, loss_type=loss_type)
        accs.append( (jnp.argmax(logits_ib, -1) == y_test).mean() )
    print(f"IBProbit Accuracy: {accs[0]:.4f}-{accs[1]:.4f}-{accs[2]:.4f}")

    key, subkey = jr.split(key)
    ib_pg = IBPolyaGamma(n_features, n_classes, key=subkey)
    updated_ib_pg = eqx.filter_jit(ib_pg.update)(X_train, y_train, num_iters=128)

    accs = []
    for loss_type in [1, 2]:
        _, logits_ib = updated_ib_pg(X_test, y_test, with_logits=True, loss_type=loss_type)
        accs.append( (jnp.argmax(logits_ib, -1) == y_test).mean() )
    print(f"IBPG Accuracy: {accs[0]:.4f}-{accs[1]:.4f}")

    key, subkey = jr.split(key)
    mpg = MultinomialPolyaGamma(n_features, n_classes, key=subkey)
    updated_mpg = eqx.filter_jit(mpg.update)(X_train, y_train, num_iters=128)
    accs = []
    for loss_type in [0, 1]:
        _, logits_mpg = updated_mpg(X_test, y_test, with_logits=True)
        accs.append( (jnp.argmax(logits_mpg, -1) == y_test).mean() )
    print(f"MultinomialPolyaGamma Accuracy: {accs[0]:.4f}-{accs[1]:.4f}")

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
        return optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, logits.shape[-1])).mean()

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

    pos = jnp.arange(n_samples)
    for step in range(1, 501):
        key, subkey = jr.split(key)
        idxs = jr.choice(subkey, pos, shape=(512,))
        model, opt_state, loss = make_step(model, opt_state, X_train[idxs], y_train[idxs])
        if step % 100 == 0:
            acc = accuracy_fn(model, X_test, y_test)
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

    print("\n--- Benchmarking Gradient Runtimes ---")
    benchmark_gradient_runtimes(main_key)
    print("Plot saved to gradient_runtimes.png")

    print("\n--- Benchmarking Accuracy ---")
    benchmark_accuracy(main_key)
