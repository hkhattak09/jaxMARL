import jax
import jax.numpy as jnp
from jax import random as jrandom
import optax
from flax.training import train_state
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from tqdm import tqdm

from .loss import get_loss, calculate_accuracy
from .data import torch_to_jax


# ---------------------------------------------------------------------------
# Extended TrainState carrying BatchNorm statistics
# ---------------------------------------------------------------------------

class TrainState(train_state.TrainState):
    """Flax TrainState extended with mutable batch_stats for BatchNorm.

    Flax's standard TrainState only stores params + opt_state.
    BatchNorm requires mutable running statistics (mean/variance) that must
    be threaded through every training step separately.
    """
    batch_stats: any


# ---------------------------------------------------------------------------
# JIT-compiled training step
# ---------------------------------------------------------------------------

@jax.jit
def train_step(state, images, labels):
    """One gradient update step, fully JIT-compiled by XLA.

    The CTM's internal recurrent loop is a plain Python for-loop that XLA
    unrolls at trace time into a single fused computation graph.

    Dropout randomness is derived from the global step counter via
    ``jrandom.fold_in(PRNGKey(0), state.step)``, giving deterministic but
    step-varying randomness with zero extra state.

    After the gradient update, decay parameters are clamped to [0, 15]
    so that ``exp(-decay)`` stays in ``[exp(-15), 1]``.

    Args:
        state:  TrainState (params, batch_stats, optimizer state, step).
        images: (B, H, W, C) input images — channels-last.
        labels: (B,) integer class labels.

    Returns:
        state:              Updated TrainState.
        loss:               Scalar training loss.
        accuracy:           Scalar accuracy at most-certain tick.
        where_most_certain: (B,) per-sample tick indices.
    """
    dropout_rng = jrandom.fold_in(jrandom.PRNGKey(0), state.step)

    def loss_fn(params):
        (predictions, certainties, _), updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            images,
            deterministic=False,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng},
        )
        loss, where_most_certain = get_loss(predictions, certainties, labels)
        return loss, (predictions, certainties, where_most_certain, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (predictions, certainties, where_most_certain, updates)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(
        batch_stats=updates['batch_stats'],
        params={
            **state.params,
            'decay_params_action': jnp.clip(state.params['decay_params_action'], 0.0, 15.0),
            'decay_params_out':    jnp.clip(state.params['decay_params_out'],    0.0, 15.0),
        },
    )

    accuracy = calculate_accuracy(predictions, labels, where_most_certain)
    return state, loss, accuracy, where_most_certain


# ---------------------------------------------------------------------------
# JIT-compiled evaluation step
# ---------------------------------------------------------------------------

@jax.jit
def eval_step(state, images, labels):
    """One evaluation step with no gradient computation.

    BatchNorm uses running statistics (deterministic=True).
    Dropout is disabled.

    Args:
        state:  TrainState.
        images: (B, H, W, C).
        labels: (B,).

    Returns:
        loss:               Scalar CE loss.
        predictions:        (B, num_classes, T) logits.
        certainties:        (B, 2, T).
        where_most_certain: (B,) tick indices.
    """
    predictions, certainties, _ = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        images,
        deterministic=True,
    )
    loss, where_most_certain = get_loss(predictions, certainties, labels)
    return loss, predictions, certainties, where_most_certain


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(state, trainloader, testloader, iterations, test_every):
    """Main training loop with periodic evaluation and live loss/accuracy plots.

    Cycles through the DataLoader indefinitely until ``iterations`` steps
    are completed. Evaluates on the full test set every ``test_every`` steps
    and at the final step.

    Args:
        state:       Initial TrainState.
        trainloader: PyTorch DataLoader for training data.
        testloader:  PyTorch DataLoader for test data.
        iterations:  Total number of gradient steps.
        test_every:  Evaluate every this many steps.

    Returns:
        state: Updated TrainState after training.
    """
    iterator = iter(trainloader)

    train_losses      = []
    test_losses       = []
    train_accuracies  = []
    test_accuracies   = []
    steps             = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    test_loss     = None
    test_accuracy = None

    with tqdm(total=iterations, dynamic_ncols=True) as pbar:
        for stepi in range(iterations):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets = next(iterator)

            images, labels = torch_to_jax(inputs, targets)

            state, t_loss, t_acc, _ = train_step(state, images, labels)
            train_losses.append(float(t_loss))
            train_accuracies.append(float(t_acc))

            if stepi % test_every == 0 or stepi == iterations - 1:
                all_test_losses  = []
                all_test_preds   = []
                all_test_targets = []
                all_test_where   = []

                for test_inputs, test_targets in testloader:
                    test_images, test_labels = torch_to_jax(test_inputs, test_targets)
                    e_loss, e_preds, e_certs, e_where = eval_step(
                        state, test_images, test_labels
                    )
                    all_test_losses.append(float(e_loss))
                    all_test_preds.append(e_preds)
                    all_test_targets.append(test_labels)
                    all_test_where.append(e_where)

                all_test_preds   = jnp.concatenate(all_test_preds,   axis=0)
                all_test_targets = jnp.concatenate(all_test_targets, axis=0)
                all_test_where   = jnp.concatenate(all_test_where,   axis=0)

                test_accuracy = float(
                    calculate_accuracy(all_test_preds, all_test_targets, all_test_where)
                )
                test_loss = sum(all_test_losses) / len(all_test_losses)

                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
                steps.append(stepi)

                _update_plot(
                    fig, ax1, ax2,
                    train_losses, test_losses,
                    train_accuracies, test_accuracies,
                    steps,
                )

            pbar.set_description(
                f'Train Loss: {t_loss:.3f}  Train Acc: {t_acc:.3f}  '
                f'Test Loss: {test_loss}  Test Acc: {test_accuracy}'
            )
            pbar.update(1)

    plt.ioff()
    plt.close(fig)
    return state


def _update_plot(fig, ax1, ax2, train_losses, test_losses,
                 train_accuracies, test_accuracies, steps):
    clear_output(wait=True)

    ax1.clear()
    ax1.plot(range(len(train_losses)), train_losses, 'b-', alpha=0.7,
             label=f'Train Loss: {train_losses[-1]:.3f}')
    ax1.plot(steps, test_losses, 'r-', marker='o',
             label=f'Test Loss: {test_losses[-1]:.3f}')
    ax1.set_title('Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.clear()
    ax2.plot(range(len(train_accuracies)), train_accuracies, 'b-', alpha=0.7,
             label=f'Train Accuracy: {train_accuracies[-1]:.3f}')
    ax2.plot(steps, test_accuracies, 'r-', marker='o',
             label=f'Test Accuracy: {test_accuracies[-1]:.3f}')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    display(fig)


def create_train_state(model, master_key, learning_rate=1e-4, eps=1e-8,
                       dummy_input_shape=(1, 28, 28, 1)):
    """Convenience factory: initialize model and create TrainState.

    Args:
        model:              Instantiated ContinuousThoughtMachine.
        master_key:         JAX PRNG key.
        learning_rate:      AdamW learning rate.
        eps:                AdamW epsilon.
        dummy_input_shape:  Shape for parameter initialization (B, H, W, C).

    Returns:
        state: TrainState ready for training.
    """
    dummy_input = jnp.ones(dummy_input_shape)
    variables   = model.init(master_key, dummy_input, deterministic=True)

    optimizer = optax.adamw(learning_rate=learning_rate, eps=eps)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        batch_stats=variables['batch_stats'],
    )
    return state
