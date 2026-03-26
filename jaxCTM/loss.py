import jax.numpy as jnp
import optax


def get_loss(predictions, certainties, targets, use_most_certain=True):
    """Certainty-weighted CTM loss.

    Averages two cross-entropy terms:
      1. Loss at the tick where the model's prediction is best (lowest CE).
      2. Loss at the tick where the model is most certain (highest 1-entropy).

    This dual objective teaches the model to be confident exactly when it is
    correct — confidence at a wrong tick is penalized by term 2, and accuracy
    at an uncertain tick is penalized by term 1.

    Args:
        predictions:     (B, num_classes, T) logits at each internal tick.
        certainties:     (B, 2, T) where [:, 1, :] = certainty (1 - norm_entropy).
        targets:         (B,) integer class labels.
        use_most_certain: If False, always use the final tick for term 2.

    Returns:
        loss:            scalar — mean of the two CE terms.
        loss_index_2:    (B,) indices of the selected tick per sample.
    """
    B, C, T = predictions.shape

    # Cross-entropy at every tick: (B, T)
    preds_btc  = jnp.transpose(predictions, (0, 2, 1))          # (B, T, C)
    targets_rep = jnp.broadcast_to(targets[:, None], (B, T))    # (B, T)
    losses = optax.softmax_cross_entropy_with_integer_labels(preds_btc, targets_rep)  # (B, T)

    loss_index_1 = jnp.argmin(losses, axis=1)  # best prediction tick per sample

    if use_most_certain:
        loss_index_2 = jnp.argmax(certainties[:, 1, :], axis=-1)
    else:
        loss_index_2 = jnp.full((B,), T - 1, dtype=jnp.int32)

    batch_idx = jnp.arange(B)
    loss_minimum_ce = jnp.mean(losses[batch_idx, loss_index_1])
    loss_selected   = jnp.mean(losses[batch_idx, loss_index_2])

    loss = (loss_minimum_ce + loss_selected) / 2.0
    return loss, loss_index_2


def calculate_accuracy(predictions, targets, where_most_certain):
    """Accuracy at the most-certain internal tick.

    Args:
        predictions:       (B, num_classes, T) logits.
        targets:           (B,) integer class labels.
        where_most_certain: (B,) tick index per sample.

    Returns:
        accuracy: scalar float in [0, 1].
    """
    B = predictions.shape[0]
    pred_classes   = jnp.argmax(predictions, axis=1)            # (B, T)
    selected_preds = pred_classes[jnp.arange(B), where_most_certain]
    return jnp.mean(selected_preds == targets)
