import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.special import softmax as scipy_softmax
import seaborn as sns
import imageio
from tqdm import tqdm


def make_gif(predictions, certainties, targets, pre_activations, post_activations,
             attention, inputs_to_model, filename):
    """Animate CTM dynamics across internal ticks and save as a GIF.

    Renders one frame per tick showing:
    - Input image
    - Attention heatmap (spatial attention over CNN feature map)
    - Class probability bar chart (green = target class, blue = other)
    - Certainty curve with a vertical cursor at the current tick
    - 16 neuron traces (pre-activation in gray dashed, post-activation in red/blue)

    All inputs must be numpy arrays (call ``np.array(jax_tensor)`` before passing).

    Args:
        predictions:     (B, num_classes, T) logits at each tick.
        certainties:     (B, 2, T)  — column 1 is certainty (1 - norm_entropy).
        targets:         (B,) integer class labels.
        pre_activations: (T, B, d_model) neuron states before trace processor.
        post_activations:(T, B, d_model) neuron activations after trace processor.
        attention:       (T, B, heads, 1, num_positions) attention weights.
        inputs_to_model: (B, H, W, C) channels-last input images.
        filename:        Output GIF path (e.g. 'mnist_logs/prediction.gif').
    """
    batch_index         = 0
    n_neurons_to_vis    = 16
    figscale            = 0.28
    n_steps             = len(pre_activations)
    heatmap_cmap        = sns.color_palette('viridis', as_cmap=True)

    attention_maps = _reshape_attention(attention)

    these_pre_acts      = pre_activations[:, batch_index, :]
    these_post_acts     = post_activations[:, batch_index, :]
    these_inputs        = inputs_to_model[batch_index]              # (H, W, C)
    these_attn          = attention_maps[:, batch_index, :, :]
    these_predictions   = predictions[batch_index, :, :]
    these_certainties   = certainties[batch_index, :, :]
    this_target         = targets[batch_index]

    class_labels = [str(i) for i in range(these_predictions.shape[0])]

    mosaic = (
        [['img_data', 'img_data', 'attention', 'attention',
          'probs', 'probs', 'probs', 'probs']] * 4
        + [['certainty'] * 8]
        + [[f'trace_{ti}'] * 8 for ti in range(n_neurons_to_vis)]
    )

    frames = []
    for stepi in tqdm(range(n_steps), desc='Rendering frames', unit='frame'):
        fig, axes = plt.subplot_mosaic(
            mosaic=mosaic,
            figsize=(31 * figscale * 8 / 4, 76 * figscale),
        )

        # --- Probability bar chart ---
        probs  = scipy_softmax(these_predictions[:, stepi])
        colors = ['g' if i == this_target else 'b' for i in range(len(probs))]
        axes['probs'].bar(np.arange(len(probs)), probs, color=colors, width=0.9, alpha=0.5)
        axes['probs'].set_title('Probabilities')
        axes['probs'].set_xticks(np.arange(len(probs)))
        axes['probs'].set_xticklabels(class_labels, fontsize=24)
        axes['probs'].set_yticks([])
        axes['probs'].tick_params(left=False, bottom=False)
        axes['probs'].set_ylim([0, 1])
        for spine in axes['probs'].spines.values():
            spine.set_visible(False)

        # --- Certainty curve ---
        axes['certainty'].plot(np.arange(n_steps), these_certainties[1], 'k-', linewidth=2)
        axes['certainty'].set_xlim([0, n_steps - 1])
        axes['certainty'].axvline(x=stepi, color='black', linewidth=1, alpha=0.5)
        axes['certainty'].set_xticklabels([])
        axes['certainty'].set_yticklabels([])
        axes['certainty'].grid(False)
        for spine in axes['certainty'].spines.values():
            spine.set_visible(False)

        # --- Neuron traces ---
        for ni in range(n_neurons_to_vis):
            ax  = axes[f'trace_{ni}']
            pre = these_pre_acts[:, ni]
            post = these_post_acts[:, ni]
            ax_pre = ax.twinx()

            ax_pre.plot(np.arange(n_steps), pre,  color='grey',
                        linestyle='--', linewidth=1, alpha=0.4)
            color = 'blue' if ni % 2 else 'red'
            ax.plot(np.arange(n_steps), post, color=color, linewidth=2, alpha=1.0)

            for a in (ax, ax_pre):
                a.set_xlim([0, n_steps - 1])
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.grid(False)
                for spine in a.spines.values():
                    spine.set_visible(False)

            ax.set_ylim([np.min(post), np.max(post)])
            ax_pre.set_ylim([np.min(pre), np.max(pre)])
            ax.axvline(x=stepi, color='black', linewidth=1, alpha=0.5)

        # --- Input image (single channel, channels-last) ---
        img = these_inputs[:, :, 0]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes['img_data'].imshow(img, cmap='binary', vmin=0, vmax=1)
        axes['img_data'].set_title('Input Image')
        axes['img_data'].axis('off')

        # --- Attention heatmap ---
        gate = these_attn[stepi]
        g_min, g_max = np.nanmin(gate), np.nanmax(gate)
        if not np.isclose(g_min, g_max):
            gate = (gate - g_min) / (g_max - g_min + 1e-8)
        else:
            gate = np.zeros_like(gate)
        axes['attention'].imshow(heatmap_cmap(gate)[:, :, :3], vmin=0, vmax=1)
        axes['attention'].set_title('Attention')
        axes['attention'].axis('off')

        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        rgba = rgba.reshape(*reversed(fig.canvas.get_width_height()), 4)
        frames.append(rgba[:, :, :3])
        plt.close(fig)

    imageio.mimsave(filename, frames, fps=5, loop=100)
    print(f'GIF saved to {filename}')


def _reshape_attention(attention, target_size=28):
    """Average attention heads and resize spatial map to target_size × target_size.

    Args:
        attention:   (T, B, heads, 1, num_positions) raw attention weights.
        target_size: output spatial resolution (default 28 for MNIST).

    Returns:
        numpy array (T, B, target_size, target_size)
    """
    T, B, num_heads, _, num_positions = attention.shape
    attention = attention.mean(axis=2).squeeze(2)   # (T, B, num_positions)

    height = int(num_positions ** 0.5)
    while num_positions % height != 0:
        height -= 1
    width = num_positions // height
    attention = attention.reshape(T, B, height, width)

    result = np.zeros((T, B, target_size, target_size))
    for t in range(T):
        for b in range(B):
            result[t, b] = zoom(
                attention[t, b],
                (target_size / height, target_size / width),
                order=1,
            )
    return result
