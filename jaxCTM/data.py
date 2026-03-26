import jax.numpy as jnp
import torch.utils.data
from torchvision import datasets, transforms


def prepare_data(batch_size=256):
    """MNIST data loaders via torchvision.

    Training set uses random affine + rotation augmentation.
    Test set is unaugmented for consistent evaluation.

    Images are kept as PyTorch tensors (B, C, H, W); use ``torch_to_jax``
    to convert each batch to channels-last JAX arrays before model input.

    Args:
        batch_size: samples per batch.

    Returns:
        trainloader: shuffled, augmented DataLoader (drop_last=True).
        testloader:  unshuffled, unaugmented DataLoader (drop_last=False).
    """
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST(
        root='./data', train=True,  download=True, transform=train_transform
    )
    test_data  = datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )
    return trainloader, testloader


def torch_to_jax(inputs, targets):
    """Convert a PyTorch batch to JAX arrays.

    PyTorch images are channels-first (B, C, H, W).
    JAX convolutions expect channels-last (B, H, W, C).

    Args:
        inputs:  torch.Tensor (B, C, H, W)
        targets: torch.Tensor (B,)

    Returns:
        images: jnp.ndarray (B, H, W, C)
        labels: jnp.ndarray (B,) int32
    """
    images = jnp.array(inputs.numpy()).transpose(0, 2, 3, 1)
    labels = jnp.array(targets.numpy(), dtype=jnp.int32)
    return images, labels
