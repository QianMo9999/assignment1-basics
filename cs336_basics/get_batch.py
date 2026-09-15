import numpy as np
import numpy.typing as npt
import torch

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.
    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    start = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x_np = []
    y_np = []
    for start_i in start:
        end = start_i.item() + context_length
        x_np.append(dataset[start_i.item() : end])
        y_np.append(dataset[start_i.item() + 1 : end + 1])

    x = torch.tensor(np.stack(x_np, axis=0), dtype=torch.long, device=device)
    y = torch.tensor(np.stack(y_np, axis=0), dtype=torch.long, device=device)

    return x, y