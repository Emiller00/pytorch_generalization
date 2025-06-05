# Tools for computing input margins as proposed in Mouton et al.
import torch
import numpy as np


def get_input_margin(model, X, y):
    """Compute input margins for a batch as proposed by Mouton et al.

    This follows "Input margins can predict generalization too" (Mouton,
    Barrett and Dinh, 2022) which defines the input margin as the
    difference between the true logit and the closest competing logit,
    normalized by the gradient of that difference with respect to the
    input. The resulting value can be used as a generalization indicator.

    Args:
        model (torch.nn.Module): trained model.
        X (torch.Tensor): input batch, requires grad.
        y (torch.Tensor): target labels.

    Returns:
        np.ndarray: margin values for each sample in the batch.
    """
    model.eval()

    batch_size = X.shape[0]

    # Forward pass
    output = model(X)

    num_class = output.shape[1]
    values, indices = torch.topk(output, k=2)
    values_true = torch.gather(output, 1, y.view(-1, 1)).squeeze(1)

    true_match = (indices[:, 0] == y).float()
    values_c = values[:, 1] * true_match + values[:, 0] * (1 - true_match)
    indices_c = indices[:, 1] * true_match + indices[:, 0] * (1 - true_match)

    numerator = values_true - values_c

    grad_ys = torch.nn.functional.one_hot(y.long(), num_class)
    grad_ys -= torch.nn.functional.one_hot(indices_c.long(), num_class)

    g = torch.autograd.grad(outputs=output,
                            inputs=X,
                            grad_outputs=grad_ys,
                            retain_graph=True)[0]

    norm = torch.norm(g.view(batch_size, -1), dim=1)
    margin = numerator / norm

    return margin.detach().cpu().numpy()
