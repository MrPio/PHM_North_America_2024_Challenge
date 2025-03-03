# The network doesn't learn when `batch_size` > 1

In the lines, a Gaussian NLL Loss is in use. Due to PyTorch operations Broadcasting, you have to make sure that `mu`, `y` and `log_var` tensors have the same shape!
```python
    mu, log_var = model(x)
    loss = criterion(mu, y.squeeze(), torch.exp(log_var))
```
Here I had `squeeze()` `y` to change its shape from `(batch_size, )` to `(batch_size, 1)`. 