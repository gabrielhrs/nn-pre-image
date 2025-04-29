from typing import Tuple, Union

import numpy as np
import scipy.linalg
import torch

np.set_printoptions(linewidth=1000)
torch.set_printoptions(linewidth=1000)


def print_without_zeros(x: np.ndarray) -> None:
    nanstr = np.get_printoptions()['nanstr']
    np.set_printoptions(nanstr="")
    print(x)
    np.set_printoptions(nanstr=nanstr)


def np_to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).type(torch.FloatTensor)


def vec(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, (-1, 1))


def build_ind_matrix(num_in_channels: int,
                     in_height: int,
                     in_width: int,
                     kernel_size: Union[int, Tuple[int, ...]],
                     stride: Union[int, Tuple[int, ...]]) -> np.ndarray:
    x_shape = (num_in_channels, in_height, in_width)
    x_size = int(np.prod(x_shape))
    x_inds = np.reshape(np.arange(x_size), x_shape)
    x_inds_torch = np_to_torch(x_inds[None, :, :])

    unfold = torch.nn.Unfold(kernel_size=kernel_size,
                             stride=stride)

    unfolded_x_inds = unfold(x_inds_torch).detach().numpy().astype(int)[0, :, :]
    return unfolded_x_inds


def build_implied_weight_matrix(w: np.ndarray,
                                b: np.ndarray,
                                c_shape: Tuple[int, ...],
                                stride: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    num_out_channels = w.shape[0]
    num_in_channels = w.shape[1]
    kernel_height = w.shape[2]
    kernel_width = w.shape[3]

    kernel_size = (kernel_height, kernel_width)

    _, output_h, output_w = c_shape
    assert _ == num_out_channels

    lhs_size = num_out_channels * output_h * output_w
    in_height = safe_int(stride[0] * (output_h - 1) + kernel_height)
    in_width = safe_int(stride[1] * (output_w - 1) + kernel_width)

    x_shape = (num_in_channels, in_height, in_width)
    x_size = int(np.prod(x_shape))
    weight_flat = np.reshape(w, (num_out_channels, -1))

    ind_matrix = build_ind_matrix(num_in_channels,
                                  in_height,
                                  in_width,
                                  kernel_size,
                                  stride)
    implied_weight_matrix = np.zeros((lhs_size, x_size))
    implied_bias_vector = np.zeros((lhs_size, 1))

    offset = output_h * output_w

    for ir in range(ind_matrix.shape[1]):
        # ir = 0
        corresp_cols = ind_matrix[:, ir]
        for ic in range(num_out_channels):
            row = (offset * ic) + ir
            implied_weight_matrix[row, corresp_cols] = weight_flat[ic, :]
            implied_bias_vector[row] = b[ic]

    return implied_weight_matrix, implied_bias_vector

#
# def _in_size(out_size: int,
#              s: int,
#              k: int)
#     in_size = (out_size - 1) * s + (k - 1)
#     return in_size


def conv2d_inversion_kernel(c: np.ndarray,
                            w: np.ndarray,
                            b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_out_channels = w.shape[0]
    num_in_channels = w.shape[1]
    kernel_height = w.shape[2]
    kernel_width = w.shape[3]

    c_shape = c.shape
    _, output_h, output_w = c_shape
    assert _ == num_out_channels

    kernel_size = (kernel_height, kernel_width)

    c_flat = vec(c)

    in_height = (output_h - 1) + kernel_size[0]
    in_width = (output_w - 1) + kernel_size[1]

    x_shape = (num_in_channels, in_height, in_width)
    stride_height = 1
    stride_width = 1

    stride = (stride_height, stride_width)
    implied_weight_matrix, \
    implied_bias_vector = build_implied_weight_matrix(w,
                                                      b,
                                                      c_shape,
                                                      stride)

    particular_solution_flat = np.linalg.pinv(implied_weight_matrix) @ c_flat
    particular_solution = np.reshape(particular_solution_flat, x_shape)

    nullspace_basis_flat = scipy.linalg.null_space(implied_weight_matrix)
    nullspace_dim = nullspace_basis_flat.shape[1]
    nullspace_basis = np.reshape(nullspace_basis_flat, (*x_shape, nullspace_dim))
    return particular_solution, nullspace_basis


def build_avgpool_weights(kernel_size: Tuple[int, int],
                          num_channels: int) -> np.ndarray:
    kernel_height = kernel_size[0]
    kernel_width = kernel_size[1]
    one_mat = np.ones((kernel_height, kernel_width))
    eye_in_channels = np.eye(num_channels)

    to_split = np.kron(eye_in_channels, one_mat)
    split_once = np.split(to_split, num_channels, axis=1)
    split_twice = [np.split(_, num_channels, axis=0) for _ in split_once]
    w_new_np = np.array(split_twice) / (kernel_width * kernel_height)
    return w_new_np


def safe_int(n: float) -> int:
    intn = int(n)
    assert intn == n
    return intn


if __name__ == "__main__":
    # num_in_channels = 1
    num_in_channels = 3
    num_out_channels = num_in_channels

    kernel_height = 2
    kernel_width = 2

    stride_height = 2
    stride_width = 2

    kernel_size = (kernel_height, kernel_width)
    stride = (stride_height, stride_width)

    in_height = 4
    in_width = 4

    # x = torch.arange(1, 17).view(-1, 1, 4, 4).float()
    x = torch.arange(1, 1 + num_in_channels * in_height * in_width).view(1,
                                                                         num_in_channels,
                                                                         in_height,
                                                                         in_width).float()
    w = build_avgpool_weights(kernel_size,
                              num_out_channels)

    # formulae at: https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d
    # with padding == 0, dilation == 1
    out_height = safe_int(((in_height - (kernel_size[0] - 1) - 1) / stride[0] + 1))
    out_width = safe_int(((in_width - (kernel_size[1] - 1) - 1) / stride[1] + 1))

    ind_matrix = build_ind_matrix(num_in_channels,
                                  in_height,
                                  in_width,
                                  kernel_size,
                                  stride)

    lhs_size = num_out_channels * out_height * out_width

    x_shape = (num_in_channels, in_height, in_width)
    x_size = int(np.prod(x_shape))
    weight_flat = np.reshape(w, (num_out_channels, -1))

    implied_weight_matrix = np.zeros((lhs_size, x_size))

    offset = out_height * out_width

    for ir in range(ind_matrix.shape[1]):
        # ir = 0
        corresp_cols = ind_matrix[:, ir]
        for ic in range(num_out_channels):
            row = (offset * ic) + ir
            implied_weight_matrix[row, corresp_cols] = weight_flat[ic, :]

    x = torch.arange(0, in_height * in_width).view(in_height, in_width).float()
    x_np = x.detach().numpy()
    x_flat_np = np.reshape(x_np, (-1, 1))
    avgpooled_flat = implied_weight_matrix @ x_flat_np
    avgpooled = np.reshape(avgpooled_flat, (out_height, out_width))

    avgpooled_torch = torch.nn.AvgPool2d((2, 2))(x[None, :, :])[0, :, :]
    np.testing.assert_allclose(avgpooled, avgpooled_torch.detach().numpy())