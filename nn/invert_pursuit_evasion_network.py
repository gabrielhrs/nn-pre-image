import os
import logging
import time
from typing import Any, Dict, List, Tuple
import pickle

import numpy as np

import caching
import tools
import decomp
import NNet.python.nnet


Polytope = Dict[str, Any]
HRepresentation = Dict[str, Any]
VRepresentation = Dict[str, Any]
Region = List[Polytope]

np.set_printoptions(linewidth=1000)

logging_format = "%(asctime)s: %(message)s"
# logging_format = "{%(func)s: %(lineno)4s: {%(asctime)s: %(message)s"

logging_level = 15
logging.basicConfig(level=logging_level,
                    format=logging_format)

logger = logging.getLogger(__name__)


def interleave_lists(l1: List[Any],
                     l2: List[Any]) -> List[Any]:
    # https://codegolf.stackexchange.com/questions/169893/python-shortest-way-to-interleave-items-from-two-lists
    return [*sum(zip(l1, l2), ())]


def _matrix_rank_empty(m: np.ndarray) -> int:
    if 0 == m.shape[0]:
        mre = 0
    else:
        mre = np.linalg.matrix_rank(m)
    return mre


def _build_layer_inverse_args(idx: int,
                              dim: int) -> Dict[str, Any]:
    # layer_inverse_args = {}
    is_rational = False
    # need_v = True
    # need_v = (idx == 0)
    need_v = False
    if dim is None:
        layer_limit = None
    else:
        layer_limit = (np.full((dim, 1), -1 * np.inf),
                       np.full((dim, 1), +1 * np.inf))

    layer_inverse_args = {
        "is_rational": is_rational,
        "need_v": need_v,
        "layer_limit": layer_limit
    }
    return layer_inverse_args


def invert_relunet_fromwb(ws, bs, input_layer_bounds):
    in_dim = ws[0].shape[1]
    out_dim = ws[-1].shape[0]
    dim = ws[0].shape[0]
    num_linear_layers = len(ws)
    num_relu_layers = len(ws) - 1
    assert 1 + num_relu_layers == num_linear_layers
    # recover the NN on coefficients, layer types and dimensions
    linear_layer_coefficients = [{"w": ws[idx], "b": bs[idx]} for idx in range(num_linear_layers)]
    relu_layer_coefficients = [dict()] * num_relu_layers
    coefficients = interleave_lists(linear_layer_coefficients[:-1],
                                    relu_layer_coefficients) + [linear_layer_coefficients[-1]]
    types = ["Linear", "FlatRelu"] * num_relu_layers + ["Linear"]
    dimensions = [{"in_features": in_dim, "out_features": dim}] + \
                 [{"in_features": dim, "out_features": dim}] * (2 * num_relu_layers - 1) + \
                 [{"in_features": dim, "out_features": out_dim}]
    args = [_build_layer_inverse_args(idx, d["in_features"]) for idx, d in enumerate(dimensions)]
    args[0]["layer_limit"] = input_layer_bounds
    args[0]["need_v"] = True
    # length check
    num_layers = num_relu_layers + num_linear_layers
    assert len(types) == num_layers
    assert len(dimensions) == num_layers
    assert len(coefficients) == num_layers

    layer_info = {'args': args,
                  'types': types,
                  'dimensions': dimensions,
                  'coefficients': coefficients}

    is_rational = False
    desired_margin = 0.0

    invert_classes = list(range(out_dim))
    cache_inversion = True

    # start the inversion
    do_inversion = True
    if do_inversion:
        inversion_par = {
            "cache_inversion": cache_inversion,
            "desired_margin": desired_margin,
            "input_layer_bounds": input_layer_bounds,
            "invert_classes": invert_classes,
            "is_rational": is_rational
        }

        logger.info("Starting decomps")
        calc_fun = decomp.compute_decomps
        calc_args = (layer_info, inversion_par)
        calc_kwargs = {}
        force_regeneraton = True
        do_caching = False
        if do_caching:
            # read the preimages from caching, or compute and cache a new one
            cache_dir = paths
            decomps = caching.cached_calc(cache_dir,
                                          calc_fun,
                                          calc_args,
                                          calc_kwargs,
                                          force_regeneraton)
        else:
            decomps = decomp.compute_decomps(layer_info, inversion_par)
    return decomps


# --------------------
paths = os.getcwd()
hcas_root = os.getcwd()
networks_dir = os.getcwd()
# --------------------


def compute_h_reprs(name, input_region_info):
    # load nn, and specify input and output bounds
    n = NNet.python.nnet.NNet(name)
    ranges = input_region_info['ranges']
    means = input_region_info['means']
    maxes = input_region_info['maxes']
    mins = input_region_info['mins']
    upper = (maxes - means) / ranges
    lower = (mins - means) / ranges
    input_layer_bounds = (lower, upper)
    ws = n.weights
    bs = [np.vstack(b) for b in n.biases]
    # compute the preimages
    decomps = invert_relunet_fromwb(ws, bs, input_layer_bounds)
    preimages = [d[0] for d in decomps]
    # save preimages by their classes
    h_reprs = []
    v_reprs = []
    for idx1, preimage in enumerate(preimages):
        h_reprs.append([])
        v_reprs.append([])
        for idx2, p in enumerate(preimage):
            h = p["h"]
            if h["is_empty"]:
                continue
            h_ineq = h["inequality"]
            h_lin = h["linear"]
            v_repr = tools.h_to_v(h_ineq, h_lin, False)
            rnk = _matrix_rank_empty(v_repr)
            usable = rnk > 0
            if usable:
                h_repr = []
                for h_plane in h_ineq:
                    # Not clear why the denomalization and subtraction are needed
                    A_plane = h_plane[1:3] @ (np.max(ranges) * np.identity(2)/(ranges @ np.ones([1, 2])))
                    b_plane = np.max(ranges) * h_plane[0] - \
                              h_plane[1:3] @ (np.max(ranges) * np.identity(2)/(ranges @ np.ones([1, 2])) @ means)
                    if len(h_repr) == 0:
                        h_repr = np.hstack((b_plane, A_plane))
                    else:
                        h_repr = np.vstack((h_repr, np.hstack((b_plane, A_plane))))
                h_reprs[idx1].append(h_repr)
                v_reprs[idx1].append(v_repr)
                # plot_v_repr = tools.apply_linear_transformation_to_v_repr(v_repr, w, b) # c, h, h_1, t
                # h_v_h_reprs[idx1].append(tools.v_to_h(plot_v_repr, None, True))
    return h_reprs, v_reprs


if __name__ == "__main__":
    # compute the preimages and return h-representations and v-representations. NN has the format ".nnet"
    time_stamp = time.time()
    ident = 'pursuit_evasion_coarse'
    ident_pattern = "pursuit_evasion_model_coarse".format(ident)
    filename = ident_pattern.format() + ".nnet"
    fullfilename = os.path.join(networks_dir, filename)
    # specify ranges in the input space
    # ranges = np.vstack([[4.], [4.]])
    # means = np.vstack([[2.], [2.]])
    # maxes = np.vstack([[4.], [4.]])
    # mins = np.vstack([[0.], [0.]])
    # ranges = np.vstack([[5.], [3.]])
    # means = np.vstack([[2.5], [1.5]])
    # maxes = np.vstack([[5.], [3.]])
    # mins = np.vstack([[0.], [0.]])
    xmax = 3.
    ymax = 3.
    ranges = np.vstack([[xmax], [ymax]])
    means = np.vstack([[xmax/2], [ymax/2]])
    maxes = np.vstack([[xmax], [ymax]])
    mins = np.vstack([[0.0], [0.0]])
    input_region_info = {'ranges': ranges,
                         'means': means,
                         'maxes': maxes,
                         'mins': mins}
    h_repr, v_repr = compute_h_reprs(fullfilename, input_region_info)

    # Computes the nn inverses and saves them into files
    with open('{}_pre_image_h'.format(ident), 'wb') as pre_image_file:
        pickle.dump(h_repr, pre_image_file)
    with open('{}_pre_image_v'.format(ident), 'wb') as pre_image_file:
        pickle.dump(v_repr, pre_image_file)
    print("inversion took " + str(time.time() - time_stamp) + " seconds")
