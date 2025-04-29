import logging
import functools
import sys
import multiprocessing
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import scipy.sparse
import tqdm

import convolution_inversion
import path_config
import inversion
import tools
import pytorch_models
import caching


DO_TQDM_DECORATION = True
# https://bentyeh.github.io/2019/07/22/Python-multiprocessing-progress.html

np.set_printoptions(linewidth=1000)

logging_format = "{%(func)s:%(lineno)4s: {%(asctime)s: %(message)s"

logging_level = 15
# logging.basicConfig(level=logging_level,
#                     format=logging_format)

logger = logging.getLogger(__name__)

Polytope = Dict[str, Any]
HRepresentation = Dict[str, Any]
VRepresentation = Dict[str, Any]
Region = List[Polytope]

# Main possible speedups
#  update/downdate would be best
# Make region a first class thing


def _flatten_list_of_lists(ll: List[list]) -> list:
    flattened = [x for l in ll for x in l]
    return flattened


def invert_linearthenrelu_layer_kernel(w: np.ndarray,
                                       b: np.ndarray,
                                       r: Region,
                                       x_lower: np.ndarray,
                                       x_upper: np.ndarray,
                                       is_rational: bool,
                                       need_v: bool) -> Region:
    log_every_iters = 10
    n, m = w.shape

    index = tuple(range(n))
    index_powerset = tools.powerset(index)
    index_subsets = index_powerset

    num_to_decompose = len(r)
    all_vs = [None] * num_to_decompose
    all_hs = [None] * num_to_decompose

    for idx, td in enumerate(r):
        # idx = 0; td = to_decompose[idx]
        if 0 == idx % log_every_iters:
            logger.info("        #{} / {}".format(idx, num_to_decompose))

        decomped_vs, decomped_hs = inversion.relu_decomposition(index_subsets,
                                                                td,
                                                                w,
                                                                b,
                                                                x_lower,
                                                                x_upper,
                                                                is_rational,
                                                                need_v)
        all_vs[idx] = decomped_vs
        all_hs[idx] = decomped_hs

    decomp_vs = _flatten_list_of_lists(all_vs)
    decomp_hs = _flatten_list_of_lists(all_hs)
    decomp = [dict(h=_h, v=_v) for _v, _h in zip(decomp_vs, decomp_hs)]
    return decomp


def _get_weight_and_bias_from_linearlike_layer(layer: Any) -> Tuple[np.ndarray, np.ndarray]:
    param_list = list(layer.parameters())
    assert 2 >= len(param_list), "Expecting at most a weight and a bias"
    w_prev = param_list[0].detach().numpy()
    if layer.bias is None:
        b_prev = np.zeros((w_prev.shape[0], 1))
    else:
        b_prev = tools.vec(param_list[1].detach().numpy())
    return w_prev, b_prev


def _invert_linear_layer_kernel_open(p: Polytope,
                                     w: np.ndarray,
                                     b: np.ndarray,
                                     h_bounds: np.ndarray,
                                     need_v: bool,
                                     is_rational: bool) -> Polytope:
    if p is None:
        ci = None
    else:

        wnr, wnc = w.shape
        if p["h"]["is_empty"] or p["v"]["is_empty"]:
            empty_row = np.empty((0, 1 + wnc))
            h = dict(inequality=empty_row, linear=empty_row, is_empty=True)
            v = dict(vertices=empty_row, is_empty=True)
        else:
            h_ineq = p["h"]["inequality"]
            h_lin = p["h"]["linear"]
            # todo: it is not hard to accomodate linear parts
            assert 0 == h_lin.shape[0], "Unexpected non-empty linear H part"

            aa = -1 * h_ineq[:, 1:]
            bb = tools.vec(h_ineq[:, 0])

            w_rank = np.linalg.matrix_rank(w)

            alt_a = aa @ w
            alt_b = bb - aa @ b
            ineq_part = np.hstack((alt_b, -1 * alt_a))
            a_h_repr = np.vstack((ineq_part, h_bounds))
            a_h_repr_lin = np.empty((0, 1 + wnc))

            if need_v:
                want_analytical_calc = True
                if want_analytical_calc:
                    v_image = p["v"]
                    vertices_image = v_image["vertices"]
                    w_full_rank = w_rank >= wnr
                    v_exists = vertices_image is not None
                    analytical_calc_possible = w_full_rank and v_exists
                    if analytical_calc_possible:
                        if 0 == vertices_image.size:
                            a_v_repr = np.empty((0, 1 + wnc))
                        else:
                            a_v_repr = _analytical_v_form_inversion(vertices_image, w, b)
                    else:
                        a_v_repr = tools.h_to_v(a_h_repr, a_h_repr_lin, is_rational)
                        a_v_repr = tools.eliminate_sign_repeated_rows(a_v_repr)
                        # if a_v_repr.shape[0] > 0:
                        #     print("Nonempty v repr")
            else:
                a_v_repr = None
                v_lin = None

            h_lin = np.empty((0, a_h_repr.shape[1]))

            h = dict(inequality=a_h_repr, linear=h_lin, is_empty=False)
            v = dict(vertices=a_v_repr, is_empty=False)
        ci = dict(h=h, v=v)
    return ci


def _analytical_v_form_inversion(v_repr: np.ndarray,
                                 w: np.ndarray,
                                 b: np.ndarray) -> np.ndarray:
    dim_in = v_repr.shape[1] - 1
    dim_out = len(b)
    assert dim_out, dim_in == w.shape

    if np.all(0 == v_repr[:, 0]):  # no vertices, just rays
        origin_vertex = np.hstack((np.eye(1), np.zeros((1, dim_in))))
        v_repr = np.vstack([origin_vertex, v_repr])
    is_ray = (0 == v_repr[:, 0])
    is_pol = ~is_ray

    is_vertex = tools.vec(v_repr[:, 0])
    vertices = v_repr[:, 1:]
    to_subtract = tools.vec(is_pol) @ b.T

    w_pinv = np.linalg.pinv(w)
    w_nullspace_basis = scipy.linalg.null_space(w)
    nullspace_dim = w_nullspace_basis.shape[1]

    aa_recentered = vertices - to_subtract
    w_pinv_aa_recentered = (w_pinv @ aa_recentered.T).T

    particular_piece = np.hstack((is_vertex, w_pinv_aa_recentered))
    nullspace_piece_unsigned = np.hstack((np.zeros((nullspace_dim, 1)), w_nullspace_basis.T))

    nullspace_piece = np.vstack((+1 * nullspace_piece_unsigned,
                                 -1 * nullspace_piece_unsigned))
    v_repr_preimage = np.vstack((particular_piece, nullspace_piece))

    do_check = False
    # do_check = True
    if do_check:
        # check_threshold = 1e-7
        # check_threshold = 5e-6
        # check_threshold = 1e-5
        check_threshold = 5e-5

        dim = v_repr_preimage.shape[1] - 1
        v_lin = np.empty((0, dim + 1))
        h_ineq = tools.v_to_h(v_repr, v_lin, True)
        point = tools.get_point_from_v_repr(v_repr_preimage)

        tmp = w @ point + b
        tmp1 = np.vstack((np.eye(1), tmp))
        threshold_values = h_ineq @ tmp1
        min_threshold_value = np.min(threshold_values)
        assert min_threshold_value > -1 * check_threshold
    return v_repr_preimage


def invert_linear_layer_kernel(w: np.ndarray,
                               b: np.ndarray,
                               r: Region,
                               x_lower: np.ndarray,
                               x_upper: np.ndarray,
                               is_rational: bool,
                               need_v: bool) -> Region:
    m, n = w.shape
    assert len(b) == m

    h_bounds = tools.build_bounding_box_h_form(x_lower, x_upper)
    _invert_linear_layer_kernel_open_to_map = functools.partial(_invert_linear_layer_kernel_open,
                                                                w=w,
                                                                b=b,
                                                                h_bounds=h_bounds,
                                                                need_v=need_v,
                                                                is_rational=is_rational)
    # do_mapping = False
    do_mapping = True
    if do_mapping:
        not_empty_list = [not _["h"]["is_empty"] for _ in r]
        i = np.argmax(not_empty_list)
        p = r[i]
        _invert_linear_layer_kernel_open(p, w, b, h_bounds, need_v, is_rational)

        # do_multiprocessing = True
        do_multiprocessing = False
        if do_multiprocessing:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                decomposed_map = p.map(_invert_linear_layer_kernel_open_to_map, r)
        else:
            decomposed_map = map(_invert_linear_layer_kernel_open_to_map, r)

        decomposed = list(tqdm.tqdm(decomposed_map, total=len(r)))
    else:
        decomposed = [None] * len(r)
        for idx, p in enumerate(r):
            # idx = 0; p = r[idx]
            decomposed[idx] = _invert_linear_layer_kernel_open_to_map(p)
    return decomposed


def _build_flat_representation(layer: Any) -> Tuple[np.ndarray, np.ndarray]:
    w, b = _get_weight_and_bias_from_linearlike_layer(layer)
    in_shape = layer.in_shape
    num_out_channels = layer.num_out_channels
    kernel_size = layer[0].kernel_size
    stride = layer[0].stride
    out_channels = layer.num_out_channels

    assert num_out_channels == w.shape[0]
    assert in_shape[0] == w.shape[1]
    assert kernel_size[0] == w.shape[2]
    assert kernel_size[1] == w.shape[3]

    out_height = safe_int((in_shape[1] - kernel_size[0]) / stride[0] + 1)
    out_width = safe_int((in_shape[2] - kernel_size[1]) / stride[1] + 1)

    c_shape = (out_channels, out_height, out_width)
    implied_weight_matrix, \
    implied_bias_vector = convolution_inversion.build_implied_weight_matrix(w,
                                                                            b,
                                                                            c_shape,
                                                                            stride)
    return implied_weight_matrix, implied_bias_vector


def densify_if_needed(x: np.ndarray) -> np.ndarray:
    if type(x) in [scipy.sparse.coo.coo_matrix]:
        x = x.todense()
        x = np.array(x)
    return x


def _densify_polytope(p: Polytope) -> Polytope:
    p_dense = {
        "h": {
            "inequality": densify_if_needed(p["h"]["inequality"]),
            "linear": densify_if_needed(p["h"]["linear"]),
            "is_empty": p["h"]["is_empty"],
        },
        "v": {
            "vertices": densify_if_needed(p["v"]["vertices"]),
            "is_empty": p["v"]["is_empty"]
        }
    }
    return p_dense


def sparsify_polytope(p: Polytope) -> Polytope:
    p_sparse = {
        "h": {
            "inequality": scipy.sparse.coo_matrix(p["h"]["inequality"]),
            "linear": scipy.sparse.coo_matrix(p["h"]["linear"]),
            "is_empty": p["h"]["is_empty"]
        },
        "v": {
            "vertices": scipy.sparse.coo_matrix(p["v"]["vertices"]),
            "is_empty": p["v"]["is_empty"]
        }
    }
    return p_sparse


def _decomposition_dispatcher(r: Region,
                              layer_coefficient: dict,
                              layer_type: str,
                              layer_arg: dict) -> Region:
    need_v = layer_arg.get("need_v", False)
    is_rational = layer_arg.get("is_rational", None)
    layer_limit = layer_arg.get("layer_limit", None)
    make_sparse = layer_arg.get("make_sparse", False)

    r = [_densify_polytope(_) for _ in r]
    if layer_type == "LinearThenRelu":
        w, b = layer_coefficient["w"], layer_coefficient["b"]
        x_lower, x_upper = layer_limit
        inverted_layer = invert_linearthenrelu_layer_kernel(w,
                                                            b,
                                                            r,
                                                            x_lower,
                                                            x_upper,
                                                            is_rational,
                                                            need_v)
    elif layer_type in ["Linear"]:
        w, b = layer_coefficient["w"], layer_coefficient["b"]
        x_lower, x_upper = layer_limit
        inverted_layer = invert_linear_layer_kernel(w,
                                                    b,
                                                    r,
                                                    x_lower,
                                                    x_upper,
                                                    is_rational,
                                                    need_v)
    elif layer_type in ["FlatConv2d", "FlatAvgPool"]:
        implied_weight_matrix = layer_coefficient["implied_weight_matrix"]
        implied_bias_vector = layer_coefficient["implied_bias_vector"]
        x_lower, x_upper = layer_limit
        inverted_layer = invert_linear_layer_kernel(implied_weight_matrix,
                                                    implied_bias_vector,
                                                    r,
                                                    x_lower,
                                                    x_upper,
                                                    is_rational,
                                                    need_v)
    elif layer_type in ["FlatRelu"]:
        # p = r[0]
        # to_flatten = [inversion.invert_relu_layer_kernel(p, need_v, is_rational)
        #               for p in r]
        do_multiprocessing = False

        if do_multiprocessing:
            raise ValueError("Not implemented yet")
        else:
            tqdm_wrap = False
            to_map = functools.partial(inversion.invert_relu_layer_kernel,
                                       need_v=need_v,
                                       is_rational=is_rational)
            the_map = map(to_map, r)
            if tqdm_wrap:
                the_map = the_map
                the_map = tqdm.tqdm(the_map, total=len(r), leave=False)
            to_flatten = list(the_map)

        inverted_layer = _flatten_list_of_lists(to_flatten)
    elif layer_type == "FlatIdent":
        inverted_layer = r
    else:
        raise ValueError("Do not know how to invert {}".format(layer_type))
    if make_sparse:
        inverted_layer = [sparsify_polytope(_) for _ in inverted_layer]
    return inverted_layer


def _build_one_layer_limits(layer_type: str,
                            layer_dimensions: Dict[str, Any],
                            bound_scale: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    if layer_type in ["Linear"]:
        bound_dim = layer_dimensions["in_features"]
    elif layer_type in ["FlatConv2d", "FlatAvgPool"]:
        p = np.prod(layer_dimensions["in_shape"])
        bound_dim = safe_int(p)
    elif layer_type in ["LinearThenRelu"]:
        bound_dim = layer_dimensions["in_features"]
    elif layer_type in ["FlatRelu", "FlatIdent"]:
        bound_dim = None
    else:
        raise ValueError("I do not know how to compute the bounds for layer {}".format(layer_type))

    if bound_dim is None:
        layer_limit = None
    else:
        bound_shape = (bound_dim, 1)
        x_lower = bound_scale[0] * np.ones(bound_shape)
        x_upper = bound_scale[1] * np.ones(bound_shape)
        layer_limit = (x_lower, x_upper)
    return layer_limit


def _build_layer_limits(layer_types: List[str],
                        layer_dimensions: List[dict],
                        bound_scale_by_layer: List[Tuple[float, float]]) -> List[Tuple[np.ndarray]]:
    num_layers = len(layer_types)
    layer_limits = [None] * num_layers
    # _build_one_layer_limits
    for idx, layer_type in enumerate(layer_types):
        layer_dimension = layer_dimensions[idx]
        bound_scale = bound_scale_by_layer[idx]
        layer_limit = _build_one_layer_limits(layer_type, layer_dimension, bound_scale)
        layer_limits[idx] = layer_limit
    return layer_limits


def _decompose_backwards(terminal_decomp: Region,
                         layer_info: Dict[str, dict]) -> List[list]:
    layer_args = layer_info["args"]
    layer_types = layer_info["types"]
    layer_dimensions = layer_info["dimensions"]
    layer_coefficients = layer_info["coefficients"]

    paths = path_config.get_paths()
    cache_dir = paths['cached_calculations']

    assert layer_types[-1] == "Linear", \
        "If this does not hold, may need to rethink some things"

    num_layers = len(layer_coefficients)
    decomp_by_layer = [None] * (num_layers + 1)
    decomp_by_layer[-1] = terminal_decomp

    for layer_idx in range(num_layers):
        # layer_idx = 0
        # layer_idx = 1
        invert_layer_idx = num_layers - layer_idx - 1
        layer_arg = layer_args[invert_layer_idx]

        layer_type = layer_types[invert_layer_idx]
        layer_dimension = layer_dimensions[invert_layer_idx]
        layer_coefficient = layer_coefficients[invert_layer_idx]

        logger.info("     Inverting Layer #{}: {}, {}".format(invert_layer_idx,
                                                              layer_type,
                                                              layer_dimension))
        is_debugging = sys.gettrace() is not None
        want_regeneration = False

        polytopes_to_decompose = decomp_by_layer[invert_layer_idx + 1]  # Type: List[Polytope]
        calc_fun = _decomposition_dispatcher
        calc_args = (polytopes_to_decompose,
                     layer_coefficient,
                     layer_type,
                     layer_arg)
        calc_kwargs = {}
        if False:
            assert layer_type == 'Linear'
            w, b = layer_coefficient["w"], layer_coefficient["b"]
            need_v = layer_arg.get("need_v", False)
            is_rational = layer_arg.get("is_rational", None)
            layer_limit = layer_arg.get("layer_limit", None)

            x_lower, x_upper = layer_limit
            inverted_layer = invert_linear_layer_kernel(w,
                                                        b,
                                                        r,
                                                        x_lower,
                                                        x_upper,
                                                        is_rational,
                                                        need_v)

        force_regeneraton = want_regeneration and is_debugging

        inverted_layer = caching.cached_calc(cache_dir,
                                             calc_fun,
                                             calc_args,
                                             calc_kwargs,
                                             force_regeneraton)
        decomp_by_layer[invert_layer_idx] = inverted_layer
    return decomp_by_layer


def _build_images_to_invert(out_dim: int,
                            inversion_par: Dict[str, Any]) -> List[Polytope]:
    invert_classes = inversion_par["invert_classes"]
    is_rational = inversion_par["is_rational"]
    desired_margin = inversion_par["desired_margin"]

    images_to_invert = [None] * len(invert_classes)
    for idx, cls in enumerate(invert_classes):
        # idx = 0; cls = invert_classes[idx]
        inequality = tools.build_polytope_where_nth_coordinate_is_greatest(cls,
                                                                           out_dim,
                                                                           desired_margin)
        h_lin = np.empty((0, inequality.shape[1]))
        v = tools.h_to_v(inequality, h_lin, is_rational)
        h_repr = dict(inequality=inequality, linear=h_lin, is_empty=False)
        v_repr = dict(vertices=v, is_empty=False)

        terminal_polytope = dict(h=h_repr, v=v_repr)
        images_to_invert[idx] = terminal_polytope
    return images_to_invert


def safe_int(n: float) -> int:
    intn = int(n)
    assert intn == n
    return intn


def get_layer_namer() -> Dict[type, str]:
    layer_namer = {
        torch.nn.modules.linear.Linear: "Linear",
        pytorch_models.FlatAvgPool: "FlatAvgPool",
        pytorch_models.FlatConv2d: "FlatConv2d",
        pytorch_models.FlatIdent: "FlatIdent",
        pytorch_models.FlatRelu: "FlatRelu",
        pytorch_models.LinearThenRelu: "LinearThenRelu"
    }
    return layer_namer


def get_layer_dimensions(layer) -> Dict[str, int]:
    layer_dimensions = dict()
    if hasattr(layer, "in_shape"):
        layer_dimensions["in_shape"] = layer.in_shape
    if hasattr(layer, "in_features"):
        layer_dimensions["in_features"] = layer.in_features
    if hasattr(layer, "out_features"):
        layer_dimensions["out_features"] = layer.out_features

    return layer_dimensions


def get_layer_coefficients(layer, layer_type: str) -> Dict[str, np.ndarray]:
    layer_coefficients = dict()
    if layer_type in ["Linear", "LinearThenRelu"]:
        w, b = _get_weight_and_bias_from_linearlike_layer(layer)
        layer_coefficients["w"] = w
        layer_coefficients["b"] = b
    elif layer_type in ["FlatConv2d", "FlatAvgPool"]:
        implied_weight_matrix, implied_bias_vector = _build_flat_representation(layer)
        layer_coefficients["implied_weight_matrix"] = implied_weight_matrix
        layer_coefficients["implied_bias_vector"] = implied_bias_vector
    else:
        pass
    return layer_coefficients


def _build_layer_args(layer_types,
                      layer_dimensions,
                      inversion_par: Dict[str, Any]) -> Dict[str, Any]:
    input_layer_bounds = inversion_par['input_layer_bounds']
    is_rational = inversion_par["is_rational"]

    num_layers = len(layer_types)

    # For futher expansion if needed
    bound_scale_by_layer = [(-np.inf, +np.inf)] * num_layers
    layer_limits = _build_layer_limits(layer_types,
                                       layer_dimensions,
                                       bound_scale_by_layer)
    input_layer_bounds_l = input_layer_bounds[0]
    input_layer_bounds_u = input_layer_bounds[1]

    if isinstance(input_layer_bounds_l, (float, int)) and \
            isinstance(input_layer_bounds_u, (float, int)):
        in_features = layer_dimensions[0]["in_features"]
        lower = np.full((in_features, 1), input_layer_bounds_l)
        upper = np.full((in_features, 1), input_layer_bounds_u)
    elif isinstance(input_layer_bounds_l, np.ndarray) and \
            isinstance(input_layer_bounds_u, np.ndarray):
        lower = input_layer_bounds_l
        upper = input_layer_bounds_u

    input_layer_bounds = (lower, upper)
    layer_limits[0] = input_layer_bounds

    need_initial_v = inversion_par["need_initial_v"]
    need_linear_layer_v = True

    need_all_v_up_to_first_relu = True

    make_sparse = False
    layer_arguments = [None] * num_layers
    need_v_by_layer = [False] * num_layers
    if need_all_v_up_to_first_relu:
        flat_relu_index = layer_types.index("FlatRelu")
        for idx in range(1 + flat_relu_index):
            need_v_by_layer[idx] = True

    need_v_by_layer[0] = need_initial_v

    if need_linear_layer_v:
        for idx, lt in enumerate(layer_types):
            if lt == 'Linear':
                need_v_by_layer[idx] = True

    for idx in range(num_layers):
        layer_type = layer_types[idx]
        idx_layer_limits = layer_limits[idx]
        need_v = need_v_by_layer[idx]
        idx_layer_arguments = {
            "is_rational": is_rational,
            "need_v": need_v,
            "make_sparse": make_sparse,
            "layer_limit": idx_layer_limits
        }
        if layer_type in ["Linear", "LinearThenRelu"]:
            pass
        elif layer_type in ["FlatConv2d", "FlatAvgPool"]:
            pass
        elif layer_type in ["FlatRelu", "LinearThenRelu"]:
            pass
        else:
            pass
        layer_arguments[idx] = idx_layer_arguments
    return layer_arguments


def build_layer_info(layers,
                     inversion_par: Dict[str, Any]) -> Dict[str, dict]:
    layer_namer = get_layer_namer()

    layer_types = [layer_namer[type(_)] for _ in layers]
    layer_dimensions = [get_layer_dimensions(_) for _ in layers]
    layer_coefficients = [get_layer_coefficients(_, lt) for _, lt in zip(layers, layer_types)]
    layer_args = _build_layer_args(layer_types, layer_dimensions, inversion_par)

    layer_info = {
        "args": layer_args,
        "types": layer_types,
        "dimensions": layer_dimensions,
        "coefficients": layer_coefficients,
    }
    return layer_info


def compute_decomps(layer_info: Dict[str, Any],
                    inversion_par: Dict[str, Any]) -> List[list]:
    layer_dimensions = layer_info["dimensions"]

    out_dim = layer_dimensions[-1]["out_features"]
    images_to_invert = _build_images_to_invert(out_dim, inversion_par)
    decomps = [None] * len(images_to_invert)  # Type: List[list]

    for idx, terminal_polytope in enumerate(images_to_invert):
        # idx = 0; terminal_polytope = images_to_invert[0]
        logger.info("  Starting decomposition #{}".format(idx))
        terminal_decomp = [terminal_polytope]
        class_decomp = _decompose_backwards(terminal_decomp, layer_info)
        decomps[idx] = class_decomp
    return decomps
