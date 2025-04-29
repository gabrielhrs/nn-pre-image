import logging
import functools
import multiprocessing
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse

import gurobipy
from gurobipy import GRB

import tqdm
import cvxpy

import tools

# https://inf.ethz.ch/personal/fukudak/lect/pclect/notes2014/PolyComp2014.pdf

logger = logging.getLogger(__name__)

DO_TQDM_DECORATION = True

Polytope = Dict[str, Any]
HRepresentation = Dict[str, Any]
VRepresentation = Dict[str, Any]
Region = List[Polytope]


def _relu_decompose_for_activation_pattern(
        subset: tuple,
        index: tuple,
        m: int,
        h_pos_invariant: np.ndarray,
        h_zeroimage_invariant: np.ndarray,
        h_relupos_invariant: np.ndarray,
        h_sum1_invariant: np.ndarray,
        h_r0_invariant: np.ndarray,
        h_nonzeroimage_invariant: np.ndarray,
        is_rational: bool,
) -> Tuple[tuple, tuple]:
    is_relued = np.in1d(index, subset)  # Type: np.ndarray

    # https://math.stackexchange.com/questions/2097171/is-it-possible-to-check-polytope-containment-by-only-checking-the-feasibility-of/2097261
    # How can we skip computing the V representation if we know that it will be empty?

    h_pos = h_pos_invariant
    h_zeroimage = h_zeroimage_invariant[is_relued, :]
    h_relupos = h_relupos_invariant[~is_relued, :]

    h_sum1 = h_sum1_invariant
    h_r0 = h_r0_invariant[is_relued, :]
    h_nonzeroimage = h_nonzeroimage_invariant[~is_relued, :]

    h_ineq = np.vstack((h_pos, h_zeroimage, h_relupos))
    h_lin = np.vstack((h_sum1, h_r0, h_nonzeroimage))

    num_ineq, dim = h_ineq.shape
    num_eq = h_lin.shape[0]
    logger.debug("H to V w/ dim {}, ineqs {}, eqs {}".format(dim, num_ineq, num_eq))
    preimage_full = tools.h_to_v(h_ineq, h_lin, is_rational)
    preimage_v = preimage_full[:, :m + 1]

    assert num_eq < num_ineq, "why is this happening?"
    is_large_problem = num_ineq - num_eq > 100
    if preimage_v.size > 0:
        assert not is_large_problem, "Look at whether we really need problems this big?"
    logger.debug("done")

    preimage_v_lin = None
    return (preimage_v, preimage_v_lin), (h_ineq, h_lin)


def _v_centric_relu_decomposition(index_subsets: List[tuple],
                                  p: Polytope,
                                  w: np.ndarray,
                                  b: np.ndarray,
                                  x_lower: np.ndarray,
                                  x_upper: np.ndarray,
                                  is_rational: bool = False) -> Tuple[list, list]:
    a = p.v[0]
    n, m = w.shape
    assert a[:, 1:].shape[1] == n, "Dimension disagreement between w and v_repr"

    index = tuple(range(n))

    n, m = w.shape
    assert (n, 1) == b.shape
    assert m == x_lower.shape[0]
    assert m == x_upper.shape[0]

    vertices = a[:, 1:]
    v, _ = vertices.shape
    assert _ == n

    is_ray = 0 == a[:, 0]
    is_pol = ~is_ray

    #                 L >= 0
    # iff [0, +I][x; L] >= 0
    h_l_pos = np.hstack((np.zeros((v, 1)), np.zeros((v, m)), np.eye(v)))

    #                 x >= x_lower
    # iff [+I, 0][x; L] >= [+1 * x_lower; 0]
    # iff [-1 * x_lower, +I, 0] [1; x; L] >= 0
    x_lower_finite = (x_lower > -1 * np.inf).flatten()
    h_x_lower = np.hstack((-1 * x_lower, +1 * np.eye(m), np.zeros((m, v))))[x_lower_finite, :]

    #                x <=  x_upper
    # iff [-I, 0][x; L] >= [-x_upper; 0]
    # iff [+x_upper, -I, 0][1; x; L] >= 0
    x_upper_finite = (x_upper < +1 * np.inf).flatten()
    h_x_upper = np.hstack((+1 * x_upper, -1 * np.eye(m), np.zeros((m, v))))[x_upper_finite, :]

    h_pos_invariant = np.vstack((h_l_pos, h_x_lower, h_x_upper))

    # polytope weights sum to one:
    if np.any(is_pol):
        ip = np.reshape(is_pol.astype(float), (1, -1))
        h_sum1 = np.hstack((np.eye(1), np.zeros((1, m)), -1 * ip))
    else:
        h_sum1 = np.empty((0, 1 + m + v))

    h_sum1_invariant = h_sum1

    # where relued
    #         wx + b <= 0
    #     iff [-w, 0][x; L] - b >= 0
    #     iff [-b, -w, 0][1; x; L] >= 0
    h_zeroimage_invariant = np.hstack((-1 * b, -1 * w, np.zeros((n, v))))

    #     vL == 0
    h_r0_invariant = np.hstack((np.zeros((n, 1)), np.zeros((n, m)), vertices.T))

    # Where not relued
    #     is in image:
    #          wx + b == vL
    #     iff b + wx - vL == 0
    #     iff [b, w, -v][1; x; L] == 0
    h_nonzeroimage_invariant = np.hstack((b, w, -1 * vertices.T))

    #     wx + b >= 0 iff [b, w, 0][1; x; L] >= 0
    h_relupos_invariant = np.hstack((b, w, np.zeros((n, v))))

    # TODO: if some index is always in or out, it can be optimised out
    wrapped_kernel = functools.partial(
        _relu_decompose_for_activation_pattern,
        index=index,
        m=m,
        h_pos_invariant=h_pos_invariant,
        h_zeroimage_invariant=h_zeroimage_invariant,
        h_relupos_invariant=h_relupos_invariant,
        h_sum1_invariant=h_sum1_invariant,
        h_r0_invariant=h_r0_invariant,
        h_nonzeroimage_invariant=h_nonzeroimage_invariant,
        is_rational=is_rational,
    )

    do_parallel = True
    # do_parallel = False

    if do_parallel:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            relu_decomp = list(p.map(wrapped_kernel, index_subsets))
    else:
        relu_decomp = list(map(wrapped_kernel, index_subsets))
    relu_decomp_vs = [x[0] for x in relu_decomp]
    relu_decomp_hs = [x[1] for x in relu_decomp]
    return relu_decomp_vs, relu_decomp_hs


def _h_centric_relu_decomposition(index_subsets: List[tuple],
                                  p: Polytope,
                                  w: np.ndarray,
                                  b: np.ndarray,
                                  x_lower: np.ndarray,
                                  x_upper: np.ndarray,
                                  is_rational: bool,
                                  need_v: bool) -> Tuple[list, list]:
    h = p["h"]
    n, m = w.shape
    index = tuple(range(n))

    hab_ineq = h["inequality"]
    hab_lin = h["linear"]
    h_bounds = tools.build_bounding_box_h_form(x_lower, x_upper)
    #         wx + b <= 0
    #     iff [-b, -w][1; x] >= 0
    h_neg_invariant = np.hstack((-1 * b, -1 * w))
    #     wx + b >= 0 iff [b, w][1; x] >= 0
    h_pos_invariant = np.hstack((b, w))

    # A[wx + b] <= a
    # iff Awx + Ab - a <= 0
    # iff [Ab - a, Aw][1; x] <= 0
    # iff [-Ab + a, -Aw][1; x] >= 0

    num_subsets = len(index_subsets)
    relu_decomp_vs = [None] * num_subsets
    relu_decomp_hs = [None] * num_subsets

    for ind, subset in enumerate(index_subsets):
        # ind = -2; subset = index_subsets[ind]
        is_relued = np.in1d(index, subset)  # Type: np.ndarray

        a_ineq = np.where(is_relued, 0, -1 * hab_ineq[:, 1:])
        a = tools.vec(hab_ineq[:, 0])
        h_image_ineq = np.hstack((-1 * a_ineq @ b + a, -1 * a_ineq @ w))

        a_ineq = np.where(is_relued, 0, -1 * hab_lin[:, 1:])
        a = tools.vec(hab_lin[:, 0])
        h_image_lin = np.hstack((-1 * a_ineq @ b + a, -1 * a_ineq @ w))

        h_pos = h_pos_invariant[~is_relued, :]
        h_neg = h_neg_invariant[is_relued, :]
        h_ineq = np.vstack((h_bounds, h_neg, h_pos, h_image_ineq))

        h_lin = h_image_lin
        v_lin = None
        if need_v:
            v = tools.h_to_v(h_ineq, h_lin, is_rational)
        else:
            v = None
        relu_decomp_vs[ind] = (v, v_lin)
        relu_decomp_hs[ind] = (h_ineq, h_lin)
    return relu_decomp_vs, relu_decomp_hs


def relu_decomposition(
        index_subsets: List[tuple],
        p: Polytope,
        w: np.ndarray,
        b: np.ndarray,
        x_lower: np.ndarray,
        x_upper: np.ndarray,
        is_rational: bool,
        need_v: bool) -> Tuple[list, list]:
    relu_decomp_vs, relu_decomp_hs = _h_centric_relu_decomposition(index_subsets,
                                                                   p,
                                                                   w,
                                                                   b,
                                                                   x_lower,
                                                                   x_upper,
                                                                   is_rational,
                                                                   need_v)
    hhh = [dict(inequality=_[0], linear=_[1], is_empty=False) for _ in relu_decomp_hs]
    vvv = [dict(vertices=_[0], is_empty=False) for _ in relu_decomp_vs]
    return vvv, hhh


def densify_if_needed(x: np.ndarray) -> np.ndarray:
    if type(x) in [scipy.sparse.coo.coo_matrix]:
        x = x.todense()
        x = np.array(x)
    return x


def _build_sign_constraint_matrix(not_relued_diag: np.ndarray) -> np.ndarray:
    n = not_relued_diag.shape[0]

    eyen = np.eye(n)
    zeron = np.zeros((n, 1))
    # Basic idea:
    #   +1 * x >= 0 iff x >= 0
    #   -1 * x >= 0 iff x >= 0
    #   (1 - iota)_i = 0 if iota_i == 1
    #                = 1 if iota_i == 0
    #  or, put differently, (1 - iota)_i = ~iota_i, where "~" give the (logical) negation

    # if not relued, this entry needs to be weakly positive
    # pos_cons = np.hstack((zeron, -(eyen - not_relued_diag)))
    pos_cons = np.hstack((zeron, not_relued_diag - eyen))

    # if relued, this entry needs to be weakly negative
    neg_cons = np.hstack((zeron, not_relued_diag))
    sign_cons = np.vstack((pos_cons, neg_cons))
    sign_cons = drop_all_zero_rows(sign_cons)
    return sign_cons


def drop_all_zero_rows(x: np.ndarray) -> np.ndarray:
    return x[~np.all(0 == x, axis=1), :]


def farkas_dual_calc(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    https://docs.mosek.com/whitepapers/infeas.pdf

    page 18:
    Ax <= b is empty iff
    there is a y st y >= 0, A'y == 0, y'b < 0.
    iff value < 0 in the below
    """
    m, n = a.shape
    y = cvxpy.Variable(m)

    constraints = [a.T @ y == 0, y >= 0]
    q = y.T @ b
    objective = cvxpy.Minimize(q)

    prob = cvxpy.Problem(objective, constraints)
    # verbose = False
    verbose = True
    solver = cvxpy.MOSEK
    value = prob.solve(solver=solver,
                       verbose=verbose)
    point = tools.vec(y.value)
    dual_point = a.T @ point
    return dual_point, value


def _compute_bound_for_index(is_upper: bool,
                             ind: int,
                             h_ineq: np.ndarray) -> float:
    dim = h_ineq.shape[1] - 1
    b = np.vstack(h_ineq[:, 0])
    a = -1 * h_ineq[:, 1:]

    x = cvxpy.Variable(dim)
    constraints = [a @ x <= b.flatten()]
    if is_upper:
        objective_sense = cvxpy.Maximize
    else:
        objective_sense = cvxpy.Minimize
    objective = objective_sense(x[ind])
    prob = cvxpy.Problem(objective, constraints)

    try:
        verbose = False
        # verbose = True
        # solver = cvxpy.MOSEK
        solver = cvxpy.GUROBI
        prob.solve(solver=solver, verbose=verbose)
    except Exception as err:
        print(err)
    prob_value = prob.value
    assert prob_value is not None

    bound_for_index = prob_value
    return bound_for_index


def compute_coordinate_bounds(h_ineq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dim = h_ineq.shape[1] - 1

    lower = np.full((dim, ), -np.inf)
    upper = np.full((dim, ), +np.inf)

    for idx in range(dim):
        lower[idx] = _compute_bound_for_index(True, idx, h_ineq)
        upper[idx] = _compute_bound_for_index(False, idx, h_ineq)
    return lower, upper


def _is_full_dim_cvxpy(a: np.ndarray, b: np.ndarray) -> bool:
    dim = a.shape[1]
    nr = a.shape[0]
    one = np.ones((nr, 1))

    x = cvxpy.Variable(dim)
    x0 = cvxpy.Variable(1)
    constraints = [a @ x + one @ x0 <= b.flatten(), x0 <= 1]
    obj_fun = x0
    objective = cvxpy.Maximize(obj_fun)
    prob = cvxpy.Problem(objective, constraints)
    try:
        verbose = False
        # solver = cvxpy.MOSEK
        solver = cvxpy.GUROBI
        prob.solve(solver=solver, verbose=verbose)
    except Exception as err:
        print(err)
    prob_value = prob.value
    is_full_dim = prob_value > 0.0
    return is_full_dim


def _is_full_dim_gurobi(a: np.ndarray,
                        b: np.ndarray) -> bool:
    ncons, dim = a.shape
    is_verbose = False
    m = gurobipy.Model("matrix1")
    # m.setParam('OutputFlag', False)
    m.setParam('OutputFlag', is_verbose)

    x = m.addMVar(shape=dim + 1,
                  lb=-1 * gurobipy.GRB.INFINITY,
                  ub=+1 * gurobipy.GRB.INFINITY,
                  name="x")

    c = np.zeros((dim + 1, ))
    c[0] = 1.0

    a_aug = np.vstack((c, np.hstack((np.ones((ncons, 1)), a))))
    b_aug = np.hstack((np.ones(1), b.flatten()))

    m.setObjective(c.T @ x, gurobipy.GRB.MAXIMIZE)
    m.addConstr(a_aug @ x <= b_aug, name="c")
    m.optimize()
    # https://www.gurobi.com/documentation/9.0/refman/optimization_status_codes.html
    assert m.status in [GRB.OPTIMAL]

    # argmax = x.X
    prob_value = m.objVal
    is_full_dim = prob_value > 0.0
    return is_full_dim


def _h_form_full_dim(h_ineq: np.ndarray) -> bool:
    # https://inf.ethz.ch/personal/fukudak/lect/pclect/notes2015/PolyComp2015.pdf
    a = -1 * h_ineq[:, 1:]
    b = np.vstack(h_ineq[:, 0])
    # is_full_dim = _is_full_dim_cvxpy(a, b)
    is_full_dim = _is_full_dim_gurobi(a, b)
    return is_full_dim


def _is_empty_unbound(idx: int,
                      index_subsets: List[tuple],
                      index: tuple,
                      a: np.ndarray,
                      b: np.ndarray) -> bool:
    subset = index_subsets[idx]
    is_not_relued = np.in1d(index, subset)  # Type: np.ndarray
    inrf = is_not_relued.astype(float)
    not_relued_diag = np.diag(inrf)
    sign_cons = _build_sign_constraint_matrix(not_relued_diag)

    containment_cons = np.hstack((b, -a @ not_relued_diag))
    a_h_repr = np.vstack((sign_cons, containment_cons))

    is_nonempty = _h_form_full_dim(a_h_repr)
    # do_debug = True
    do_debug = False
    if do_debug:
        a_h_lin = np.empty((0, a_h_repr.shape[1]))
        v_repr = tools.h_to_v(a_h_repr, a_h_lin)
        if 0 == v_repr.shape[0]:
            poly_dim = 0  # meaning empty!
        else:
            poly_dim = np.linalg.matrix_rank(v_repr.T) - 1
        actual_dim = a.shape[1]
    ie = not is_nonempty
    return ie


def gurobi_update_calc(index_subsets: List[tuple],
                       index: tuple,
                       a: np.ndarray,
                       b: np.ndarray) -> List[bool]:
    do_tqdm_decoration = DO_TQDM_DECORATION
    num_subsets = len(index_subsets)
    dim = len(index)
    ncons = a.shape[0]
    is_verbose = False
    # test_save_model = True
    test_save_model = False

    # Computations common to all subsets
    c_sp = scipy.sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 1+dim))
    c = c_sp.toarray().flatten()

    m = gurobipy.Model("matrix1")
    x = m.addMVar(shape=dim + 1,
                  lb=-1 * gurobipy.GRB.INFINITY,
                  ub=+1 * gurobipy.GRB.INFINITY,
                  name="x")
    m.setObjective(c.T @ x, gurobipy.GRB.MAXIMIZE)
    m.setParam('OutputFlag', is_verbose)

    is_empty_list = [None] * num_subsets
    a_sparse = scipy.sparse.csr_matrix(a)
    ones_nc = np.ones((ncons, 1))
    ones_dim = np.ones((dim, 1))
    top_zero = scipy.sparse.csr_matrix((1, dim))

    b_0 = np.vstack((np.zeros((dim, 1)), b))
    b_1 = np.hstack((np.ones(1), b_0.flatten()))

    the_enumerate = enumerate(index_subsets)
    if do_tqdm_decoration:
        the_enumerate = tqdm.tqdm(the_enumerate, total=num_subsets, leave=False)

    written_filenames = []

    for idx, subset in the_enumerate:
        # idx = int(num_subsets / 2); subset = index_subsets[idx]
        is_not_relued = np.in1d(index, subset)  # Type: np.ndarray
        inrf = is_not_relued.astype(float)
        not_relued_diag = np.diag(inrf)

        sign_cons_sp = scipy.sparse.diags(inrf * 2 - 1,
                                          offsets=0,
                                          shape=(dim, dim),
                                          format="csr")
        a_1_sp = scipy.sparse.bmat(([1, top_zero],
                                    [ones_dim, -1 * sign_cons_sp],
                                    [ones_nc, a_sparse @ not_relued_diag],
                                    ), format="csr")
        m.remove(m.getConstrs())
        constr = a_1_sp @ x <= b_1
        m.addConstr(constr, name="c")

        # big_enough = a_1_sp.nnz >= 100
        # big_enough = a_1_sp.nnz >= 0
        # if test_save_model and big_enough:
        #     filename = "test_{}.mps".format(idx)
        #
        #     paths = path_config.get_paths()
        #     filedir = paths["work"]
        #     full_filename = os.path.join(filedir, filename)
        #     m.write(full_filename)
        #     written_filenames.append(full_filename)

        m.optimize()
        prob_value = m.objVal
        is_full_dim = prob_value > 0.0
        is_empty = not is_full_dim

        is_empty_list[idx] = is_empty

    # args = "TuneTimeLimit=10"
    # all_filenames = " ".join(written_filenames)
    # command = "grbtune {} {}".format(args, all_filenames)
    #
    # print("Run '{}'".format(command))
    return is_empty_list


def _compute_emptiness(index_subsets: List[tuple],
                       index: tuple,
                       a: np.ndarray,
                       b: np.ndarray) -> List[bool]:
    do_tqdm_decoration = DO_TQDM_DECORATION
    mininterval = .5
    maxinterval = 20

    # calc_type = "base"
    calc_type = "gurobi_update"
    if "gurobi_update" == calc_type:
        is_empty_list = gurobi_update_calc(index_subsets, index, a, b)
    elif "base" == calc_type:
        num_subsets = len(index_subsets)
        _is_empty_bound = functools.partial(_is_empty_unbound,
                                            index_subsets=index_subsets,
                                            index=index,
                                            a=a,
                                            b=b)
        map_over = range(num_subsets)

        # do_mapping = True
        do_mapping = False
        if do_mapping:
            # do_multiprocessing = True
            do_multiprocessing = False
            if do_multiprocessing:
                with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                    the_map = p.map(_is_empty_bound, map_over)
            else:
                the_map = map(_is_empty_bound, map_over)

            if do_tqdm_decoration:
                the_map = tqdm.tqdm(the_map, total=num_subsets, leave=False)
            is_empty_list = list(the_map)
        else:
            is_empty_list = [None] * num_subsets
            the_enumerate = enumerate(map_over)
            if do_tqdm_decoration:
                the_enumerate = tqdm.tqdm(the_enumerate, total=num_subsets, leave=False)

            for idx, val in the_enumerate:
                # idx = 0; val = map_over[idx]
                is_empty_list[idx] = _is_empty_bound(val)
    assert np.all(np.in1d(is_empty_list, [False, True]))
    return is_empty_list


def _v_to_h_not_rational_if_possible(h_ineq: np.ndarray,
                                     h_lin: np.ndarray) -> np.ndarray:
    try:
        v_repr = tools.h_to_v(h_ineq, h_lin, False)
    except Exception as e:
        v_repr = tools.h_to_v(h_ineq, h_lin, True)
    return v_repr


def _loop_body_unbound(idx_ie: Tuple[int, bool],
                       index_subsets: List[tuple],
                       index: tuple,
                       a: np.ndarray,
                       b: np.ndarray,
                       need_v: bool,
                       is_rational: bool) -> dict:
    idx = idx_ie[0]
    ie = idx_ie[1]
    subset = index_subsets[idx]
    is_not_relued = np.in1d(index, subset)  # Type: np.ndarray
    not_relued_diag = np.diag(is_not_relued).astype(float)
    sign_cons = _build_sign_constraint_matrix(not_relued_diag)

    containment_cons = np.hstack((b, -a @ not_relued_diag))
    all_cons = np.vstack((sign_cons, containment_cons))
    a_h_repr = drop_all_zero_rows(all_cons)

    n = a.shape[1]
    no_row_polytope = np.empty((0, n + 1))
    if ie:
        empty_h_polytope = np.hstack((-1 * np.eye(1), np.zeros((1, n))))
        _h = dict(inequality=empty_h_polytope, linear=no_row_polytope, is_empty=True)
        _v = dict(vertices=no_row_polytope, is_empty=True)
    else:
        if need_v:
            # v_repr = tools.h_to_v(a_h_repr, no_row_polytope, is_rational)
            v_repr = _v_to_h_not_rational_if_possible(a_h_repr, no_row_polytope)

        else:
            v_repr = None
        _h = dict(inequality=a_h_repr, linear=no_row_polytope, is_empty=False)
        _v = dict(vertices=v_repr, is_empty=v_repr is not None and v_repr.shape[0] == 0)
    _r = dict(h=_h, v=_v)
    return _r


def invert_relu_layer_kernel(p: Polytope,
                             need_v: bool,
                             is_rational: bool) -> Region:
    do_tqdm_decoration = DO_TQDM_DECORATION

    # https://borrelli.me.berkeley.edu/pdfpub/pub-12.pdf
    h = p["h"]
    v = p["v"]

    h_ineq = h["inequality"]
    b = tools.vec(h_ineq[:, 0])
    a = -1 * h_ineq[:, 1:]
    n = h_ineq.shape[1] - 1

    index = tuple(range(n))
    index_powerset = tools.powerset(index)

    index_subsets = index_powerset
    num_subsets = len(index_subsets)

    is_empty_list = _compute_emptiness(index_subsets, index, a, b)
    # logger.info("{}% empty".format(100 * np.mean(is_empty_list)))

    _loop_body_bound = functools.partial(_loop_body_unbound,
                                         index_subsets=index_subsets,
                                         index=index,
                                         a=a,
                                         b=b,
                                         need_v=need_v,
                                         is_rational=is_rational)
    do_mapping = False
    # do_mapping = True
    if do_mapping:
        do_multiprocessing = False
        # do_multiprocessing = True
        map_over = list(zip(range(num_subsets), is_empty_list))
        if do_multiprocessing:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                the_map = p.map(_loop_body_bound, map_over)
        else:
            the_map = map(_loop_body_bound, map_over)

        if do_tqdm_decoration:
            the_map = tqdm.tqdm(the_map, total=num_subsets, leave=False)
        r = list(the_map)
    else:
        r = [None] * num_subsets
        # _r = _loop_body_unbound(idx, index_subsets, index, a, b, need_v, is_rational)
        the_enumerate = enumerate(is_empty_list)
        if do_tqdm_decoration:
            the_enumerate = tqdm.tqdm(the_enumerate, total=num_subsets, leave=False)

        for idx, ie in the_enumerate:
            # idx = 0; ie = is_empty_list[idx]
            arg = (idx, ie)
            _r = _loop_body_bound(arg)
            r[idx] = _r
    return r

"""
Theorem 1. Consider a deep rectifier network with L layers, 
n_l rectified linear units at each layer l, and an input of 
dimension n_0.  The maximal number of regions of this neural 
network is at most:

sum_{\in J} \prod_{i = 1}^L {n_l \choose j_l}

where

J = {(j_1, ..., j_L) in Z^L: 
     0 <= j_l <= min(n_0, 
                     n_1 - j_1,
                     n_2 - j_2, 
                     ..., 
                     n_{l-1} - j_{l - 1}, 
                     n_l) for all l = 1, 2, ..., L}  
"""