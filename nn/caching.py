import hashlib
import pickle
import re
import logging
import os
import unicodedata
from typing import Any, Callable, Dict, List, Tuple, Union
import json

import hash_arbitrary

logger = logging.getLogger(__name__)

Args = Tuple[Any]
KwArgs = Dict[str, Any]


def _hash_kernel(x: Any) -> str:
    x_json = json.dumps(x, sort_keys=True)
    y = hashlib.sha256(x_json.encode('utf-8')).hexdigest()
    return y


def is_json_serializable(x: Any) -> bool:
    # https://stackoverflow.com/questions/42033142/is-there-an-easy-way-to-check-if-an-object-is-json-serializable-in-python
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def make_dict_json_serializable(p: Dict[str, Any]) -> Dict[str, Any]:
    ps = {k: v if is_json_serializable(v) else str(v) for (k, v) in p.items()}
    return ps


def hash_par(p: Dict[str, Any]) -> str:
    p_json_serializable = make_dict_json_serializable(p)
    return _hash_kernel(p_json_serializable)


def save_with_hash(data: Any,
                   fullfilename: str,
                   data_hash: str) -> None:
    to_pickle = {"data": data, "data_hash": data_hash}
    with open(fullfilename, 'wb') as pklfile:
        pickle.dump(to_pickle, pklfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_with_hash(fullfilename: str,
                   data_hash: str) -> Any:
    try:
        with open(fullfilename, 'rb') as pklfile:
            loaded = pickle.load(pklfile)
            assert loaded["data_hash"] == data_hash
            data = loaded["data"]
    except Exception as e:
        data = None
    return data


def load_if_present(fullfilename: str) -> Any:
    do_error_log = False
    try:
        with open(fullfilename, 'rb') as pklfile:
            loaded = pickle.load(pklfile)
    except Exception as e:
        if do_error_log:
            logger.error(e)
        loaded = None
    return loaded


def inspect_cached_file(fullfilename: str) -> None:
    with open(fullfilename, 'rb') as pklfile:
        loaded = pickle.load(pklfile)
    print(loaded.keys())


def _slugify_text(s: str) -> str:
    slug = unicodedata.normalize('NFKD', s)
    slug = slug.encode('ascii', 'ignore').lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug).strip('-')
    slug = re.sub(r'[-]+', '-', slug)
    return slug


def _build_cache_fullfilename(cache_dir: str,
                              calc_fun: Callable,
                              calc_args: Args,
                              calc_kwargs: KwArgs) -> str:
    fun_ident = os.path.join(calc_fun.__module__, calc_fun.__name__)

    calc_prngs = (1, 1, 1)
    args_hash = hash_arbitrary.hash(calc_args)
    kwargs_hash = hash_arbitrary.hash(calc_kwargs)

    prngs_hash = hash_arbitrary.hash(calc_prngs)
    filename = "{}_{}_{}.pkl".format(args_hash, kwargs_hash, prngs_hash)

    calc_full_dir = os.path.join(cache_dir, fun_ident)
    os.makedirs(calc_full_dir, exist_ok=True)
    calc_fullfilename = os.path.join(calc_full_dir, filename)
    return calc_fullfilename


def cached_calc(cache_dir: str,
                calc_fun: Callable,
                calc_args: Args,
                calc_kwargs: KwArgs,
                force_regeneraton: bool) -> Any:
    calc_fullfilename = _build_cache_fullfilename(cache_dir, calc_fun, calc_args, calc_kwargs)
    do_logging = False
    calc = load_if_present(calc_fullfilename)
    calc_is_present = not (calc is None)

    if not calc_is_present or force_regeneraton:
        if do_logging:
            logger.info("Regenerating {}".format(calc_fullfilename))
        calc = calc_fun(*calc_args, **calc_kwargs)
        if do_logging:
            logger.info("Saving results to {}".format(calc_fullfilename))

        with open(calc_fullfilename, 'wb') as pklfile:
            pickle.dump(calc, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if do_logging:
            logger.info("Loading {} from cache".format(calc_fullfilename))
    return calc


def clear_cache(cache_dir: str,
                calc_fun: Callable,
                calc_args: Args,
                calc_kwargs: KwArgs) -> None:
    calc_fullfilename = _build_cache_fullfilename(cache_dir,
                                                  calc_fun,
                                                  calc_args,
                                                  calc_kwargs)
    os.rmdir(calc_fullfilename)


if __name__ == "__main__":
    import numpy as np
    a = np.ones((10, ))
    _hash_kernel(a)