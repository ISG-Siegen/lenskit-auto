# test_bias_strict.py
import inspect
import unittest
from typing import Dict, Any

import ConfigSpace as CS

from lkauto.algorithms.bias import Bias


# Helpers (shared style with ALS tests)
def discover_lenskit_config_fields() -> Dict[str, Dict[str, Any]]:
    """
    Discover LensKit's BiasConfig fields (pydantic or fallback to signature).
    Returns mapping: name -> {"annotation": str_or_None, "default": value_or_None, "required": bool}
    """
    import importlib

    try:
        mod = importlib.import_module("lenskit.basic.bias")
    except Exception as e:
        raise RuntimeError("Could not import lenskit.basic.bias; ensure 'lenskit' is installed in the test environment.") from e

    # Prefer BiasConfig (pydantic)
    cfg = getattr(mod, "BiasConfig", None)
    if cfg is not None:
        # pydantic v2
        if hasattr(cfg, "model_fields"):
            fields = {}
            for n, mf in cfg.model_fields.items():
                annotation = getattr(mf, "annotation", None)
                default = getattr(mf, "default", None)
                required = getattr(mf, "required", False)
                fields[n] = {"annotation": getattr(annotation, "__name__", str(annotation)) if annotation is not None else None,
                             "default": default,
                             "required": required}
            return fields

        # pydantic v1
        if hasattr(cfg, "__fields__"):
            fields = {}
            for n, f in cfg.__fields__.items():
                ann = getattr(f, "type_", None)
                default = getattr(f, "default", None)
                required = getattr(f, "required", False)
                fields[n] = {"annotation": getattr(ann, "__name__", str(ann)) if ann is not None else None,
                             "default": default,
                             "required": required}
            return fields

        # fallback to class annotations
        if hasattr(cfg, "__annotations__"):
            fields = {}
            for n, a in cfg.__annotations__.items():
                default = getattr(cfg, n, None)
                fields[n] = {"annotation": getattr(a, "__name__", str(a)) if a is not None else None,
                             "default": default,
                             "required": default is None}
            return fields

    # fallback: inspect BiasScorer signature
    cls = getattr(mod, "BiasScorer", None)
    if cls is None:
        return {}

    sig = inspect.signature(cls.__init__)
    fields = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        ann = param.annotation if param.annotation is not inspect._empty else None
        default = None if param.default is inspect._empty else param.default
        required = param.default is inspect._empty
        fields[name] = {"annotation": getattr(ann, "__name__", str(ann)) if ann is not None else None,
                        "default": default,
                        "required": required}
    return fields


def cs_get_hyperparameters(cs: CS.ConfigurationSpace):
    """Return hyperparameters without using deprecated API."""
    try:
        return list(cs.values())
    except Exception:
        if hasattr(cs, "get_hyperparameters"):
            return cs.get_hyperparameters()
    raise RuntimeError("Could not extract hyperparameters from ConfigurationSpace.")


def approx_equal(a, b, rel=1e-6, abs_tol=1e-12):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_tol)
    except Exception:
        return a == b


class TestBiasStrict(unittest.TestCase):
    def test_init_givenObjectInitialized_ObjectInitializedCorrectlyExpected(self):
        bias = Bias()
        self.assertIsInstance(bias, Bias)

    def test_configspace_matches_lenskit_bias_config(self):
        """
        Ensure our Bias.get_default_configspace(number_item, number_user) provides
        configuration equivalent to LensKit's BiasConfig.damping expectation.

        LensKit BiasConfig currently exposes 'damping' (float or dict['user'|'item': float]).
        Our code historically exposes separate 'item_damping' and 'user_damping' hyperparams.
        Accept either:
          - a 'damping' hyperparameter in CS (preferred, one-to-one), OR
          - both 'item_damping' and 'user_damping' that scale according to the implemented formula.
        """
        # discover LensKit fields
        lk_fields = discover_lenskit_config_fields()
        if not lk_fields:
            self.fail("Failed to discover lenskit.basic.bias.BiasConfig fields. Ensure lenskit is importable in the test environment.")

        # check LensKit has 'damping' (it should, per docs)
        lenskit_has_damping = "damping" in lk_fields

        # create our CS for some counts
        number_item = 10
        number_user = 100
        cs = Bias.get_default_configspace(number_item=number_item, number_user=number_user)
        self.assertIsInstance(cs, CS.ConfigurationSpace)
        params = cs_get_hyperparameters(cs)
        pmap = {p.name: p for p in params}

        # If LensKit expects 'damping', accept either a single 'damping' HP or pair 'item_damping'/'user_damping'
        if lenskit_has_damping:
            # Case A: single 'damping' hyperparameter present in our CS
            if "damping" in pmap:
                # basic sanity: make sure it's numeric (float) or categorical allowing dict-like representations is not easy;
                # we at least ensure some attributes are present
                hp = pmap["damping"]
                self.assertTrue(hasattr(hp, "default_value") or hasattr(hp, "choices"),
                                msg=f"'damping' hyperparameter present but doesn't look numeric/categorical: {hp}")
                return  # test passes (we found damping)
            # Case B: we expose 'item_damping' and 'user_damping' (legacy)
            if "item_damping" in pmap and "user_damping" in pmap:
                item_hp = pmap["item_damping"]
                user_hp = pmap["user_damping"]

                # expected scaling from implementation:
                exp_item_lower = 1e-5 * number_item
                exp_item_upper = 85 * number_item
                exp_item_default = 0.0025 * number_item

                exp_user_lower = 1e-5 * number_user
                exp_user_upper = 85 * number_user
                exp_user_default = 0.0025 * number_user

                self.assertTrue(approx_equal(getattr(item_hp, "lower", None), exp_item_lower),
                                msg=f"item_damping.lower expected {exp_item_lower} got {getattr(item_hp, 'lower', None)}")
                self.assertTrue(approx_equal(getattr(item_hp, "upper", None), exp_item_upper),
                                msg=f"item_damping.upper expected {exp_item_upper} got {getattr(item_hp, 'upper', None)}")
                self.assertTrue(approx_equal(getattr(item_hp, "default_value", None), exp_item_default),
                                msg=f"item_damping.default expected {exp_item_default} got {getattr(item_hp, 'default_value', None)}")
                self.assertTrue(getattr(item_hp, "log", False), msg="item_damping should be log=True")

                self.assertTrue(approx_equal(getattr(user_hp, "lower", None), exp_user_lower),
                                msg=f"user_damping.lower expected {exp_user_lower} got {getattr(user_hp, 'lower', None)}")
                self.assertTrue(approx_equal(getattr(user_hp, "upper", None), exp_user_upper),
                                msg=f"user_damping.upper expected {exp_user_upper} got {getattr(user_hp, 'upper', None)}")
                self.assertTrue(approx_equal(getattr(user_hp, "default_value", None), exp_user_default),
                                msg=f"user_damping.default expected {exp_user_default} got {getattr(user_hp, 'default_value', None)}")
                self.assertTrue(getattr(user_hp, "log", False), msg="user_damping should be log=True")

                return  # test passes (legacy pair matches)
            # Neither damping nor item/user damping present -> fail
            self.fail("LensKit expects a 'damping' parameter, but project CS has neither 'damping' nor both 'item_damping' and 'user_damping'. CS params: " + ", ".join(sorted(pmap.keys())))
        else:
            # If LensKit does not have damping (unlikely per docs), ensure project exposes item/user damping at least
            missing = []
            if "item_damping" not in pmap:
                missing.append("item_damping")
            if "user_damping" not in pmap:
                missing.append("user_damping")
            if missing:
                self.fail("LensKit BiasConfig does not expose 'damping' (unexpected), and project CS is missing: " + ", ".join(missing))

if __name__ == "__main__":
    unittest.main()
