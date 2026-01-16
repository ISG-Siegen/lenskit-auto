# test_funksvd_strict.py
import inspect
import unittest
from typing import Dict, Any

import ConfigSpace as CS

from lkauto.algorithms.funksvd import FunkSVD


# -----------------------
# Helpers
# -----------------------
def discover_lenskit_config_fields() -> Dict[str, Dict[str, Any]]:
    """
    Discover lenskit.funksvd.FunkSVDConfig fields (pydantic or fallback to signature).
    Returns mapping: name -> {"annotation": str_or_None, "default": value_or_None, "required": bool}
    Raises RuntimeError if lenskit.funksvd cannot be imported.
    """
    import importlib

    try:
        mod = importlib.import_module("lenskit.funksvd")
    except Exception as e:
        raise RuntimeError("Could not import lenskit.funksvd; ensure lenskit is installed in the test environment.") from e

    # Prefer FunkSVDConfig (pydantic-style)
    cfg = getattr(mod, "FunkSVDConfig", None)
    if cfg is not None:
        # pydantic v2
        if hasattr(cfg, "model_fields"):
            fields = {}
            for n, mf in cfg.model_fields.items():
                annotation = getattr(mf, "annotation", None)
                default = getattr(mf, "default", None)
                required = getattr(mf, "required", False)
                fields[n] = {
                    "annotation": getattr(annotation, "__name__", str(annotation)) if annotation is not None else None,
                    "default": default,
                    "required": required,
                }
            return fields

        # pydantic v1
        if hasattr(cfg, "__fields__"):
            fields = {}
            for n, f in cfg.__fields__.items():
                ann = getattr(f, "type_", None)
                default = getattr(f, "default", None)
                required = getattr(f, "required", False)
                fields[n] = {
                    "annotation": getattr(ann, "__name__", str(ann)) if ann is not None else None,
                    "default": default,
                    "required": required,
                }
            return fields

        # fallback to class annotations
        if hasattr(cfg, "__annotations__"):
            fields = {}
            for n, a in cfg.__annotations__.items():
                default = getattr(cfg, n, None)
                fields[n] = {
                    "annotation": getattr(a, "__name__", str(a)) if a is not None else None,
                    "default": default,
                    "required": default is None,
                }
            return fields

    # fallback: inspect FunkSVDScorer or FunkSVD __init__ signature
    cls = getattr(mod, "FunkSVDScorer", None) or getattr(mod, "FunkSVD", None)
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
        fields[name] = {
            "annotation": getattr(ann, "__name__", str(ann)) if ann is not None else None,
            "default": default,
            "required": required,
        }
    return fields


def cs_get_hyperparameters(cs: CS.ConfigurationSpace):
    """Return list of hyperparameters from ConfigurationSpace using supported API (list(cs.values()))."""
    try:
        return list(cs.values())
    except Exception:
        # fallback for very old ConfigSpace versions
        if hasattr(cs, "get_hyperparameters"):
            return cs.get_hyperparameters()
    raise RuntimeError("Could not extract hyperparameters from ConfigurationSpace.")


def cs_hyperparam_summary(hp) -> Dict[str, Any]:
    """
    Summarize a ConfigSpace hyperparameter into a dict for comparison.
    Keys produced: type, lower (opt), upper (opt), default_value (opt), log (opt), choices (opt)
    """
    summary = {"type": type(hp).__name__}
    for attr in ("lower", "upper", "default_value", "log"):
        if hasattr(hp, attr):
            summary[attr] = getattr(hp, attr)
    if hasattr(hp, "choices"):
        summary["choices"] = list(hp.choices)
    return summary


def approx_equal(a, b, rel=1e-6, abs_tol=1e-12):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_tol)
    except Exception:
        return a == b


# -----------------------
# Tests
# -----------------------
class TestFunkSVDStrict(unittest.TestCase):
    def test_init_givenObjectInitialized_ObjectInitializedCorrectlyExpected(self):
        features = 100
        funk_svd = FunkSVD(features)
        self.assertIsInstance(funk_svd, FunkSVD)
        self.assertEqual(features, funk_svd.features)

    def test_configspace_exactly_matches_lenskit(self):
        """
        Strict test: project FunkSVD ConfigurationSpace must match LensKit's FunkSVDConfig fields,
        except for 'epochs' and 'range' which are allowed to be absent from the project's CS.
        """
        # discover LensKit fields
        lk_fields = discover_lenskit_config_fields()
        if not lk_fields:
            self.fail("Failed to discover lenskit.funksvd.FunkSVDConfig fields. Ensure lenskit is importable in the test environment.")

        # our CS
        cs = FunkSVD.get_default_configspace()
        cs_hps = {hp.name: cs_hyperparam_summary(hp) for hp in cs_get_hyperparameters(cs)}

        lk_names = set(lk_fields.keys())
        cs_names = set(cs_hps.keys())

        # Allow these LensKit fields to be missing from the project's CS (training-only or non-learnable)
        allowed_missing = {"epochs", "range"}

        # compute missing/extra after allowing allowed_missing
        missing = sorted(list(lk_names - cs_names - allowed_missing))
        unexpected = sorted(list(cs_names - lk_names))

        if missing or unexpected:
            msg_lines = ["Configuration hyperparameter NAME mismatch with LensKit FunkSVDConfig:"]
            if missing:
                msg_lines.append(f"  - Missing in project CS (present in LensKit): {missing}")
            if unexpected:
                msg_lines.append(f"  - Unexpected in project CS (not in LensKit): {unexpected}")
            msg_lines.append(f"LensKit fields: {sorted(lk_names)}")
            msg_lines.append(f"Project CS fields: {sorted(cs_names)}")
            msg_lines.append(f"Allowed missing LensKit fields (will not cause failure): {sorted(allowed_missing)}")
            self.fail("\n".join(msg_lines))

        # 2) coarse type family + default checks for the remaining (non-allowed-missing) fields
        diffs = []
        for name in sorted(lk_names - allowed_missing):
            lk_meta = lk_fields[name]
            cs_meta = cs_hps.get(name)
            # cs_meta must exist because we removed allowed_missing above
            if cs_meta is None:
                diffs.append(f"{name}: missing in CS (unexpected after allowed_missing removal)")
                continue

            ann = (lk_meta.get("annotation") or "").lower()
            cs_type = cs_meta["type"].lower()

            # coarse type check
            type_ok = True
            if "int" in ann or "gt=" in str(lk_meta.get("annotation", "")).lower():
                if "integer" not in cs_type:
                    type_ok = False
            elif "float" in ann or "double" in ann or name in ("learning_rate", "regularization"):
                if "float" not in cs_type:
                    type_ok = False
            elif "bool" in ann or "literal" in ann or "enum" in ann:
                if "categorical" not in cs_type and "constant" not in cs_type:
                    type_ok = False
            # else: skip strict checks for complex annotations

            if not type_ok:
                diffs.append(f"{name}: type mismatch -> lenskit annotation='{lk_meta.get('annotation')}' vs CS type='{cs_meta['type']}'")

            # defaults
            lk_default = lk_meta.get("default")
            cs_default = cs_meta.get("default_value", None)

            if lk_meta.get("required", False):
                # LensKit requires a value -> expect no default in CS
                if cs_default is not None:
                    diffs.append(f"{name}: LensKit requires parameter (no default) but CS provides default {cs_default!r}")
            else:
                if lk_default is not None:
                    if isinstance(lk_default, float):
                        if not approx_equal(lk_default, cs_default):
                            diffs.append(f"{name}: default mismatch -> lenskit={lk_default!r} cs={cs_default!r}")
                    else:
                        if lk_default != cs_default:
                            diffs.append(f"{name}: default mismatch -> lenskit={lk_default!r} cs={cs_default!r}")

        if diffs:
            msg = ["Configuration attribute mismatches with LensKit FunkSVDConfig:"]
            msg.extend("  - " + d for d in diffs)
            msg.append("")
            msg.append("LensKit fields and metadata:")
            for n in sorted(lk_names - allowed_missing):
                msg.append(f"  {n}: {lk_fields[n]}")
            msg.append("")
            msg.append("Project CS hyperparameters:")
            for n in sorted(cs_names):
                msg.append(f"  {n}: {cs_hps[n]}")
            self.fail("\n".join(msg))


if __name__ == "__main__":
    unittest.main()
