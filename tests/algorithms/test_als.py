# test_als_strict.py
import inspect
import unittest
from typing import Dict, Any

import ConfigSpace as CS

from lkauto.algorithms.als import ImplicitMF, BiasedMF


# -----------------------
# Helpers to inspect LensKit and ConfigSpace
# -----------------------
def discover_lenskit_config_fields() -> Dict[str, Dict[str, Any]]:
    """
    Discover LensKit's BiasedMF/ALS config fields and metadata.
    Returns mapping: name -> {"annotation": str_or_None, "default": value_or_None, "required": bool}
    Raises RuntimeError if lenskit cannot be imported.
    """
    import importlib
    try:
        mod = importlib.import_module("lenskit.als")
    except Exception as e:
        raise RuntimeError("Could not import lenskit.als; ensure 'lenskit' is installed in the test environment.") from e

    # Prefer pydantic-style config classes
    for cfg_name in ("BiasedMFConfig", "ALSConfig"):
        cfg = getattr(mod, cfg_name, None)
        if cfg is None:
            continue

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

    # Fallback: inspect class __init__ signature
    cls = getattr(mod, "BiasedMF", None) or getattr(mod, "ALS", None)
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
    """Return list of hyperparameters from ConfigurationSpace in a uniform manner."""
    if hasattr(cs, "get_hyperparameters"):
        return cs.get_hyperparameters()
    return list(cs.values())


def cs_hyperparam_summary(hp) -> Dict[str, Any]:
    """Summarize a ConfigSpace hyperparameter for comparison."""
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
class TestImplicitMF(unittest.TestCase):
    def test_init_givenObjectInitialized_ObjectInitializedCorrectlyExpected(self):
        expected_features = 10
        implicit_mf = ImplicitMF(expected_features)
        self.assertIsInstance(implicit_mf, ImplicitMF)
        self.assertEqual(expected_features, implicit_mf.features)

    def test_configspace_exactly_matches_lenskit_for_implicit(self):
        """
        Strict test: project ImplicitMF ConfigurationSpace must exactly match LensKit's ALS/ImplicitMF config fields.
        Fails if names, coarse types, or defaults differ.
        """
        lk_fields = discover_lenskit_config_fields()
        if not lk_fields:
            self.fail("Failed to discover LensKit config fields. Ensure lenskit is importable in the test environment.")

        cs = ImplicitMF.get_default_configspace()
        cs_hps = {hp.name: cs_hyperparam_summary(hp) for hp in cs_get_hyperparameters(cs)}

        lk_names = set(lk_fields.keys())
        cs_names = set(cs_hps.keys())

        # exact name equality
        if lk_names != cs_names:
            missing = sorted(list(lk_names - cs_names))
            unexpected = sorted(list(cs_names - lk_names))
            msg_lines = ["ImplicitMF Configuration NAME mismatch with LensKit:"]
            if missing:
                msg_lines.append(f"  - Missing in project CS (present in LensKit): {missing}")
            if unexpected:
                msg_lines.append(f"  - Unexpected in project CS (not in LensKit): {unexpected}")
            msg_lines.append(f"LensKit fields: {sorted(lk_names)}")
            msg_lines.append(f"Project CS fields: {sorted(cs_names)}")
            self.fail("\n".join(msg_lines))

        # type family + default checks
        diffs = []
        for name in sorted(lk_names):
            lk_meta = lk_fields[name]
            cs_meta = cs_hps[name]

            ann = (lk_meta.get("annotation") or "").lower()
            cs_type = cs_meta["type"].lower()

            # coarse type check
            type_ok = True
            if "int" in ann or "gt=" in str(lk_meta.get("annotation", "")).lower():
                if "integer" not in cs_type:
                    type_ok = False
            elif "float" in ann or "double" in ann:
                if "float" not in cs_type:
                    type_ok = False
            elif "bool" in ann or "literal" in ann or "enum" in ann:
                if "categorical" not in cs_type and "constant" not in cs_type:
                    type_ok = False

            if not type_ok:
                diffs.append(f"{name}: type mismatch -> lenskit annotation='{lk_meta.get('annotation')}' vs CS type='{cs_meta['type']}'")

            # default checks
            lk_default = lk_meta.get("default")
            cs_default = cs_meta.get("default_value", None)

            if lk_meta.get("required", False):
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

            # boolean/literal -> ensure categorical choices exist
            if "bool" in (lk_meta.get("annotation") or "").lower() or "literal" in (lk_meta.get("annotation") or "").lower():
                choices = cs_meta.get("choices")
                if not choices:
                    diffs.append(f"{name}: expected CS to be categorical (choices) but none found; CS type={cs_meta['type']}")
                else:
                    if lk_default is not None and lk_default not in choices:
                        diffs.append(f"{name}: default {lk_default!r} not in CS choices {choices!r}")

        if diffs:
            msg = ["ImplicitMF Configuration attribute mismatches with LensKit:"]
            msg.extend("  - " + d for d in diffs)
            msg.append("")
            msg.append("LensKit fields and metadata:")
            for n in sorted(lk_names):
                msg.append(f"  {n}: {lk_fields[n]}")
            msg.append("")
            msg.append("Project CS hyperparameters:")
            for n in sorted(cs_names):
                msg.append(f"  {n}: {cs_hps[n]}")
            self.fail("\n".join(msg))


class TestBiasedMF(unittest.TestCase):
    def test_init_givenObjectInitialized_ObjectInitializedCorrectlyExpected(self):
        expected_features = 10
        biased_mf = BiasedMF(expected_features)
        self.assertIsInstance(biased_mf, BiasedMF)
        self.assertEqual(expected_features, biased_mf.features)

    def test_configspace_exactly_matches_lenskit_for_biased(self):
        """
        Strict test: project BiasedMF ConfigurationSpace must exactly match LensKit's BiasedMF/ALS config fields.
        Fails if names, coarse types, or defaults differ.
        """
        lk_fields = discover_lenskit_config_fields()
        if not lk_fields:
            self.fail("Failed to discover LensKit config fields. Ensure lenskit is importable in the test environment.")

        cs = BiasedMF.get_default_configspace()
        cs_hps = {hp.name: cs_hyperparam_summary(hp) for hp in cs_get_hyperparameters(cs)}

        lk_names = set(lk_fields.keys())
        cs_names = set(cs_hps.keys())

        # exact name equality
        if lk_names != cs_names:
            missing = sorted(list(lk_names - cs_names))
            unexpected = sorted(list(cs_names - lk_names))
            msg_lines = ["BiasedMF Configuration NAME mismatch with LensKit:"]
            if missing:
                msg_lines.append(f"  - Missing in project CS (present in LensKit): {missing}")
            if unexpected:
                msg_lines.append(f"  - Unexpected in project CS (not in LensKit): {unexpected}")
            msg_lines.append(f"LensKit fields: {sorted(lk_names)}")
            msg_lines.append(f"Project CS fields: {sorted(cs_names)}")
            self.fail("\n".join(msg_lines))

        # type family + default checks
        diffs = []
        for name in sorted(lk_names):
            lk_meta = lk_fields[name]
            cs_meta = cs_hps[name]

            ann = (lk_meta.get("annotation") or "").lower()
            cs_type = cs_meta["type"].lower()

            # coarse type check
            type_ok = True
            if "int" in ann or "gt=" in str(lk_meta.get("annotation", "")).lower():
                if "integer" not in cs_type:
                    type_ok = False
            elif "float" in ann or "double" in ann:
                if "float" not in cs_type:
                    type_ok = False
            elif "bool" in ann or "literal" in ann or "enum" in ann:
                if "categorical" not in cs_type and "constant" not in cs_type:
                    type_ok = False

            if not type_ok:
                diffs.append(f"{name}: type mismatch -> lenskit annotation='{lk_meta.get('annotation')}' vs CS type='{cs_meta['type']}'")

            # default checks
            lk_default = lk_meta.get("default")
            cs_default = cs_meta.get("default_value", None)

            if lk_meta.get("required", False):
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

            # boolean/literal -> ensure categorical choices exist
            if "bool" in (lk_meta.get("annotation") or "").lower() or "literal" in (lk_meta.get("annotation") or "").lower():
                choices = cs_meta.get("choices")
                if not choices:
                    diffs.append(f"{name}: expected CS to be categorical (choices) but none found; CS type={cs_meta['type']}")
                else:
                    if lk_default is not None and lk_default not in choices:
                        diffs.append(f"{name}: default {lk_default!r} not in CS choices {choices!r}")

        if diffs:
            msg = ["BiasedMF Configuration attribute mismatches with LensKit:"]
            msg.extend("  - " + d for d in diffs)
            msg.append("")
            msg.append("LensKit fields and metadata:")
            for n in sorted(lk_names):
                msg.append(f"  {n}: {lk_fields[n]}")
            msg.append("")
            msg.append("Project CS hyperparameters:")
            for n in sorted(cs_names):
                msg.append(f"  {n}: {cs_hps[n]}")
            self.fail("\n".join(msg))


if __name__ == "__main__":
    unittest.main()
