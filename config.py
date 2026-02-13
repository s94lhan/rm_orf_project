
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DGPConfig:
    name: str
    p: int
    is_rct: bool
    clip_a: Optional[float]
    clip_b: Optional[float]
    alignment: Optional[str]
    tau_type: str
    sigma: float


def make_dgp_configs() -> Dict[str, DGPConfig]:
    """创建所有DGP配置"""
    return {
        "DGP0": DGPConfig("DGP0", p=5, is_rct=False, clip_a=0.2, clip_b=0.8, alignment="high", tau_type="strong", sigma=1.0),
        "DGP1": DGPConfig("DGP1", p=5, is_rct=False, clip_a=0.02, clip_b=0.98, alignment="high", tau_type="strong", sigma=1.0),
        "DGP2": DGPConfig("DGP2", p=5, is_rct=False, clip_a=0.2, clip_b=0.8, alignment="low", tau_type="strong", sigma=1.0),
        "DGP3": DGPConfig("DGP3", p=5, is_rct=False, clip_a=0.02, clip_b=0.98, alignment="low", tau_type="strong", sigma=1.0),
        "DGP4": DGPConfig("DGP4", p=5, is_rct=False, clip_a=0.2, clip_b=0.8, alignment="high", tau_type="strong", sigma=2.0),
        "DGP5": DGPConfig("DGP5", p=5, is_rct=False, clip_a=0.2, clip_b=0.8, alignment="high", tau_type="weak", sigma=1.0),
        "DGP6": DGPConfig("DGP6", p=5, is_rct=False, clip_a=0.2, clip_b=0.8, alignment="high", tau_type="threshold", sigma=1.0),
        "DGP7": DGPConfig("DGP7", p=50, is_rct=False, clip_a=0.2, clip_b=0.8, alignment="high", tau_type="strong", sigma=1.0),
        "DGP8": DGPConfig("DGP8", p=100, is_rct=False, clip_a=0.02, clip_b=0.98, alignment="low", tau_type="strong", sigma=1.0),
        "RCT": DGPConfig("RCT", p=5, is_rct=True, clip_a=None, clip_b=None, alignment=None, tau_type="strong", sigma=1.0),
    }



DEFAULT_N_LIST = [250, 500, 1000, 2000]
DEFAULT_METHODS = ["ORF", "CF", "DML-CF"]
DEFAULT_R = 50
DEFAULT_N_TEST = 500


DEFAULT_NJOBS = 1


MAIN_N_ESTIMATORS = 300
MAIN_SUBSAMPLE = 0.5
MAIN_CV = 5


NUIS_N_ESTIMATORS = 200

DEFAULT_SEED = 123