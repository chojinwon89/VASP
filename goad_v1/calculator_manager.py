"""
Calculator management for GOAD v1.0

Handles different MatterSim, SevenNet, CHGNet, and MACE calculator configurations

MatterSim v1.2.3 note
---------------------
In v1.2.3, Potential.from_checkpoint only accepts model_name="m3gnet".
The 5M checkpoint must be selected via load_path, not model_name.
    1M: MatterSimCalculator()                                    (default)
    5M: MatterSimCalculator(load_path="mattersim-v1.0.0-5M")
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CalculatorManager:
    """Manage different ASE calculator configurations"""

    # ---------------------- MatterSim ----------------------
    @staticmethod
    def get_mattersim_1m():
        from mattersim.forcefield import MatterSimCalculator
        logger.info("Loading MatterSim 1M calculator...")
        calc = MatterSimCalculator()
        logger.info("✓ MatterSim 1M loaded successfully")
        return calc

    @staticmethod
    def get_mattersim_5m():
        from mattersim.forcefield import MatterSimCalculator
        logger.info("Loading MatterSim 5M calculator...")
        # v1.2.3: model_name must be "m3gnet"; select 5M via load_path
        calc = MatterSimCalculator(load_path="mattersim-v1.0.0-5M")
        logger.info("✓ MatterSim 5M loaded successfully")
        return calc

    @staticmethod
    def get_mattersim_5m_d3():
        from mattersim.forcefield import MatterSimCalculator
        logger.info("Loading MatterSim 5M + D3 calculator...")
        try:
            calc = MatterSimCalculator(load_path="mattersim-v1.0.0-5M", use_d3=True)
            logger.info("✓ MatterSim 5M + D3 loaded successfully")
            return calc
        except Exception as e:
            logger.error(f"MatterSim 5M+D3 failed: {e!r}", exc_info=True)
            raise

    # ---------------------- SevenNet ----------------------
    @staticmethod
    def get_sevennet_omni(modal: str = "omat24"):
        """SevenNet-OMNI (7net-mf-ompa). modal='omat24' (PBE+D3) or 'mpa' (PBE)."""
        from sevenn.sevennet_calculator import SevenNetCalculator
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SevenNet-OMNI (7net-mf-ompa, modal={modal}) on {device}...")
        calc = SevenNetCalculator("7net-mf-ompa", modal=modal, device=device)
        logger.info(f"✓ SevenNet-OMNI ({modal}) loaded successfully")
        return calc

    @staticmethod
    def get_sevennet_omat():
        from sevenn.sevennet_calculator import SevenNetCalculator
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SevenNet-OMat (7net-omat) on {device}...")
        calc = SevenNetCalculator("7net-omat", device=device)
        logger.info("✓ SevenNet-OMat loaded successfully")
        return calc

    # ---------------------- CHGNet ----------------------
    @staticmethod
    def get_chgnet():
        """CHGNet universal interatomic potential (v0.3.0)."""
        from chgnet.model.dynamics import CHGNetCalculator
        logger.info("Loading CHGNet calculator...")
        calc = CHGNetCalculator()
        logger.info("✓ CHGNet loaded successfully")
        return calc

    # ---------------------- MACE ----------------------
    @staticmethod
    def get_mace_mp():
        """MACE-MP-0 universal potential (medium model, no dispersion)."""
        from mace.calculators import mace_mp
        logger.info("Loading MACE-MP-0 (medium) calculator...")
        calc = mace_mp(model="medium", dispersion=False, default_dtype="float32")
        logger.info("✓ MACE-MP-0 loaded successfully")
        return calc

    @staticmethod
    def get_mace_mp_d3():
        """MACE-MP-0 universal potential (medium model) + D3 dispersion."""
        from mace.calculators import mace_mp
        logger.info("Loading MACE-MP-0 (medium) + D3 calculator...")
        calc = mace_mp(model="medium", dispersion=True, default_dtype="float32")
        logger.info("✓ MACE-MP-0 + D3 loaded successfully")
        return calc

    @staticmethod
    def get_mace_off():
        """MACE-OFF23 organic force field (medium model)."""
        from mace.calculators import mace_off
        logger.info("Loading MACE-OFF23 (medium) calculator...")
        calc = mace_off(model="medium", default_dtype="float32")
        logger.info("✓ MACE-OFF23 loaded successfully")
        return calc

    # ---------------------- Dispatcher ----------------------
    @staticmethod
    def get_calculator(calculator_type: str = "1m"):
        t = calculator_type.lower().strip().replace("+", "_").replace("-", "_")

        if t in ("1m",):                            return CalculatorManager.get_mattersim_1m()
        if t in ("5m",):                            return CalculatorManager.get_mattersim_5m()
        if t in ("5m_d3", "5md3"):                  return CalculatorManager.get_mattersim_5m_d3()

        if t in ("sevennet_omni", "7net_omni", "7net", "omni", "7net_mf_ompa",
                 "sevennet_omni_omat24", "7net_omni_omat24"):
            return CalculatorManager.get_sevennet_omni(modal="omat24")
        if t in ("sevennet_omni_mpa", "7net_omni_mpa"):
            return CalculatorManager.get_sevennet_omni(modal="mpa")
        if t in ("sevennet_omat", "7net_omat"):
            return CalculatorManager.get_sevennet_omat()

        if t in ("chgnet",):                        return CalculatorManager.get_chgnet()

        if t in ("mace", "mace_mp", "mace_mp0"):   return CalculatorManager.get_mace_mp()
        if t in ("mace_d3", "mace_mp_d3"):          return CalculatorManager.get_mace_mp_d3()
        if t in ("mace_off", "mace_off23"):         return CalculatorManager.get_mace_off()

        raise ValueError(
            f"Unknown calculator type: {calculator_type!r}\n"
            f"Available: 1m, 5m, 5m_d3, sevennet_omni, sevennet_omni_mpa, "
            f"sevennet_omat, chgnet, mace_mp, mace_mp_d3, mace_off"
        )

    @staticmethod
    def get_calculator_info(calculator_type: str) -> dict:
        info_map = {
            "1m":                {"name": "MatterSim 1M",             "dispersion": "No"},
            "5m":                {"name": "MatterSim 5M",             "dispersion": "No"},
            "5m_d3":             {"name": "MatterSim 5M + D3",        "dispersion": "Yes (D3, post-hoc)"},
            "sevennet_omni":     {"name": "SevenNet-OMNI (omat24)",   "dispersion": "Yes (D3, native via OMat24)"},
            "sevennet_omni_mpa": {"name": "SevenNet-OMNI (mpa)",      "dispersion": "No (PBE head)"},
            "sevennet_omat":     {"name": "SevenNet-OMat",            "dispersion": "Yes (D3, native)"},
            "chgnet":            {"name": "CHGNet",                   "dispersion": "No"},
            "mace_mp":           {"name": "MACE-MP-0 (medium)",       "dispersion": "No"},
            "mace_mp_d3":        {"name": "MACE-MP-0 (medium) + D3",  "dispersion": "Yes (D3)"},
            "mace_off":          {"name": "MACE-OFF23 (medium)",      "dispersion": "No"},
        }
        return info_map.get(calculator_type.lower().replace("+", "_").replace("-", "_"), {})
