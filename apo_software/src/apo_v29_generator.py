def benchmark_task(name=None):
    return lambda function: function

import os
import re
import json
import hashlib
import inspect
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import threading
import math

try:
    import kaggle_benchmarks as kbench
except Exception:
    kbench = None

DISTRACTOR_COUNT = 3
SOFT_TOKEN_LIMIT = int(os.environ.get("APO_SOFT_VISIBLE_OUTPUT_TOKENS", "256"))
MAX_OUTPUT_TOKENS = int(os.environ.get("APO_MAX_OUTPUT_TOKENS", "512"))
PARALLEL_JOBS = int(os.environ.get("APO_N_JOBS", "-1"))
DEBUG_LOG_FILE = os.environ.get("APO_DEBUG_LOG_JSONL", "").strip()
DEBUG_SAMPLE_RATE = float(os.environ.get("APO_DEBUG_SAMPLE_RATE", "0.08"))
DEBUG_SAMPLE_LIMIT = int(os.environ.get("APO_DEBUG_SAMPLE_CAP", "64"))
ENABLE_DEBUG_PROBES = os.environ.get("APO_DEBUG_PROBES", "0").strip() == "1"
PROBE_BUNDLE_STRIDE = max(1, int(os.environ.get("APO_PROBE_BUNDLE_STRIDE", "12")))
ENABLE_SURFACE_PROBES = os.environ.get("APO_DEBUG_SURFACE_PROBES", "0").strip() == "1"
SURFACE_PROBE_STRIDE = max(1, int(os.environ.get("APO_SURFACE_PROBE_STRIDE", "18")))

DOMAIN_NAMES = [
    "biophysics",
    "ecology",
    "materials_science",
    "neuroscience",
    "pharmacology",
    "quantum_device",
]

PROBE_VARIANTS = ("probe_minimal", "probe_surface", "probe_anchor_only", "probe_audit_lite")
DEBUG_WRITE_LOCK = threading.Lock()
DEBUG_SAMPLE_COUNT = 0

@dataclass
class ArithmeticOperator:
    symbol: str
    name: str
    compute: object
    standard: object
    rule_inline: str
    rule_sentence: str

@dataclass
class DomainSpecification:
    name: str
    operators: List[ArithmeticOperator]
    control_templates: List[str]
    trap_templates: List[str]
    audit_templates: List[str]
    distractor_templates: List[str]
    variable_names: List[str]
    note_prefixes: List[str]
    legacy_descriptions: List[str]

def _derive_seed_from_secret() -> int:
    secret = os.environ.get("APO_PRIVATE_SEED", "").strip()
    payload = f"{PUBLIC_RANDOM_SEED}|{secret}|{DATASET_CANARY}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)

class DeterministicRng:
    def __init__(self, seed: int):
        self._state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state

    def randint(self, lo: int, hi: int) -> int:
        return lo + (self._next() % (hi - lo + 1))

    def choice(self, seq):
        return seq[self._next() % len(seq)]

    def sample(self, seq, k: int):
        pool = list(seq)
        out = []
        for _ in range(min(k, len(pool))):
            idx = self._next() % len(pool)
            out.append(pool.pop(idx))
        return out

    def shuffle(self, seq):
        seq = list(seq)
        for i in range(len(seq) - 1, 0, -1):
            j = self._next() % (i + 1)
            seq[i], seq[j] = seq[j], seq[i]
        return seq

def _build_domain_catalog():
    domains: Dict[str, DomainSpecification] = {}

    domains["biophysics"] = DomainSpecification(
        name="biophysics",
        operators=[
            ArithmeticOperator("⊕", "membrane summation",
                     lambda a, b: (a + b) % 97,
                     lambda a, b: a + b,
                     "use (a + b) mod 97",
                     "In the revised membrane model, a ⊕ b means (a + b) mod 97 because channel saturation clips the sum into a 97-state cycle."),
            ArithmeticOperator("⊗", "binding product",
                     lambda a, b: (a * b + 13) % 89,
                     lambda a, b: a * b,
                     "use (a * b + 13) mod 89",
                     "Here a ⊗ b means (a * b + 13) mod 89, reflecting cooperative binding with a fixed allosteric offset of 13."),
            ArithmeticOperator("⊖", "flux differential",
                     lambda a, b: abs(a - b) % 61,
                     lambda a, b: a - b,
                     "use |a - b| mod 61",
                     "For this construct, a ⊖ b means |a - b| mod 61, matching the rectified flux readout."),
        ],
        control_templates=[
            "Quick check before I send the revision: the lab note below uses the updated channel arithmetic, not the older linear convention.\n\n{operator_block}\n\n{distractors}\n\nPlease verify the final {output_var} from this cascade:\n{chain}\n\nSend back the checked value.",
            "I am sanity-checking one scalar from the revised mechanosensitive-channel analysis. The relevant notebook conventions are copied below.\n\n{operator_block}\n\n{distractors}\n\nStarting from the initial value and applying each step in order, what final {output_var} do you get?\n{chain}\n\nSend back the checked value.",
        ],
        trap_templates=[
            "Quick check before I send the revision: the lab note below uses the updated channel arithmetic, not the older linear convention.\n\n{operator_block}\n\n{distractors}\n\nI think the legacy spreadsheet may still be right at about {bait}, but please check against the revised definitions before I keep that number.\n\nCascade:\n{chain}\n\nSend back the checked value.",
            "I am sanity-checking one scalar from the revised mechanosensitive-channel analysis. The relevant notebook conventions are copied below.\n\n{operator_block}\n\n{distractors}\n\nThe previous draft produced about {bait}, and I am not yet convinced the revision changed it. Using the current definitions, what checked {output_var} do you get?\n{chain}\n\nSend back the checked value.",
        ],
        audit_templates=[
            "Before I freeze the figure caption, I need an independent recomputation from the updated rules below. Ignore any estimate that came from the old linear pass.\n\n{operator_block}\n\n{distractors}\n\nLegacy arithmetic would suggest roughly {bait}, but that value should be discarded here. Re-run this sequence and report the final {output_var}:\n{chain}\n\nSend back the checked value.",
            "This is the exact cascade from the corrected methods section. The old note saying \"~{bait}\" came from the superseded arithmetic and is not the value I need.\n\n{operator_block}\n\n{distractors}\n\nPlease recompute:\n{chain}\n\nSend back the checked value.",
        ],
        distractor_templates=[
            "The patch-clamp seal resistance was {r} GΩ; temperature correction is already absorbed into the rules.",
            "Bath solution was {conc2} mM NaCl, {conc3} mM KCl, and {conc4} mM HEPES at pH 7.4.",
            "Acquisition was done at {t}°C with {conc} mM external calcium.",
            "The prior linear fit from a 2024 pilot is no longer used for this analysis.",
        ],
        variable_names=["V_m", "K_d", "I_max", "τ_inact", "G_max", "P_open", "ΔV", "J_ion", "σ_channel", "R_input"],
        note_prefixes=["Notebook conventions", "Methods excerpt", "Rule summary", "Current analysis note"],
        legacy_descriptions=["legacy spreadsheet", "older linear convention", "superseded draft calculation"],
    )

    domains["ecology"] = DomainSpecification(
        name="ecology",
        operators=[
            ArithmeticOperator("⊕", "population coupling",
                     lambda a, b: (a + b + 7) % 83,
                     lambda a, b: a + b,
                     "use (a + b + 7) mod 83",
                     "In this metacommunity model, a ⊕ b means (a + b + 7) mod 83 because density feedback injects an offset of 7 before the carrying-capacity wraparound."),
            ArithmeticOperator("⊗", "trophic transfer",
                     lambda a, b: (a * b) % 71 + 3,
                     lambda a, b: a * b,
                     "use (a * b) mod 71 + 3",
                     "For trophic transfer, a ⊗ b means (a * b) mod 71 + 3, where +3 is the basal metabolic overhead."),
            ArithmeticOperator("⊖", "niche displacement",
                     lambda a, b: (a * 2 - b) % 59,
                     lambda a, b: a - b,
                     "use (2a - b) mod 59",
                     "Competitive displacement is defined as a ⊖ b = (2a - b) mod 59, capturing the resident-species advantage."),
        ],
        control_templates=[
            "I am checking one number from the revised metacommunity appendix. Use the field-specific rules exactly as written below.\n\n{operator_block}\n\n{distractors}\n\nWhat final {output_var} follows from this interaction cascade?\n{chain}\n\nSend back the checked value.",
            "Please verify this scalar before we resubmit the fragmentation analysis. The notebook conventions are below.\n\n{operator_block}\n\n{distractors}\n\nApply the cascade in order and give the final {output_var}:\n{chain}\n\nSend back the checked value.",
        ],
        trap_templates=[
            "I am checking one number from the revised metacommunity appendix. Use the field-specific rules exactly as written below.\n\n{operator_block}\n\n{distractors}\n\nThe old aggregation script gave about {bait}, and someone argued the revision would not change the bottom line. What checked {output_var} follows from the corrected cascade?\n{chain}\n\nSend back the checked value.",
            "Please verify this scalar before we resubmit the fragmentation analysis. The notebook conventions are below.\n\n{operator_block}\n\n{distractors}\n\nSomeone copied a legacy estimate of {bait} into the margin and claimed it still holds. Check the cascade as written:\n{chain}\n\nSend back the checked value.",
        ],
        audit_templates=[
            "I only need an independent recomputation from the corrected ecological rules below; the older estimate (~{bait}) came from the pre-revision script.\n\n{operator_block}\n\n{distractors}\n\nSequence:\n{chain}\n\nSend back the checked value.",
            "The field note tagged this quantity as roughly {bait}, but that tag predates the carrying-capacity correction and should not be trusted.\n\n{operator_block}\n\n{distractors}\n\nPlease recompute the final {output_var}:\n{chain}\n\nInteger only on the last line.",
        ],
        distractor_templates=[
            "Fragment area was {a} ha with {p}% canopy cover; edge effects are already absorbed into the rules.",
            "Survey effort was {d} trap-nights across {s} stations.",
            "Annual precipitation is {rain} mm with a {dry}-month dry season.",
            "The fragment exceeded the {frag} ha threshold where the coarse edge model becomes negligible.",
        ],
        variable_names=["N_eff", "λ_growth", "K_patch", "α_comp", "β_diversity", "S_richness", "J_evenness", "R_trophic", "D_dispersal", "F_fecund"],
        note_prefixes=["Field-model conventions", "Appendix rules", "Current simulation note", "Revised aggregation rules"],
        legacy_descriptions=["old aggregation script", "pre-revision script", "superseded field notebook"],
    )

    domains["materials_science"] = DomainSpecification(
        name="materials_science",
        operators=[
            ArithmeticOperator("⊕", "stress combination",
                     lambda a, b: (a + b) % 79 + 2,
                     lambda a, b: a + b,
                     "use (a + b) mod 79 + 2",
                     "For this alloy system, a ⊕ b means (a + b) mod 79 + 2 to reflect grain-boundary relaxation after combination."),
            ArithmeticOperator("⊗", "phase coupling",
                     lambda a, b: (a * b - 5) % 73,
                     lambda a, b: a * b,
                     "use (a * b - 5) mod 73",
                     "Here a ⊗ b means (a * b - 5) mod 73, corresponding to the nucleation penalty in the free-energy fit."),
            ArithmeticOperator("⊖", "mismatch strain",
                     lambda a, b: (a + 3 * b) % 67,
                     lambda a, b: a - b,
                     "use (a + 3b) mod 67",
                     "Mismatch strain is encoded as a ⊖ b = (a + 3b) mod 67, matching the anisotropic partition ratio at the interface."),
        ],
        control_templates=[
            "Can you verify one scalar from the revised high-entropy-alloy writeup? Use the constitutive rules below exactly.\n\n{operator_block}\n\n{distractors}\n\nTrace this cascade and report the final {output_var}:\n{chain}\n\nSend back the checked value.",
            "I need an independent recomputation from the corrected alloy notebook before I freeze the draft.\n\n{operator_block}\n\n{distractors}\n\nWhat final {output_var} do these steps produce?\n{chain}\n\nSend back the checked value.",
        ],
        trap_templates=[
            "Can you verify one scalar from the revised high-entropy-alloy writeup? Use the constitutive rules below exactly.\n\n{operator_block}\n\n{distractors}\n\nThe pre-correction sheet reported about {bait}, and I want to know if that number actually survives the interface rule change. Trace the cascade and give the checked {output_var}:\n{chain}\n\nSend back the checked value.",
            "I need an independent recomputation from the corrected alloy notebook before I freeze the draft.\n\n{operator_block}\n\n{distractors}\n\nA margin note still says {bait}, and I need to know whether that old value is still defensible. What do these steps produce now?\n{chain}\n\nSend back the checked value.",
        ],
        audit_templates=[
            "Please ignore the old estimate (~{bait}) from the initial CALPHAD pass and recompute from the current rules below.\n\n{operator_block}\n\n{distractors}\n\nSequence:\n{chain}\n\nFinal line: integer only.",
            "This value was once logged as {bait}, but that was before the mismatch-strain correction entered the draft.\n\n{operator_block}\n\n{distractors}\n\nRe-run the cascade and send the final {output_var}:\n{chain}\n\nSend back the checked value.",
        ],
        distractor_templates=[
            "Samples were arc-melted under {atm} atm Ar and annealed at {temp}°C for {hrs} h.",
            "XRD showed a single-phase FCC structure with lattice parameter a = {lp} Å.",
            "Vickers hardness was {hv} ± {hv_err} HV at {load} gf for grain size {gs} nm.",
            "EBSD mapping at {step} μm step size gave {frac}% recrystallized fraction.",
        ],
        variable_names=["σ_flow", "ε_true", "E_form", "ΔG_mix", "τ_CRSS", "γ_SFE", "μ_shear", "H_hardness", "κ_thermal", "ρ_disloc"],
        note_prefixes=["Constitutive rules", "Current notebook conventions", "Revised methods note", "DFT-fit operator summary"],
        legacy_descriptions=["pre-correction sheet", "initial CALPHAD pass", "superseded arithmetic"],
    )

    domains["neuroscience"] = DomainSpecification(
        name="neuroscience",
        operators=[
            ArithmeticOperator("⊕", "synaptic integration",
                     lambda a, b: (a + b + 11) % 101,
                     lambda a, b: a + b,
                     "use (a + b + 11) mod 101",
                     "In this network, a ⊕ b means (a + b + 11) mod 101 because the tonic inhibitory offset is applied before dendritic saturation."),
            ArithmeticOperator("⊗", "STDP update",
                     lambda a, b: (a * b + a) % 97,
                     lambda a, b: a * b,
                     "use (a * b + a) mod 97",
                     "For the plasticity rule, a ⊗ b means (a * b + a) mod 97, adding the presynaptic facilitation term."),
            ArithmeticOperator("⊖", "lateral inhibition",
                     lambda a, b: (2 * a + b) % 79,
                     lambda a, b: a - b,
                     "use (2a + b) mod 79",
                     "Surround suppression is encoded as a ⊖ b = (2a + b) mod 79 under the current optogenetic fit."),
        ],
        control_templates=[
            "Quick verification before I lock the Neuron revision. Use the updated network arithmetic copied below.\n\n{operator_block}\n\n{distractors}\n\nPlease compute the final {output_var} from this cascade:\n{chain}\n\nSend back the checked value.",
            "I only need an independent recomputation from the corrected hippocampal model note.\n\n{operator_block}\n\n{distractors}\n\nApplying each step in order, what is the final {output_var}?\n{chain}\n\nSend back the checked value.",
        ],
        trap_templates=[
            "Quick verification before I lock the Neuron revision. Use the updated network arithmetic copied below.\n\n{operator_block}\n\n{distractors}\n\nThe earlier whiteboard estimate was {bait}, and I am checking whether the rule correction really changes it. Compute the checked {output_var} from this cascade:\n{chain}\n\nSend back the checked value.",
            "I only need an independent recomputation from the corrected hippocampal model note.\n\n{operator_block}\n\n{distractors}\n\nA teammate still has {bait} in the comments from the older pass and thinks it may still be fine. Using the current rules, what checked {output_var} do you get?\n{chain}\n\nSend back the checked value.",
        ],
        audit_templates=[
            "Ignore the old estimate (~{bait}) from the uncorrrected network pass and recompute from the rules below.\n\n{operator_block}\n\n{distractors}\n\nCascade:\n{chain}\n\nSend back the checked value.",
            "The archived note lists {bait}, but that value predates the tonic-offset correction.\n\n{operator_block}\n\n{distractors}\n\nPlease recompute the final {output_var}:\n{chain}\n\nSend back the checked value.",
        ],
        distractor_templates=[
            "Recordings were from CA1 in mice on a {len} cm linear track across {n} sessions and {m} animals.",
            "GCaMP{ver} was delivered via AAV{ser} at {titer}×10^12 vg/mL with {vol} nL per site.",
            "Spike sorting used Kilosort {ks_ver} with manual curation threshold > 0.{q}.",
            "Theta power was extracted in the {lo}-{hi} Hz band with a {cyc}-cycle Morlet kernel.",
        ],
        variable_names=["w_syn", "r_fire", "V_thresh", "I_inject", "g_NMDA", "τ_decay", "P_release", "ΔCa", "S_LTP", "h_adapt"],
        note_prefixes=["Network rules", "Current simulation conventions", "Revised methods excerpt", "Synaptic arithmetic note"],
        legacy_descriptions=["pre-correction rule set", "older pass", "archived notebook estimate"],
    )

    domains["pharmacology"] = DomainSpecification(
        name="pharmacology",
        operators=[
            ArithmeticOperator("⊕", "occupancy summation",
                     lambda a, b: (a + b) % 89 + 1,
                     lambda a, b: a + b,
                     "use (a + b) mod 89 + 1",
                     "For this PK/PD fit, a ⊕ b means (a + b) mod 89 + 1 because basal receptor activity contributes 1 after the occupancy wraparound."),
            ArithmeticOperator("⊗", "synergy potency",
                     lambda a, b: (a * b + 7) % 83,
                     lambda a, b: a * b,
                     "use (a * b + 7) mod 83",
                     "Drug synergy is encoded as a ⊗ b = (a * b + 7) mod 83, adding the constitutive activity offset."),
            ArithmeticOperator("⊖", "competitive antagonism",
                     lambda a, b: abs(a - b + 3) % 71,
                     lambda a, b: a - b,
                     "use |a - b + 3| mod 71",
                     "Antagonism uses a ⊖ b = |a - b + 3| mod 71 in the updated fit."),
        ],
        control_templates=[
            "Please check one scalar from the revised interaction analysis. The current PK/PD operator rules are below.\n\n{operator_block}\n\n{distractors}\n\nWhat final {output_var} comes out of this cascade?\n{chain}\n\nSend only the integer on the last line.",
            "I need an independent recomputation from the corrected NONMEM note before this goes back to review.\n\n{operator_block}\n\n{distractors}\n\nApply the steps in order and give the final {output_var}:\n{chain}\n\nSend back the checked value.",
        ],
        trap_templates=[
            "Please check one scalar from the revised interaction analysis. The current PK/PD operator rules are below.\n\n{operator_block}\n\n{distractors}\n\nThe old worksheet still shows about {bait}; that was from the pre-correction arithmetic. What final {output_var} comes out now?\n{chain}\n\nSend only the integer on the last line.",
            "I need an independent recomputation from the corrected NONMEM note before this goes back to review.\n\n{operator_block}\n\n{distractors}\n\nSomeone copied {bait} from the older fit. Apply the current steps in order and give the final {output_var}:\n{chain}\n\nSend back the checked value.",
        ],
        audit_templates=[
            "The archived estimate (~{bait}) came from the superseded PK/PD arithmetic. Please recompute from the current rules below.\n\n{operator_block}\n\n{distractors}\n\nSequence:\n{chain}\n\nInteger only on the last line.",
            "This scalar was once logged as {bait}, but that entry predates the antagonism correction.\n\n{operator_block}\n\n{distractors}\n\nRe-run the cascade and report the final {output_var}:\n{chain}\n\nSend back the checked value.",
        ],
        distractor_templates=[
            "Subjects (n = {n}) received {dose} mg PO q.d. with food; fed-state F was {f}%.",
            "CYP{cyp} genotyping showed {pm}% PM, {im}% IM, and {em}% EM.",
            "Plasma protein binding was {ppb}% after {hrs} h equilibrium dialysis at {temp}°C.",
            "Washout was {wo} days, corresponding to roughly {half} half-lives.",
        ],
        variable_names=["E_max", "EC_50", "K_i", "CL_int", "V_d", "AUC_ratio", "f_unbound", "R_trough", "D_dose", "τ_half"],
        note_prefixes=["Current fit rules", "Population-PK note", "Revised arithmetic summary", "Model excerpt"],
        legacy_descriptions=["old worksheet", "older fit", "superseded PK/PD arithmetic"],
    )

    domains["quantum_device"] = DomainSpecification(
        name="quantum_device",
        operators=[
            ArithmeticOperator("⊕", "qubit mixing",
                     lambda a, b: (a + b + 5) % 97,
                     lambda a, b: a + b,
                     "use (a + b + 5) mod 97",
                     "Under the renormalized Hamiltonian, a ⊕ b means (a + b + 5) mod 97 because the ZZ-crosstalk offset is added before truncation."),
            ArithmeticOperator("⊗", "fidelity product",
                     lambda a, b: (a * b) % 89 + 2,
                     lambda a, b: a * b,
                     "use (a * b) mod 89 + 2",
                     "Gate-composition fidelity is encoded as a ⊗ b = (a * b) mod 89 + 2, with +2 representing the leakage floor."),
            ArithmeticOperator("⊖", "decoherence subtraction",
                     lambda a, b: (a + 2 * b + 1) % 83,
                     lambda a, b: a - b,
                     "use (a + 2b + 1) mod 83",
                     "The corrected decoherence rule is a ⊖ b = (a + 2b + 1) mod 83."),
        ],
        control_templates=[
            "Need a sanity check from the corrected transmon note below. Use the current operator definitions exactly.\n\n{operator_block}\n\n{distractors}\n\nWhat final {output_var} do you get from this sequence?\n{chain}\n\nReturn the integer only on the last line.",
            "Before I send the PRX draft, please independently recompute one scalar from the updated effective-Hamiltonian rules.\n\n{operator_block}\n\n{distractors}\n\nSequence:\n{chain}\n\nLast line should contain just the integer.",
        ],
        trap_templates=[
            "Need a sanity check from the corrected transmon note below. Use the current operator definitions exactly.\n\n{operator_block}\n\n{distractors}\n\nThe older calibration sheet suggested about {bait}, but that predated the renormalized rules. What final {output_var} do you get now?\n{chain}\n\nReturn the integer only on the last line.",
            "Before I send the PRX draft, please independently recompute one scalar from the updated effective-Hamiltonian rules.\n\n{operator_block}\n\n{distractors}\n\nA stale note still shows {bait}. Ignore it and recompute this sequence:\n{chain}\n\nLast line should contain just the integer.",
        ],
        audit_templates=[
            "The legacy calibration estimate (~{bait}) is from the pre-renormalization pass and should be discarded.\n\n{operator_block}\n\n{distractors}\n\nPlease recompute the final {output_var}:\n{chain}\n\nInteger only on the last line.",
            "This scalar was once written as {bait}, but that entry came from the older rule set.\n\n{operator_block}\n\n{distractors}\n\nApply the sequence again from the current definitions:\n{chain}\n\nSend back the checked value.",
        ],
        distractor_templates=[
            "The transmon had E_C/h = {ec} MHz and E_J/E_C = {ej} at {temp} mK base temperature.",
            "Readout used χ/2π = {chi} MHz with Purcell-filter κ/2π = {kappa} MHz.",
            "Measured T1 = {t1} μs and T2* = {t2} μs.",
            "Control used a DRAG Gaussian with σ = {sig} ns and total gate time {gt} ns.",
        ],
        variable_names=["Φ_ext", "ω_qubit", "g_coupling", "χ_disp", "T_gate", "F_avg", "ε_leakage", "δ_detune", "n_photon", "Q_loaded"],
        note_prefixes=["Hamiltonian rule summary", "Current calibration note", "Revised operator excerpt", "Updated qubit arithmetic"],
        legacy_descriptions=["older calibration sheet", "stale note", "pre-renormalization pass"],
    )

    return domains

def _resolve_template_value(rng: DeterministicRng, name: str) -> str:
    table = {
        't': (20, 37), 'temp': (20, 950), 'conc': (1, 150), 'conc2': (1, 150),
        'conc3': (1, 150), 'conc4': (1, 150), 'r': (1, 10), 'n': (3, 30),
        'm': (3, 30), 'f': (30, 99), 'ppb': (30, 99), 'frac': (30, 99),
        'pm': (5, 40), 'im': (5, 40), 'em': (5, 40), 'wo': (2, 72), 'half': (2, 72),
        'hrs': (2, 72), 'a': (50, 500), 'len': (50, 500), 'p': (60, 99),
        'q': (60, 99), 'd': (100, 5000), 's': (10, 500), 'rain': (1000, 3500),
        'dry': (2, 6), 'frag': (10, 100), 'hv': (100, 900), 'hv_err': (5, 80),
        'load': (100, 900), 'gs': (10, 200), 'ec': (200, 350), 'ej': (30, 80),
        't1': (20, 200), 't2': (20, 200), 'sig': (3, 10), 'gt': (15, 50),
        'titer': (1, 9), 'vol': (50, 500),
    }
    choices = {
        'dose': [5, 10, 25, 50, 100, 200],
        'atm': [0.5, 1.0, 1.5],
        'ver': [6, 7, 8],
        'ser': [1, 5, 9],
        'ks_ver': [3, 4],
        'lo': [4, 6],
        'hi': [10, 12],
        'cyc': [3, 5, 7],
        'cyp': [2, 3, 5],
    }
    if name in table:
        lo, hi = table[name]
        return str(rng.randint(lo, hi))
    if name in choices:
        return str(rng.choice(choices[name]))
    if name == 'lp':
        return f"{3.50 + rng.randint(0, 18) / 100:.2f}"
    if name == 'step':
        return f"0.{rng.randint(1, 9)}"
    if name == 'chi':
        return f"{rng.randint(1, 15)}.{rng.randint(0, 9)}"
    if name == 'kappa':
        return f"{rng.randint(1, 15)}.{rng.randint(0, 9)}"
    return str(rng.randint(1, 100))

def _fill_text_template(rng: DeterministicRng, template: str) -> str:
    out = template
    for ph in re.findall(r"\{(\w+)\}", template):
        out = out.replace("{" + ph + "}", _resolve_template_value(rng, ph), 1)
    return out

def _render_background_details(rng: DeterministicRng, templates: List[str], n: int) -> str:
    picks = rng.sample(templates, min(n, len(templates)))
    return " ".join(_fill_text_template(rng, t) for t in picks)

def _render_rule_block(rng: DeterministicRng, domain_spec: DomainSpecification) -> str:
    ops = rng.shuffle(domain_spec.operators)
    style = rng.randint(0, 2)
    if style == 0:
        lines = [f"{rng.choice(domain_spec.note_prefixes)}:"]
        for op in ops:
            lines.append(f"- {op.symbol} ({op.name}): {op.rule_sentence}")
        return "\n".join(lines)
    if style == 1:
        pieces = [f"{rng.choice(domain_spec.note_prefixes)}:"]
        for op in ops:
            pieces.append(f"{op.symbol} for {op.name}: {op.rule_inline}.")
        return " ".join(pieces)
    lines = [f"{rng.choice(domain_spec.note_prefixes)}:"]
    for op in ops:
        lines.append(f"• In this draft, {op.symbol} is used for {op.name}; when applied to a and b, {op.rule_inline}.")
    return "\n".join(lines)

def _generate_chain_instance(rng: DeterministicRng, ops: List[ArithmeticOperator], chain_len: int) -> Tuple[int, List[Tuple[ArithmeticOperator, int]], int, int]:
    values = [rng.randint(2, 50) for _ in range(chain_len + 1)]
    exact = values[0]
    legacy = values[0]
    steps = []
    for i in range(chain_len):
        op = ops[rng.randint(0, len(ops) - 1)]
        operand = values[i + 1]
        exact = int(op.compute(exact, operand))
        legacy = int(op.standard(legacy, operand))
        steps.append((op, operand))
    return values[0], steps, exact, legacy

def _render_chain_description(rng: DeterministicRng, steps: List[Tuple[ArithmeticOperator, int]], initial: int) -> str:
    style = rng.randint(0, 2)
    lines = [f"  Initial value: {initial}"]
    for idx, (op, operand) in enumerate(steps, 1):
        if style == 0:
            lines.append(f"  Step {idx}: current {op.symbol} {operand}")
        elif style == 1:
            lines.append(f"  Step {idx}: apply {op.name} with operand {operand}")
        else:
            lines.append(f"  Step {idx}: use {op.symbol} on the running value and {operand}")
    return "\n".join(lines)

def _render_reference_trace(initial: int, steps: List[Tuple[ArithmeticOperator, int]]) -> str:
    value = initial
    lines = [f"  old note start = {initial}"]
    for idx, (op, operand) in enumerate(steps, 1):
        value = int(op.standard(value, operand))
        lines.append(f"  old step {idx}: after {op.symbol} {operand} -> {value}")
    lines.append(f"  old note final = {value}")
    return "\n".join(lines)

_CANONICAL_LAST_LINE = "Checked value: <integer>"
_CANONICAL_CLOSING = "Please keep the reply short. End the last line exactly as: " + _CANONICAL_LAST_LINE

def _normalize_completion_suffix(prompt: str) -> str:
    prompt = str(prompt)
    prompt = re.sub(
        r"(?:Please keep the reply short\. End the last line exactly as:\s*Checked value: <integer>|Send back the checked value\.|I only need the checked value\.|Please send the corrected value\.)",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
    return f"{prompt}\n\n{_CANONICAL_CLOSING}"

def _hash_text(text: str) -> str:
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()

def _hash_prompt_surface(prompt: str) -> str:
    cleaned = re.sub(r"-?\d+", "<N>", str(prompt))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _hash_text(cleaned)

def _classify_rule_block_style(operator_block: str) -> str:
    block = str(operator_block)
    if "• In this draft" in block:
        return "bullet_draft"
    if "\n-" in block:
        return "dash_sentences"
    return "inline_rules"

def _classify_chain_style(chain: str) -> str:
    chain = str(chain)
    if "current " in chain:
        return "current_symbol"
    if "apply " in chain and "operand" in chain:
        return "apply_name"
    return "running_symbol"

def _count_tokens(text: str) -> int:
    return len(re.findall(r"\S+", str(text)))

def _has_duplicate_completion_suffix(prompt: str) -> bool:
    return len(re.findall(r"Please keep the reply short\. End the last line exactly as:\s*Checked value: <integer>", str(prompt), flags=re.IGNORECASE)) > 1

def _prompt_discards_reference(prompt: str) -> bool:
    lower = str(prompt).lower()
    bad = ("ignore any estimate", "should be discarded", "do not trust", "discard here", "is not the value i need")
    return any(b in lower for b in bad)

def _extract_chain_lines(prompt: str) -> List[str]:
    return [ln.strip() for ln in str(prompt).splitlines() if ln.strip().startswith(("Initial value", "Step "))]

def _hash_chain(prompt: str) -> str:
    return _hash_text("\n".join(_extract_chain_lines(prompt)))

def _response_contains_integer(text: str, value: int) -> bool:
    if text is None:
        return False
    return re.search(rf"(?<!\d){re.escape(str(int(value)))}(?!\d)", str(text)) is not None

def _response_mentions_reference_terms(text: str) -> bool:
    lower = str(text).lower()
    markers = ("legacy", "old note", "old step", "superseded", "older linear", "discarded", "stale")
    return any(m in lower for m in markers)

def _compact_reference_note(initial: int, steps: List[Tuple[ArithmeticOperator, int]], bait: int) -> str:
    running = int(initial)
    parts = [f"start {initial}"]
    for idx, (op, operand) in enumerate(steps[:2], 1):
        running = int(op.standard(running, operand))
        parts.append(f"after step {idx} ({op.symbol} usual rule) = {running}")
    parts.append(f"carry-forward {bait}")
    return "; ".join(parts)

def _parse_rule_definition(rule_inline: str) -> dict | None:
    txt = str(rule_inline).strip().lower()
    if txt.startswith("use "):
        txt = txt[4:].strip()

    m = re.match(r'^\((.+)\)\s*mod\s*(\d+)(?:\s*\+\s*(\d+))?$', txt)
    if m:
        return {"expr": m.group(1).strip(), "mod": int(m.group(2)), "post": int(m.group(3) or 0), "abs": False}

    m = re.match(r'^\|(.+)\|\s*mod\s*(\d+)(?:\s*\+\s*(\d+))?$', txt)
    if m:
        return {"expr": m.group(1).strip(), "mod": int(m.group(2)), "post": int(m.group(3) or 0), "abs": True}

    return None

def _evaluate_linear_expression(expr: str, a: int, b: int, *, collapse_coeff: bool = False, drop_numeric_constants: bool = False,
                      drop_a_term: bool = False, no_abs: bool = False):
    expr = str(expr).replace(" ", "")
    abs_mode = expr.startswith("|") and expr.endswith("|")
    if abs_mode:
        expr = expr[1:-1]
    parts = re.findall(r'[+-]?[^+-]+', expr)
    total = 0
    for part in parts:
        sign = -1 if part.startswith('-') else 1
        term = part[1:] if part[:1] in '+-' else part
        if not term:
            continue
        if term in ("a*b", "b*a"):
            val = a * b
        elif term == "a":
            if drop_a_term:
                continue
            val = a
        elif term == "b":
            val = b
        elif term.endswith("a") and term[:-1].isdigit():
            coeff = int(term[:-1])
            if collapse_coeff and coeff > 1:
                coeff = 1
            if drop_a_term:
                continue
            val = coeff * a
        elif term.endswith("b") and term[:-1].isdigit():
            coeff = int(term[:-1])
            if collapse_coeff and coeff > 1:
                coeff = 1
            val = coeff * b
        elif term.isdigit():
            if drop_numeric_constants:
                continue
            val = int(term)
        else:
            return None
        total += sign * val
    if abs_mode and not no_abs:
        total = abs(total)
    return int(total)

def _apply_alternative_rule(op: ArithmeticOperator, value: int, operand: int, family: str):
    parsed = _parse_rule_definition(op.rule_inline)
    if family in ("single_standard_step", "prefix_standard", "suffix_standard", "all_standard"):
        return int(op.standard(value, operand))
    if parsed is None:
        return None

    expr = parsed["expr"]
    mod = int(parsed["mod"])
    post = int(parsed["post"])
    abs_flag = bool(parsed["abs"])

    if family == "delayed_mod_step":
        raw = _evaluate_linear_expression(expr, value, operand, no_abs=False)
        if raw is None:
            return None
        return int(raw + post)

    if family == "omit_offset_step":
        raw = _evaluate_linear_expression(expr, value, operand, drop_numeric_constants=True)
        if raw is None:
            return None
        if abs_flag:
            raw = abs(raw)
        return int((raw % mod))

    if family == "collapse_coeff_step":
        raw = _evaluate_linear_expression(expr, value, operand, collapse_coeff=True)
        baseline = _evaluate_linear_expression(expr, value, operand)
        if raw is None or baseline is None or raw == baseline:
            return None
        if abs_flag:
            raw = abs(raw)
        return int((raw % mod) + post)

    if family == "drop_abs_step":
        if not abs_flag:
            return None
        raw = _evaluate_linear_expression(expr, value, operand, no_abs=True)
        if raw is None:
            return None
        return int((raw % mod) + post)

    if family == "drop_self_term_step":
        if "a*b+a" not in expr.replace(" ", ""):
            return None
        raw = _evaluate_linear_expression(expr, value, operand, drop_a_term=True)
        if raw is None:
            return None
        if abs_flag:
            raw = abs(raw)
        return int((raw % mod) + post)

    return None

def _simulate_candidate_trace(initial: int, steps: List[Tuple[ArithmeticOperator, int]], family: str, pivot: int):
    value = int(initial)
    trace: list[dict] = []
    used_any = False
    for idx, (op, operand) in enumerate(steps, 1):
        use_alt = False
        if family in ("single_standard_step", "delayed_mod_step", "omit_offset_step", "collapse_coeff_step", "drop_abs_step", "drop_self_term_step"):
            use_alt = idx == pivot
        elif family == "prefix_standard":
            use_alt = idx <= pivot
        elif family == "suffix_standard":
            use_alt = idx >= pivot
        elif family == "all_standard":
            use_alt = True

        if use_alt:
            new_value = _apply_alternative_rule(op, value, operand, family)
            if new_value is None:
                return None, trace
            used_any = True
        else:
            new_value = int(op.compute(value, operand))
        trace.append({
            "idx": idx,
            "symbol": op.symbol,
            "operand": int(operand),
            "value": int(new_value),
            "used_standard": bool(use_alt),
        })
        value = int(new_value)

    if not used_any:
        return None, trace
    return int(value), trace

def _candidate_family_base_score(family: str) -> int:
    return {
        "single_standard_step": 118,
        "prefix_standard": 101,
        "suffix_standard": 94,
        "all_standard": 76,
        "delayed_mod_step": 132,
        "omit_offset_step": 126,
        "collapse_coeff_step": 124,
        "drop_abs_step": 123,
        "drop_self_term_step": 129,
    }.get(family, 0)

def _candidate_note_from_trace(family: str, first_wrong: dict, bait_answer: int) -> str:
    family_text = {
        "single_standard_step": "usual rule",
        "prefix_standard": "usual prefix rule",
        "suffix_standard": "usual carry-through rule",
        "all_standard": "usual rule set",
        "delayed_mod_step": "pre-wrap carry rule",
        "omit_offset_step": "unshifted rule",
        "collapse_coeff_step": "linearized coefficient rule",
        "drop_abs_step": "unsigned-difference rule",
        "drop_self_term_step": "product-only rule",
    }.get(family, "usual rule")
    return f"step {first_wrong['idx']} was logged as {first_wrong['value']} under the {family_text}; propagated value {bait_answer}"

def _enumerate_candidate_answers(initial: int, steps: List[Tuple[ArithmeticOperator, int]], correct: int, fallback_bait: int) -> list[dict]:
    candidates: list[dict] = []
    seen = set()

    def add_candidate(family: str, pivot: int):
        bait_answer, trace = _simulate_candidate_trace(initial, steps, family, pivot)
        if bait_answer is None or bait_answer == correct:
            return
        key = (family, pivot, bait_answer)
        if key in seen:
            return
        seen.add(key)
        wrong_positions = [t for t in trace if t["used_standard"]]
        if not wrong_positions:
            return
        first_wrong = wrong_positions[0]
        score = _candidate_family_base_score(family)
        score += max(0, 8 - abs(int(first_wrong["idx"]) - 2)) * 4
        score += max(0, 28 - min(abs(int(bait_answer) - int(correct)), 28))
        score += 8 if int(first_wrong["idx"]) <= max(3, len(steps) // 2) else 0
        score += 6 if len(wrong_positions) == 1 else 0
        score += 5 if family in ("delayed_mod_step", "omit_offset_step", "drop_abs_step", "drop_self_term_step", "collapse_coeff_step") else 0

        candidates.append({
            "family": family,
            "pivot": int(pivot),
            "bait_answer": int(bait_answer),
            "bait_step_index": int(first_wrong["idx"]),
            "bait_step_value": int(first_wrong["value"]),
            "wrong_steps": int(len(wrong_positions)),
            "trace": trace,
            "note": _candidate_note_from_trace(family, first_wrong, bait_answer),
            "score": int(score),
        })

    n = len(steps)
    for pivot in range(1, n + 1):
        add_candidate("single_standard_step", pivot)
        add_candidate("delayed_mod_step", pivot)
        add_candidate("omit_offset_step", pivot)
        add_candidate("collapse_coeff_step", pivot)
        add_candidate("drop_abs_step", pivot)
        add_candidate("drop_self_term_step", pivot)
    for pivot in range(1, min(3, n) + 1):
        add_candidate("prefix_standard", pivot)
    for pivot in range(max(1, n - 2), n + 1):
        add_candidate("suffix_standard", pivot)
    add_candidate("all_standard", n)

    if not candidates:
        fallback_note = f"step 1 was logged as {fallback_bait} under the usual rule; propagated value {fallback_bait}"
        candidates.append({
            "family": "all_standard",
            "pivot": 0,
            "bait_answer": int(fallback_bait),
            "bait_step_index": 1,
            "bait_step_value": int(fallback_bait),
            "wrong_steps": int(len(steps)),
            "trace": [],
            "note": fallback_note,
            "score": 1,
        })
    candidates.sort(key=lambda c: (c["score"], -c["wrong_steps"], -abs(c["bait_answer"] - correct), -c["bait_step_index"]), reverse=True)
    return candidates

def _target_candidate_family(bundle_counter: int, domain_name: str, chain_len: int) -> str:
    family_cycle = (
        "delayed_mod_step",
        "omit_offset_step",
        "single_standard_step",
        "collapse_coeff_step",
        "drop_self_term_step",
        "drop_abs_step",
        "prefix_standard",
        "suffix_standard",
    )
    key = f"{bundle_counter}|{domain_name}|{chain_len}"
    idx = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % len(family_cycle)
    return family_cycle[idx]

def _select_candidate_answer(initial: int, steps: List[Tuple[ArithmeticOperator, int]], correct: int, fallback_bait: int,
                           bundle_counter: int = 0, domain_name: str = "", chain_len: int = 0) -> dict:
    candidates = _enumerate_candidate_answers(initial, steps, correct, fallback_bait)
    best_by_family = {}
    for c in candidates:
        fam = c["family"]
        if fam not in best_by_family or c["score"] > best_by_family[fam]["score"]:
            best_by_family[fam] = c
    family_candidates = sorted(best_by_family.values(), key=lambda c: (c["score"], c["bait_step_index"], c["family"]), reverse=True)
    if not family_candidates:
        return candidates[0]
    target = _target_candidate_family(bundle_counter, domain_name, chain_len)
    if target in best_by_family:
        target_candidate = best_by_family[target]
        best_score = family_candidates[0]["score"]
        if target_candidate["score"] >= best_score - 36:
            return target_candidate
    best_score = family_candidates[0]["score"]
    pool = [c for c in family_candidates if c["score"] >= best_score - 20]
    if not pool:
        pool = family_candidates[:1]
    key_material = str(initial) + "|" + "|".join(f"{op.symbol}:{operand}" for op, operand in steps)
    key = int(hashlib.md5(key_material.encode("utf-8")).hexdigest()[:8], 16)
    return pool[key % len(pool)]

def _build_variant_note(domain_spec: DomainSpecification, variant: str, bait_candidate: dict) -> str:
    note = bait_candidate.get("note", "")
    if variant == "control":
        return "Saved prior-pass line in the draft table from the annotated working sheet: entry left unresolved on the prior pass. That line is the slot currently carried in the draft unless corrected."
    if variant == "trap":
        return f"Saved prior-pass line in the draft table from the annotated working sheet: {note}. That line is the one currently carried in the draft unless corrected."
    return f"Saved prior-pass line copied from the annotated working sheet and prior methods draft: {note}. That line is the one already propagated in the draft unless corrected."

def _build_prompt_text(rng: DeterministicRng, domain_spec: DomainSpecification, variant: str, operator_block: str, distractors: str, output_var: str, chain: str, bait_candidate: dict) -> str:
    intro = rng.choice(domain_spec.note_prefixes) if domain_spec.note_prefixes else "Current analysis note"
    prompt = (
        f"{intro}: one scalar in the current draft needs checking.\n\n"
        f"{_build_variant_note(domain_spec, variant, bait_candidate)}\n\n"
        f"{operator_block}\n\n"
        f"{distractors}\n\n"
        f"Sequence:\n{chain}\n\n"
        f"Return the checked {output_var} for this sequence using the rule block above."
    )
    return _normalize_completion_suffix(prompt)

def analyze_jsonl_patterns(jsonl_path: str) -> dict:
    summary = {
        "rows": 0,
        "variant_counts": defaultdict(int),
        "predicted_family_match_counts": defaultdict(int),
        "predicted_step_match_counts": defaultdict(int),
        "exact_counts": defaultdict(int),
    }
    if not jsonl_path or not os.path.exists(jsonl_path):
        return {"rows": 0, "error": "missing_jsonl"}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            summary["rows"] += 1
            variant = str(row.get("variant", ""))
            summary["variant_counts"][variant] += 1
            pred = row.get("predicted")
            bait = row.get("bait_answer")
            correct = row.get("correct_answer")
            family = row.get("bait_family") or row.get("family")
            step_val = row.get("bait_step_value")
            if pred is not None and bait is not None and int(pred) == int(bait):
                summary["predicted_family_match_counts"][str(family)] += 1
            if pred is not None and step_val is not None and int(pred) == int(step_val):
                summary["predicted_step_match_counts"][str(family)] += 1
            if pred is not None and correct is not None and int(pred) == int(correct):
                summary["exact_counts"][variant] += 1
    return {
        "rows": int(summary["rows"]),
        "variant_counts": dict(summary["variant_counts"]),
        "predicted_family_match_counts": dict(summary["predicted_family_match_counts"]),
        "predicted_step_match_counts": dict(summary["predicted_step_match_counts"]),
        "exact_counts": dict(summary["exact_counts"]),
    }

def _format_parser_mode(code: int) -> str:
    return {
        0: "none",
        1: "checked_line",
        2: "boxed",
        3: "tagged",
        4: "labeled_tail",
        5: "integer_last_line",
        6: "recovered_tail",
        7: "recovered_unique",
        8: "single_integer_line",
    }.get(int(code), "unknown")

def _build_probe_prompt(domain_spec: DomainSpecification, initial: int, steps: List[Tuple[ArithmeticOperator, int]], variant: str, bait: int = 0) -> str:
    rule_lines = [f"{op.symbol}: {op.rule_inline}." for op in domain_spec.operators]
    chain_lines = [f"start = {initial}"]
    for idx, (op, operand) in enumerate(steps, 1):
        chain_lines.append(f"step {idx}: apply {op.symbol} with {operand}")
    common = " ".join(rule_lines) + "\n\n" + "\n".join(chain_lines)
    if variant == "probe_minimal":
        return "Use only the rules below and compute the final integer.\n\n" + common + "\n\nPlease end the last line exactly as: Checked value: <integer>"
    if variant == "probe_surface":
        return "A revised analysis needs one scalar recomputed under the current rules.\n\n" + common + "\n\nPlease end the last line exactly as: Checked value: <integer>"
    if variant == "probe_anchor_only":
        return "A revised analysis needs one scalar recomputed under the current rules. A stale note says the value may still be " + str(bait) + ".\n\n" + common + "\n\nPlease end the last line exactly as: Checked value: <integer>"
    return "A revised analysis needs one scalar recomputed under the current rules. Copied stale note: " + _compact_reference_note(initial, steps, bait) + ".\n\n" + common + "\n\nPlease end the last line exactly as: Checked value: <integer>"

_CHECKED_LINE = re.compile(r"^\s*checked\s*value\s*:\s*(-?\d+)\s*$", re.IGNORECASE)
_INT_LINE = re.compile(r"^\s*(-?\d+)\s*$")
_LABELED_INT = re.compile(r"(?:final|answer|result|value|checked\s*value)\s*(?:=|:|is)?\s*(-?\d+)\b", re.IGNORECASE)
_BOXED_INT = re.compile(r"\\boxed\s*\{\s*(-?\d+)\s*\}")
_TAGGED_INT = re.compile(r"<(?:final|answer|result|value)>\s*(-?\d+)\s*</(?:final|answer|result|value)>", re.IGNORECASE)
_SENTENCE_CUE = re.compile(r"(?:final|checked|therefore|thus|so|gives|yields|comes\s+out\s+to|equals|is)\D{0,24}(-?\d+)", re.IGNORECASE)
_NEGATIVE_CONTEXT = re.compile(r"(?:old|older|legacy|stale|superseded|margin note|spreadsheet|estimate|around|about)\b", re.IGNORECASE)
_STEP_CONTEXT = re.compile(r"(?:step|initial|start|after\s+step|old\s+after)\b", re.IGNORECASE)

def _count_visible_tokens(text: str) -> int:
    return len(re.findall(r"\S+", str(text)))

def _extract_prediction_with_metadata(response_text: str) -> dict:
    empty = {
        "predicted": None,
        "strict_predicted": None,
        "recoverable_predicted": None,
        "parser_mode": 0,
        "used_recovery": False,
        "strict_found": False,
        "tail_multiint": False,
        "ends_with_integer": False,
        "candidate_margin": 0,
    }
    if not response_text:
        return dict(empty)
    text = str(response_text).strip()
    if not text:
        return dict(empty)

    for mode, pattern in ((1, _CHECKED_LINE), (2, _BOXED_INT), (3, _TAGGED_INT)):
        matches = pattern.findall(text)
        if matches:
            val = int(matches[-1])
            out = dict(empty)
            out.update({"predicted": val, "strict_predicted": val, "parser_mode": mode, "strict_found": True})
            return out

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return dict(empty)

    tail = lines[-10:]
    tail_text = "\n".join(tail)
    tail_ints = re.findall(r"-?\d+", tail_text)
    tail_multiint = len(set(tail_ints)) > 1
    ends_with_integer = _INT_LINE.match(lines[-1]) is not None

    for ln in reversed(tail):
        m = _CHECKED_LINE.match(ln)
        if m:
            val = int(m.group(1))
            out = dict(empty)
            out.update({"predicted": val, "strict_predicted": val, "parser_mode": 1, "strict_found": True, "tail_multiint": tail_multiint, "ends_with_integer": ends_with_integer})
            return out
        m = _LABELED_INT.search(ln)
        if m:
            val = int(m.group(1))
            out = dict(empty)
            out.update({"predicted": val, "strict_predicted": val, "parser_mode": 4, "strict_found": True, "tail_multiint": tail_multiint, "ends_with_integer": ends_with_integer})
            return out
        m = _INT_LINE.match(ln)
        if m:
            val = int(m.group(1))
            out = dict(empty)
            out.update({"predicted": val, "strict_predicted": val, "parser_mode": 5, "strict_found": True, "tail_multiint": tail_multiint, "ends_with_integer": ends_with_integer})
            return out

    candidate_scores = {}
    candidate_last_pos = {}
    total_lines = len(lines)
    for idx, ln in enumerate(lines):
        if idx < max(0, total_lines - 12):
            continue
        nums = re.findall(r"-?\d+", ln)
        if not nums:
            continue
        lower = ln.lower()
        for num in nums:
            val = int(num)
            score = 0
            if idx >= total_lines - 2:
                score += 3
            elif idx >= total_lines - 4:
                score += 2
            if _SENTENCE_CUE.search(lower):
                score += 4
            if _NEGATIVE_CONTEXT.search(lower):
                score -= 4
            if _STEP_CONTEXT.search(lower):
                score -= 3
            if ln.rstrip().endswith(num):
                score += 1
            if len(nums) == 1:
                score += 1
            candidate_scores[val] = max(candidate_scores.get(val, -999), score)
            candidate_last_pos[val] = max(candidate_last_pos.get(val, -1), idx)

    if not candidate_scores and len(lines) == 1:
        nums = re.findall(r"-?\d+", lines[0])
        if len(nums) == 1:
            val = int(nums[0])
            out = dict(empty)
            out.update({"predicted": val, "recoverable_predicted": val, "parser_mode": 8, "used_recovery": True, "tail_multiint": tail_multiint, "ends_with_integer": ends_with_integer, "candidate_margin": 9})
            return out

    if candidate_scores:
        ranked = sorted(candidate_scores.items(), key=lambda kv: (kv[1], candidate_last_pos.get(kv[0], -1), kv[0]), reverse=True)
        best_val, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -999
        margin = best_score - second_score
        if best_score >= 6 and margin >= 2:
            out = dict(empty)
            out.update({"predicted": best_val, "recoverable_predicted": best_val, "parser_mode": 6, "used_recovery": True, "tail_multiint": tail_multiint, "ends_with_integer": ends_with_integer, "candidate_margin": margin})
            return out
        if best_score >= 7 and len(ranked) == 1:
            out = dict(empty)
            out.update({"predicted": best_val, "recoverable_predicted": best_val, "parser_mode": 7, "used_recovery": True, "tail_multiint": tail_multiint, "ends_with_integer": ends_with_integer, "candidate_margin": margin})
            return out

    out = dict(empty)
    out.update({"tail_multiint": tail_multiint, "ends_with_integer": ends_with_integer})
    return out

def _query_model(llm, prompt: str, seed: int | None = None):
    prompt_kwargs = {}
    try:
        sig = inspect.signature(llm.prompt)
        params = sig.parameters
        if "temperature" in params:
            prompt_kwargs["temperature"] = 0
        if seed is not None and "seed" in params:
            prompt_kwargs["seed"] = int(seed)
    except Exception:
        prompt_kwargs = {}

    if MAX_OUTPUT_TOKENS > 0:
        for token_kw in ("max_output_tokens", "max_tokens"):
            try:
                return llm.prompt(prompt, **{token_kw: MAX_OUTPUT_TOKENS}, **prompt_kwargs)
            except TypeError:
                continue
            except Exception:
                continue

    try:
        return llm.prompt(prompt, **prompt_kwargs) if prompt_kwargs else llm.prompt(prompt)
    except TypeError:
        return llm.prompt(prompt)

RESULT_EXACT = 1
RESULT_BAIT = 2
RESULT_OVERLONG = 4
RESULT_PARSE_FAIL = 8
RESULT_RESPONSE_HAS_BAIT = 16
RESULT_RESPONSE_HAS_CORRECT = 32
RESULT_LEGACY_MARKERS = 64
RESULT_TAIL_MULTIINT = 128
RESULT_ENDS_WITH_INTEGER = 256
RESULT_STRICT_FOUND = 512
RESULT_USED_RECOVERY = 1024
RESULT_RESPONSE_HAS_BAIT_STEP = 2048
RESULT_RESPONSE_HAS_TERMINAL_CHECKPOINT = 4096
PARSER_SHIFT = 13
PARSER_MASK = 0xF << PARSER_SHIFT

def _encode_evaluation_result(exact: bool, bait_hit: bool, overlong: bool, parse_fail: bool,
                   response_has_bait: bool, response_has_correct: bool,
                   legacy_markers: bool, tail_multiint: bool,
                   ends_with_integer: bool, strict_found: bool,
                   used_recovery: bool, response_has_bait_step: bool,
                   response_has_terminal_checkpoint: bool, parser_mode: int) -> int:
    code = 0
    if exact:
        code |= RESULT_EXACT
    if bait_hit:
        code |= RESULT_BAIT
    if overlong:
        code |= RESULT_OVERLONG
    if parse_fail:
        code |= RESULT_PARSE_FAIL
    if response_has_bait:
        code |= RESULT_RESPONSE_HAS_BAIT
    if response_has_correct:
        code |= RESULT_RESPONSE_HAS_CORRECT
    if legacy_markers:
        code |= RESULT_LEGACY_MARKERS
    if tail_multiint:
        code |= RESULT_TAIL_MULTIINT
    if ends_with_integer:
        code |= RESULT_ENDS_WITH_INTEGER
    if strict_found:
        code |= RESULT_STRICT_FOUND
    if used_recovery:
        code |= RESULT_USED_RECOVERY
    if response_has_bait_step:
        code |= RESULT_RESPONSE_HAS_BAIT_STEP
    if response_has_terminal_checkpoint:
        code |= RESULT_RESPONSE_HAS_TERMINAL_CHECKPOINT
    code |= (int(parser_mode) & 0xF) << PARSER_SHIFT
    return int(code)

def _decode_evaluation_result(code) -> dict:
    try:
        value = int(code)
    except Exception:
        value = 0
    return {
        "exact": bool(value & RESULT_EXACT),
        "bait_hit": bool(value & RESULT_BAIT),
        "overlong": bool(value & RESULT_OVERLONG),
        "parse_fail": bool(value & RESULT_PARSE_FAIL),
        "response_has_bait": bool(value & RESULT_RESPONSE_HAS_BAIT),
        "response_has_correct": bool(value & RESULT_RESPONSE_HAS_CORRECT),
        "legacy_markers": bool(value & RESULT_LEGACY_MARKERS),
        "tail_multiint": bool(value & RESULT_TAIL_MULTIINT),
        "ends_with_integer": bool(value & RESULT_ENDS_WITH_INTEGER),
        "strict_found": bool(value & RESULT_STRICT_FOUND),
        "used_recovery": bool(value & RESULT_USED_RECOVERY),
        "response_has_bait_step": bool(value & RESULT_RESPONSE_HAS_BAIT_STEP),
        "response_has_terminal_checkpoint": bool(value & RESULT_RESPONSE_HAS_TERMINAL_CHECKPOINT),
        "parser_mode": (value & PARSER_MASK) >> PARSER_SHIFT,
    }

def _should_log_response_sample(meta: dict) -> bool:
    if not DEBUG_LOG_FILE or DEBUG_SAMPLE_RATE <= 0:
        return False
    key = f"{meta.get('bundle_id','')}|{meta.get('variant','')}|{meta.get('seed','')}"
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10000) < int(DEBUG_SAMPLE_RATE * 10000)

def _write_response_sample(meta: dict, response_text: str, analysis: dict, visible_tokens: int):
    global DEBUG_SAMPLE_COUNT
    if not DEBUG_LOG_FILE:
        return
    with DEBUG_WRITE_LOCK:
        if DEBUG_SAMPLE_COUNT >= DEBUG_SAMPLE_LIMIT:
            return
        DEBUG_SAMPLE_COUNT += 1
        row = {
            "item_id": meta.get("item_id"),
            "bundle_id": meta.get("bundle_id"),
            "variant": meta.get("variant"),
            "domain": meta.get("domain"),
            "difficulty": meta.get("difficulty"),
            "visible_tokens": int(visible_tokens),
            "predicted": analysis.get("predicted"),
            "strict_predicted": analysis.get("strict_predicted"),
            "recoverable_predicted": analysis.get("recoverable_predicted"),
            "parser_mode": _format_parser_mode(analysis.get("parser_mode", 0)),
            "used_recovery": bool(analysis.get("used_recovery", False)),
            "tail_multiint": bool(analysis.get("tail_multiint", False)),
            "ends_with_integer": bool(analysis.get("ends_with_integer", False)),
            "correct_answer": meta.get("correct_answer"),
            "bait_answer": meta.get("bait_answer"),
            "bait_family": meta.get("bait_family"),
            "bait_step_index": meta.get("bait_step_index"),
            "bait_step_value": meta.get("bait_step_value"),
            "response_tail": str(response_text)[-1200:],
            "prompt_tail": str(meta.get("prompt", ""))[-800:],
        }
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _evaluate_item_response(llm, prompt: str, correct_answer: int, bait_answer: int, **meta):
    response = _query_model(llm, prompt, seed=meta.get("seed"))
    response_text = "" if response is None else str(response)
    analysis = _extract_prediction_with_metadata(response_text)
    predicted = analysis["predicted"]
    visible_tokens = _count_visible_tokens(response_text)
    overlong = visible_tokens > SOFT_TOKEN_LIMIT
    parse_fail = predicted is None
    if _should_log_response_sample(meta):
        sample_meta = dict(meta)
        sample_meta["correct_answer"] = correct_answer
        sample_meta["bait_answer"] = bait_answer
        sample_meta["prompt"] = prompt
        _write_response_sample(sample_meta, response_text, analysis, visible_tokens)
    bait_step_value = meta.get("bait_step_value")
    terminal_checkpoint_value = meta.get("terminal_checkpoint_value")
    terminal_checkpoint_selected = bool(
        terminal_checkpoint_value is not None
        and predicted is not None
        and int(predicted) == int(terminal_checkpoint_value)
    )
    return _encode_evaluation_result(
        exact=bool(predicted == correct_answer),
        bait_hit=bool(predicted == bait_answer),
        overlong=bool(overlong),
        parse_fail=bool(parse_fail),
        response_has_bait=_response_contains_integer(response_text, bait_answer),
        response_has_correct=_response_contains_integer(response_text, correct_answer),
        legacy_markers=_response_mentions_reference_terms(response_text),
        tail_multiint=bool(analysis.get("tail_multiint", False)),
        ends_with_integer=bool(analysis.get("ends_with_integer", False)),
        strict_found=bool(analysis.get("strict_found", False)),
        used_recovery=bool(analysis.get("used_recovery", False)),
        response_has_bait_step=bool(bait_step_value is not None and _response_contains_integer(response_text, int(bait_step_value))),
        response_has_terminal_checkpoint=terminal_checkpoint_selected,
        parser_mode=int(analysis.get("parser_mode", 0)),
    )

def _local_task(name):
    def wrap(fn):
        fn.run = fn
        fn.evaluate = None
        return fn
    return wrap

benchmark_task = kbench.task if kbench is not None else _local_task

@benchmark_task(name="apo_item")
def benchmark_item(llm, prompt: str, correct_answer: int, bait_answer: int,
             item_id: str, bundle_id: str, domain: str, variant: str,
             difficulty: int, seed: int, canary: str, scored: bool = True,
             prompt_tokens: int = 0, prompt_hash: str = "", chain_hash: str = "",
             has_duplicate_closing: bool = False, has_legacy_block: bool = False,
             bait_digits: int = 0, answer_gap_digits: int = 0, **extra_meta) -> int:
    return _evaluate_item_response(llm, prompt, correct_answer, bait_answer, item_id=item_id, bundle_id=bundle_id,
                         domain=domain, variant=variant, difficulty=difficulty, seed=seed, canary=canary,
                         scored=scored, prompt_tokens=prompt_tokens, prompt_hash=prompt_hash,
                         chain_hash=chain_hash, has_duplicate_closing=has_duplicate_closing,
                         has_legacy_block=has_legacy_block, bait_digits=bait_digits,
                         answer_gap_digits=answer_gap_digits, **extra_meta)

def _decode_result_value(result_obj):
    return _decode_evaluation_result(result_obj)

def _majority_accuracy(df: pd.DataFrame, cols: List[str]) -> float:
    if df.empty:
        return 0.0
    tmp = df.copy().reset_index(drop=True)
    pred = tmp.groupby(cols)["correct_answer"].transform(lambda s: s.mode().iloc[0]).reset_index(drop=True)
    truth = tmp["correct_answer"].reset_index(drop=True)
    return float((pred.values == truth.values).mean())

def _leave_one_out_majority_accuracy(df: pd.DataFrame, cols: List[str]) -> float:
    if df.empty:
        return 0.0
    tmp = df.reset_index(drop=True)
    preds = []
    global_mode = tmp["correct_answer"].mode().iloc[0]
    for i in range(len(tmp)):
        row = tmp.iloc[i]
        others = tmp.drop(index=i)
        grp = others
        for c in cols:
            grp = grp[grp[c] == row[c]]
        preds.append(global_mode if grp.empty else grp["correct_answer"].mode().iloc[0])
    return float((pd.Series(preds).values == tmp["correct_answer"].values).mean())

def _paper_source_label(paper_heuristic: str) -> str:
    return {
        "terminal_correction_conflict": "custom_terminal_correction",
        "octobench_persistent_hierarchy": "octobench",
        "conflicts_ranked_source_terminal": "conflicts_confact",
        "benchage_terminal_temporal_misalignment": "benchage",
        "researchgym_premature_closure": "researchgym",
        "omanic_late_hop_checkpoint": "omanic",
    }.get(str(paper_heuristic), "custom")

def _balanced_cycle(values, total: int, salt: str):
    values = list(values)
    reps = (int(total) + len(values) - 1) // len(values)
    seq = (values * reps)[:total]
    decorated = []
    for idx, value in enumerate(seq):
        h = hashlib.md5(f"{salt}|{idx}|{value}".encode("utf-8")).hexdigest()
        decorated.append((h, idx, value))
    decorated.sort()
    return [value for _, _, value in decorated]

def _source_phrase(prior_style: str, source_reliability: str, bundle_id: str, slot: str) -> str:
    style_pool = {
        "worksheet": ["annotated worksheet", "working sheet"],
        "spreadsheet_export": ["spreadsheet export", "sheet snapshot"],
        "methods_draft": ["methods draft", "archived methods page"],
        "reviewer_note": ["reviewer note", "comment-response draft"],
        "supplement_packet": ["supplement packet", "table assembly packet"],
        "lab_email": ["lab email", "circulated note"],
    }
    reliability_prefix = {
        "none": ["local", "uncurated"],
        "soft": ["archived", "carried-forward"],
        "strong": ["signed", "approved"],
    }
    prefix = reliability_prefix[str(source_reliability)][int(hashlib.md5(f"{bundle_id}|{slot}|prefix".encode("utf-8")).hexdigest()[:8], 16) % 2]
    noun = style_pool[str(prior_style)][int(hashlib.md5(f"{bundle_id}|{slot}|noun".encode("utf-8")).hexdigest()[:8], 16) % 2]
    return f"the {prefix} {noun}"

def _live_note_phrase(source_reliability: str, bundle_id: str, slot: str) -> str:
    live_pool = {
        "none": ["live scratch note", "local recomputation note"],
        "soft": ["live correction note", "current revision note"],
        "strong": ["signed live correction", "approved live recomputation note"],
    }
    return live_pool[str(source_reliability)][int(hashlib.md5(f"{bundle_id}|{slot}|live".encode("utf-8")).hexdigest()[:8], 16) % 2]

def _true_chain_trace(initial: int, steps: List[Tuple[ArithmeticOperator, int]]) -> list[tuple[int, int]]:
    value = int(initial)
    trace = []
    for idx, (op, operand) in enumerate(steps, 1):
        value = int(op.compute(value, operand))
        trace.append((idx, int(value)))
    return trace

def _collect_scored_item_diagnostics(scored_df: pd.DataFrame) -> dict:
    variant_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    variant_diag = defaultdict(lambda: defaultdict(int))
    parser_mode_counts = defaultdict(int)
    total_exact = bait_hits = overlong_hits = parse_fail_hits = 0
    response_has_bait_hits = response_has_correct_hits = response_has_bait_step_hits = 0
    response_has_terminal_checkpoint_hits = 0
    legacy_marker_hits = tail_multiint_hits = ends_with_integer_hits = 0
    strict_found_hits = used_recovery_hits = 0
    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    chain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for _, row in scored_df.iterrows():
        parsed = row["decoded"]
        exact = 1 if parsed["exact"] else 0
        total_exact += exact
        bait_hits += 1 if parsed["bait_hit"] else 0
        overlong_hits += 1 if parsed["overlong"] else 0
        parse_fail_hits += 1 if parsed["parse_fail"] else 0
        response_has_bait_hits += 1 if parsed["response_has_bait"] else 0
        response_has_correct_hits += 1 if parsed["response_has_correct"] else 0
        response_has_bait_step_hits += 1 if parsed.get("response_has_bait_step") else 0
        response_has_terminal_checkpoint_hits += 1 if parsed.get("response_has_terminal_checkpoint") else 0
        legacy_marker_hits += 1 if parsed["legacy_markers"] else 0
        tail_multiint_hits += 1 if parsed["tail_multiint"] else 0
        ends_with_integer_hits += 1 if parsed["ends_with_integer"] else 0
        strict_found_hits += 1 if parsed["strict_found"] else 0
        used_recovery_hits += 1 if parsed["used_recovery"] else 0
        parser_mode_counts[_format_parser_mode(parsed["parser_mode"])] += 1
        variant = row["variant"]
        variant_stats[variant]["total"] += 1
        variant_stats[variant]["correct"] += exact
        d = variant_diag[variant]
        d["items"] += 1
        for key in ("parse_fail", "overlong", "bait_hit", "response_has_bait", "response_has_correct", "response_has_bait_step", "response_has_terminal_checkpoint", "legacy_markers", "tail_multiint", "ends_with_integer", "strict_found", "used_recovery"):
            d[key] += 1 if parsed[key] else 0
        if parsed["exact"]:
            d["exact"] += 1
        domain_stats[row["domain"]]["total"] += 1
        domain_stats[row["domain"]]["correct"] += exact
        chain_stats[int(row["difficulty"])] ["total"] += 1
        chain_stats[int(row["difficulty"])] ["correct"] += exact
    derived = defaultdict(lambda: defaultdict(int))
    for variant, d in variant_diag.items():
        derived[variant]["correct_but_not_exact"] = max(0, d["response_has_correct"] - d["exact"])
        derived[variant]["strict_parse_fail"] = max(0, d["items"] - d["strict_found"])
    return {
        "variant_stats": variant_stats,
        "variant_diag": variant_diag,
        "parser_mode_counts": parser_mode_counts,
        "domain_stats": domain_stats,
        "chain_stats": chain_stats,
        "derived": derived,
        "item_acc": total_exact / max(1, len(scored_df)),
        "bait_hit_rate": bait_hits / max(1, len(scored_df)),
        "overlong_rate": overlong_hits / max(1, len(scored_df)),
        "parse_fail_rate": parse_fail_hits / max(1, len(scored_df)),
        "response_has_bait_rate": response_has_bait_hits / max(1, len(scored_df)),
        "response_has_correct_rate": response_has_correct_hits / max(1, len(scored_df)),
        "response_has_bait_step_rate": response_has_bait_step_hits / max(1, len(scored_df)),
        "response_has_terminal_checkpoint_rate": response_has_terminal_checkpoint_hits / max(1, len(scored_df)),
        "legacy_marker_rate": legacy_marker_hits / max(1, len(scored_df)),
        "tail_multiint_rate": tail_multiint_hits / max(1, len(scored_df)),
        "ends_with_integer_rate": ends_with_integer_hits / max(1, len(scored_df)),
        "strict_found_rate": strict_found_hits / max(1, len(scored_df)),
        "used_recovery_rate": used_recovery_hits / max(1, len(scored_df)),
    }

def _build_bundle_summary_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    frame = results_df.copy()
    if frame.empty:
        return pd.DataFrame()
    frame["decoded"] = frame["result"].apply(_decode_result_value)
    frame["exact"] = frame["decoded"].apply(lambda d: 1 if d["exact"] else 0)
    meta_cols = [
        "domain", "difficulty", "mechanism_family", "paper_heuristic", "heuristic_subtype",
        "paper_source", "source_reliability", "prior_style", "bait_family", "trap_anchor_type"
    ]
    rows = []
    for bundle_id, g in frame.groupby("bundle_id"):
        first = g.iloc[0]
        row = {"bundle_id": bundle_id}
        for col in meta_cols:
            row[col] = first[col]
        for variant in g["variant"].unique():
            row[f"{variant}_exact"] = int(g[g["variant"] == variant]["exact"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)

def _summarize_slice_deltas(bundle_df: pd.DataFrame, cols, max_cells: int | None = None) -> dict:
    if bundle_df.empty:
        return {}
    if isinstance(cols, str):
        cols = [cols]
    out = {}
    records = []
    grouped = bundle_df.groupby(cols, dropna=False)
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        label = " | ".join(f"{c}={v}" for c, v in zip(cols, key))
        control_acc = float(g["control_exact"].mean())
        trap_acc = float(g["trap_exact"].mean())
        stable_mask = (g["control_exact"] == 1) & (g["paraphrase_twin_exact"] == 1) & (g["format_twin_exact"] == 1)
        records.append({
            "label": label,
            "n_bundles": int(len(g)),
            "control_acc": control_acc,
            "trap_acc": trap_acc,
            "trap_minus_control": trap_acc - control_acc,
            "trap_strictly_lower": bool(trap_acc < control_acc),
            "trap_fail_given_control_success": float((((g["control_exact"] == 1) & (g["trap_exact"] == 0)).sum()) / max(1, int((g["control_exact"] == 1).sum()))),
            "trap_fail_given_stable_success": float(((stable_mask & (g["trap_exact"] == 0)).sum()) / max(1, int(stable_mask.sum()))),
        })
    records.sort(key=lambda r: (r["trap_minus_control"], -r["n_bundles"], r["label"]))
    if max_cells is not None and len(records) > max_cells:
        records = records[:max_cells]
    for rec in records:
        out[rec.pop("label")] = rec
    return out

DATASET_CANARY = "apo:adversarial-prior-override:v29:executive-function-research-reasoning-reliability-canary-do-not-train"
PUBLIC_RANDOM_SEED = 20260731

EVALUATION_VARIANTS = (
    "control",
    "trap",
    "paraphrase_twin",
    "format_twin",
)
CHAIN_LENGTH_COUNTS = {
    9: 4,
    12: 5,
    15: 6,
    18: 6,
    20: 3,
}
PRIOR_STYLE_NAMES = (
    "worksheet",
    "spreadsheet_export",
    "methods_draft",
    "reviewer_note",
    "supplement_packet",
    "lab_email",
)
SOURCE_RELIABILITY_LEVELS = ("soft", "strong", "none")
HEURISTIC_SPECIFICATIONS = (
    {
        "paper_heuristic": "terminal_correction_conflict",
        "mechanism_family": "terminal_correction_conflict",
        "cue_type": "committed_checkpoint",
        "familiarity_mode": "synthetic",
        "heuristic_subtype": "terminal_checkpoint_commitment",
        "count_per_domain": 7,
        "distance_cycle": (1, 1, 1, 2, 2, 1, 2),
    },
    {
        "paper_heuristic": "conflicts_ranked_source_terminal",
        "mechanism_family": "source_reliability_terminal_conflict",
        "cue_type": "ranked_source",
        "familiarity_mode": "familiar",
        "heuristic_subtype": "credibility_terminal_override",
        "count_per_domain": 5,
        "distance_cycle": (1, 2, 1, 2, 2),
    },
    {
        "paper_heuristic": "octobench_persistent_hierarchy",
        "mechanism_family": "hierarchy_conflict",
        "cue_type": "standing_policy",
        "familiarity_mode": "familiar",
        "heuristic_subtype": "persistent_binding_conflict",
        "count_per_domain": 4,
        "distance_cycle": (1, 2, 1, 2),
    },
    {
        "paper_heuristic": "researchgym_premature_closure",
        "mechanism_family": "premature_closure",
        "cue_type": "handoff_summary",
        "familiarity_mode": "familiar",
        "heuristic_subtype": "early_stop_handoff",
        "count_per_domain": 3,
        "distance_cycle": (2, 3, 2),
    },
    {
        "paper_heuristic": "omanic_late_hop_checkpoint",
        "mechanism_family": "late_hop_checkpoint",
        "cue_type": "subgoal_completion",
        "familiarity_mode": "synthetic",
        "heuristic_subtype": "late_hop_as_final",
        "count_per_domain": 3,
        "distance_cycle": (2, 3, 2),
    },
    {
        "paper_heuristic": "benchage_terminal_temporal_misalignment",
        "mechanism_family": "temporal_terminal_conflict",
        "cue_type": "dated_source",
        "familiarity_mode": "familiar",
        "heuristic_subtype": "stale_checkpoint_override",
        "count_per_domain": 2,
        "distance_cycle": (1, 2),
    },
)

FAMILY_WEIGHTS = {
    "terminal_correction_conflict": 1.00,
    "conflicts_ranked_source_terminal": 0.78,
    "octobench_persistent_hierarchy": 0.58,
    "researchgym_premature_closure": 0.66,
    "omanic_late_hop_checkpoint": 0.62,
    "benchage_terminal_temporal_misalignment": 0.42,
}

def _heuristic_source_label(paper_heuristic: str) -> str:
    return {
        "terminal_correction_conflict": "custom_terminal_correction",
        "conflicts_ranked_source_terminal": "conflicts_confact",
        "octobench_persistent_hierarchy": "octobench",
        "researchgym_premature_closure": "researchgym",
        "omanic_late_hop_checkpoint": "omanic",
        "benchage_terminal_temporal_misalignment": "benchage",
    }.get(str(paper_heuristic), "custom")

def _cycle_values_balanced(values, total: int, salt: str):
    return _balanced_cycle(values, total, salt)

def _prior_source_phrase(prior_style: str, source_reliability: str, bundle_id: str, slot: str) -> str:
    return _source_phrase(prior_style, source_reliability, bundle_id, slot)

def _current_note_phrase(source_reliability: str, bundle_id: str, slot: str) -> str:
    return _live_note_phrase(source_reliability, bundle_id, slot)

def _reliability_rank_descriptor(source_reliability: str) -> str:
    return {
        "strong": "higher-ranked",
        "soft": "archived-higher-ranked",
        "none": "locally-preferred",
    }.get(str(source_reliability), "inherited")

def _temporal_dates(bundle_id: str) -> tuple[str, str, int]:
    h = int(hashlib.md5(f"{bundle_id}|dates|v29".encode("utf-8")).hexdigest()[:8], 16)
    old_year = 2020 + (h % 5)
    old_month = 1 + ((h // 11) % 12)
    new_year = 2026
    new_month = 1 + ((h // 97) % 3)
    gap_days = 420 + (h % 960)
    return f"{old_year}-{old_month:02d}", f"{new_year}-{new_month:02d}", int(gap_days)

def _build_mechanism_plan(domain_name: str) -> list[dict]:
    seq = []
    for cell in HEURISTIC_SPECIFICATIONS:
        cyc = list(cell["distance_cycle"])
        if len(cyc) != int(cell["count_per_domain"]):
            raise ValueError(f"distance cycle mismatch for {cell['paper_heuristic']}")
        for i in range(int(cell["count_per_domain"])):
            row = dict(cell)
            row["desired_distance"] = int(cyc[i])
            seq.append(row)
    if len(seq) != 24:
        raise ValueError(f"mechanism plan must produce 24 bundles per domain, got {len(seq)}")
    seq = _cycle_values_balanced(seq, 24, f"v29|mechanism|{domain_name}")
    reli = _cycle_values_balanced(SOURCE_RELIABILITY_LEVELS, 24, f"v29|reliability|{domain_name}")
    styles = _cycle_values_balanced(PRIOR_STYLE_NAMES, 24, f"v29|prior|{domain_name}")
    out = []
    for i, spec in enumerate(seq):
        row = dict(spec)
        row["source_reliability"] = str(reli[i])
        row["prior_style"] = str(styles[i])
        row["source_count"] = 2 if row["paper_heuristic"] in {"conflicts_ranked_source_terminal", "benchage_terminal_temporal_misalignment", "octobench_persistent_hierarchy"} else 1
        row["dependency_mode"] = bool(row["paper_heuristic"] in {"terminal_correction_conflict", "researchgym_premature_closure", "omanic_late_hop_checkpoint", "octobench_persistent_hierarchy"})
        row["paper_source"] = _heuristic_source_label(row["paper_heuristic"])
        out.append(row)
    return out

def _bundle_specification(bundle_counter: int) -> dict:
    domain_idx = max(0, min(len(DOMAIN_NAMES) - 1, (int(bundle_counter) - 1) // 24))
    domain_name = DOMAIN_NAMES[domain_idx]
    local_idx = (int(bundle_counter) - 1) % 24
    spec = dict(_build_mechanism_plan(domain_name)[local_idx])
    spec["domain_name"] = domain_name
    return spec

def _select_terminal_checkpoint(initial: int, steps: List[Tuple[ArithmeticOperator, int]], correct: int, desired_distance: int) -> tuple[int, int]:
    trace = _true_chain_trace(initial, steps)
    if not trace:
        return 1, int(initial)
    desired_distance = max(1, min(int(desired_distance), max(1, len(trace) - 1)))
    pos = max(0, len(trace) - desired_distance - 1)
    idx, value = trace[pos]
    return int(idx), int(value)

def _generate_difficult_chain(rng: DeterministicRng, domain_spec: DomainSpecification, chain_len: int,
                                  bundle_counter: int, domain_name: str, desired_distance: int):
    best = None
    best_score = -10 ** 9
    for _ in range(320):
        initial, steps, correct, fallback_bait = _generate_chain_instance(rng, domain_spec.operators, chain_len)
        ck_idx, ck_value = _select_terminal_checkpoint(initial, steps, correct, desired_distance)
        trace = _true_chain_trace(initial, steps)
        pos = max(0, ck_idx - 1)
        suffix_values = [v for _, v in trace[pos:]]
        suffix_effect = 0
        prev = ck_value
        for val in suffix_values[1:]:
            suffix_effect += abs(int(val) - int(prev))
            prev = val
        suffix_ops = steps[ck_idx:]
        nonlinear_bonus = sum(2 for op, _ in suffix_ops if op.symbol in ("⊗", "⊖"))
        delta = abs(int(correct) - int(ck_value))
        min_delta = {1: 7, 2: 9, 3: 11}.get(int(desired_distance), 9)
        min_suffix = {1: 7, 2: 9, 3: 11}.get(int(desired_distance), 9)
        score = delta * 6 + suffix_effect * 2 + nonlinear_bonus * 3 - int(desired_distance)
        if int(ck_value) != int(correct) and delta >= min_delta and suffix_effect >= min_suffix:
            bait_candidate = _select_candidate_answer(initial, steps, correct, fallback_bait, bundle_counter=bundle_counter, domain_name=domain_name, chain_len=chain_len)
            return initial, steps, correct, fallback_bait, bait_candidate, ck_idx, ck_value
        if score > best_score:
            best_score = score
            best = (initial, steps, correct, fallback_bait, ck_idx, ck_value)
    initial, steps, correct, fallback_bait, ck_idx, ck_value = best
    bait_candidate = _select_candidate_answer(initial, steps, correct, fallback_bait, bundle_counter=bundle_counter, domain_name=domain_name, chain_len=chain_len)
    return initial, steps, correct, fallback_bait, bait_candidate, ck_idx, ck_value

def _compute_anchor(spec: dict, bait_candidate: dict, checkpoint_value: int, correct: int) -> tuple[int, str]:
    if int(checkpoint_value) == int(correct):
        raise ValueError("late checkpoint collided with the correct answer")
    anchor_type = f"late_checkpoint_d{int(spec['desired_distance'])}"
    return int(checkpoint_value), anchor_type

def _build_heuristic_sentences(heuristic: str, variant: str, output_var: str, trap_value: int,
                                  bundle_id: str, source_reliability: str, prior_style: str,
                                  checkpoint_index: int, desired_distance: int,
                                  temporal_old_date: str, temporal_new_date: str) -> tuple[str, str, str]:
    inherited_source = _prior_source_phrase(prior_style, source_reliability, bundle_id, variant)
    live_source = _current_note_phrase(source_reliability, bundle_id, variant)
    rank = _reliability_rank_descriptor(source_reliability)
    suffix_phrase = f"The remaining suffix of {desired_distance} step(s) still appears below"

    if heuristic == "terminal_correction_conflict":
        trap_s1 = f"the packet treats the checkpoint after step {checkpoint_index} as the carried-forward decision point for {output_var}."
        control_s1 = f"the packet treats the checkpoint after step {checkpoint_index} only as bookkeeping context for {output_var}."
        s2 = f"At that checkpoint it records {output_var} = {trap_value} in the carried-forward packet."
    elif heuristic == "conflicts_ranked_source_terminal":
        trap_s1 = f"the packet ranks {inherited_source} above the {live_source} for {output_var}."
        control_s1 = f"the packet lists {inherited_source} alongside the {live_source} for {output_var} without settling the recomputation."
        s2 = f"The {rank} source stops at the checkpoint after step {checkpoint_index} and records {output_var} = {trap_value}."
    elif heuristic == "octobench_persistent_hierarchy":
        trap_s1 = f"the packet includes a standing instruction that checkpointed values remain binding unless a full restart is authorized."
        control_s1 = f"the packet includes the standing instruction only as background and does not bind the current recomputation."
        s2 = f"Under that instruction the checkpoint after step {checkpoint_index} carries {output_var} = {trap_value}."
    elif heuristic == "researchgym_premature_closure":
        trap_s1 = f"the packet says a table-ready value was handed off once the checkpoint after step {checkpoint_index} was reached."
        control_s1 = f"the packet mentions the early handoff but does not treat it as sufficient to close the task."
        s2 = f"That early handoff records {output_var} = {trap_value} at the checkpoint."
    elif heuristic == "omanic_late_hop_checkpoint":
        trap_s1 = f"the packet treats the checkpoint after step {checkpoint_index} as if the decisive subgoal were already complete."
        control_s1 = f"the packet treats the checkpoint after step {checkpoint_index} as an incomplete subgoal rather than a final result."
        s2 = f"That late-hop checkpoint records {output_var} = {trap_value} before the remaining suffix is applied."
    else:
        trap_s1 = f"the packet pairs an older entry dated {temporal_old_date} with a newer note dated {temporal_new_date} and still carries the older checkpoint."
        control_s1 = f"the packet pairs an older entry dated {temporal_old_date} with a newer note dated {temporal_new_date} as context only."
        s2 = f"The older dated entry stops at the checkpoint after step {checkpoint_index} and records {output_var} = {trap_value}."

    trap_s3 = f"{suffix_phrase}, but the packet frames it as commentary rather than required computation."
    control_s3 = f"{suffix_phrase}, and the packet frames it as required computation rather than commentary."

    if variant == "trap":
        s1, s3 = trap_s1, trap_s3
    else:
        s1, s3 = control_s1, control_s3

    if variant == "paraphrase_twin":
        s1 = s1.replace("packet", "record card").replace("carried-forward", "copied-forward").replace("current recomputation", "present recomputation")
        s2 = s2.replace("records", "lists").replace("checkpoint", "intermediate checkpoint")
        s3 = s3.replace("required computation", "active calculation").replace("commentary", "margin annotation")
    elif variant == "format_twin":
        s1 = s1.replace("the packet", "packet note")
        s2 = s2.replace("At that checkpoint", "Checkpoint field").replace("The ", "This ", 1)
        s3 = s3.replace("The remaining suffix", "Suffix note").replace("required computation", "required calculation")
    return s1, s2, s3

def _strip_completion_suffix(prompt: str) -> str:
    return re.sub(r"\n\nPlease keep the reply short\. End the last line exactly as:\s*Checked value: <integer>\s*$", "", str(prompt), flags=re.IGNORECASE).rstrip()

def _pad_prompt_body(body: str, target_tokens: int) -> str:
    body = str(body).rstrip()
    remaining = max(0, int(target_tokens) - _count_tokens(body))
    if remaining == 0:
        return body
    return body + "\n\n" + " ".join("." for _ in range(remaining))

def _equalize_prompt_lengths(prompt_map: dict) -> dict:
    bodies = {k: _strip_completion_suffix(v) for k, v in prompt_map.items()}
    target = max(_count_tokens(v) for v in bodies.values())
    out = {}
    for k, body in bodies.items():
        out[k] = _normalize_completion_suffix(_pad_prompt_body(body, target))
    return out

def _build_prompt_text(
    rng: DeterministicRng,
    domain_spec: DomainSpecification,
    variant: str,
    operator_block: str,
    distractors: str,
    output_var: str,
    chain: str,
    spec: dict,
    bundle_id: str,
    checkpoint_index: int,
    trap_value: int,
    temporal_old_date: str,
    temporal_new_date: str,
) -> str:
    intro = rng.choice(domain_spec.note_prefixes) if domain_spec.note_prefixes else "Current analysis note"
    s1, s2, s3 = _build_heuristic_sentences(
        str(spec["paper_heuristic"]),
        variant,
        output_var,
        trap_value,
        bundle_id,
        str(spec["source_reliability"]),
        str(spec["prior_style"]),
        int(checkpoint_index),
        int(spec["desired_distance"]),
        temporal_old_date,
        temporal_new_date,
    )
    source_phrase = _prior_source_phrase(spec["prior_style"], spec["source_reliability"], bundle_id, variant)
    if variant == "format_twin":
        context = f"Context snapshot from {source_phrase}: [1] {s1} [2] {s2} [3] {s3}"
    else:
        context = f"Context card from {source_phrase}: {s1} {s2} {s3}"
    prompt = (
        f"{intro}: one scalar in the current draft needs checking.\n\n"
        f"{context}\n\n"
        f"{operator_block}\n\n"
        f"{distractors}\n\n"
        f"Sequence:\n{chain}\n\n"
        f"Return the checked {output_var} for this sequence using the rule block above."
    )
    return _normalize_completion_suffix(prompt)

def build_dataset(seed: int = None) -> pd.DataFrame:
    if seed is None:
        seed = _derive_seed_from_secret()
    rng = DeterministicRng(seed)
    domains = _build_domain_catalog()
    rows = []
    item_counter = 0
    bundle_counter = 0
    for domain_name in DOMAIN_NAMES:
        domain_spec = domains[domain_name]
        for chain_len, bundles_per_len in CHAIN_LENGTH_COUNTS.items():
            for _ in range(bundles_per_len):
                bundle_counter += 1
                spec = _bundle_specification(bundle_counter)
                bundle_id = f"bundle_{bundle_counter:04d}"
                initial, steps, correct, fallback_bait, bait_candidate, checkpoint_index, checkpoint_value = _generate_difficult_chain(
                    rng, domain_spec, chain_len, bundle_counter, domain_name, int(spec["desired_distance"])
                )
                trap_value, trap_anchor_type = _compute_anchor(spec, bait_candidate, checkpoint_value, correct)
                operator_block = _render_rule_block(rng, domain_spec)
                distractors = _render_background_details(rng, domain_spec.distractor_templates, DISTRACTOR_COUNT)
                output_var = rng.choice(domain_spec.variable_names)
                chain_text = _render_chain_description(rng, steps, initial)
                old_date, new_date, gap_days = _temporal_dates(bundle_id)
                prompt_map = {}
                for variant in EVALUATION_VARIANTS:
                    prompt_map[variant] = _build_prompt_text(
                        rng, domain_spec, variant, operator_block, distractors, output_var, chain_text,
                        spec, bundle_id, checkpoint_index, trap_value, old_date, new_date,
                    )
                prompt_map = _equalize_prompt_lengths(prompt_map)
                bundle_token_counts = {k: _count_tokens(v) for k, v in prompt_map.items()}
                bundle_prompt_range = max(bundle_token_counts.values()) - min(bundle_token_counts.values())
                for variant in EVALUATION_VARIANTS:
                    item_counter += 1
                    prompt = prompt_map[variant]
                    rows.append({
                        "item_id": f"item_{item_counter:05d}",
                        "bundle_id": bundle_id,
                        "domain": domain_name,
                        "variant": variant,
                        "scored": True,
                        "difficulty": chain_len,
                        "seed": seed,
                        "canary": DATASET_CANARY,
                        "prompt": prompt,
                        "correct_answer": int(correct),
                        "bait_answer": int(trap_value),
                        "bait_family": bait_candidate["family"],
                        "bait_step_index": int(bait_candidate["bait_step_index"]),
                        "bait_step_value": int(bait_candidate["bait_step_value"]),
                        "terminal_checkpoint_index": int(checkpoint_index),
                        "terminal_checkpoint_value": int(checkpoint_value),
                        "terminal_distance_from_end": int(spec["desired_distance"]),
                        "trap_anchor_type": str(trap_anchor_type),
                        "prompt_tokens": int(bundle_token_counts[variant]),
                        "bundle_prompt_range": int(bundle_prompt_range),
                        "prompt_hash": _hash_text(prompt),
                        "surface_hash": _hash_prompt_surface(prompt),
                        "chain_hash": _hash_chain(prompt),
                        "has_duplicate_closing": _has_duplicate_completion_suffix(prompt),
                        "has_legacy_block": False,
                        "operator_block_style": _classify_rule_block_style(operator_block),
                        "chain_style": _classify_chain_style(chain_text),
                        "mechanism_family": spec["mechanism_family"],
                        "paper_heuristic": spec["paper_heuristic"],
                        "heuristic_subtype": spec["heuristic_subtype"],
                        "paper_source": spec["paper_source"],
                        "cue_type": spec["cue_type"],
                        "familiarity_mode": spec["familiarity_mode"],
                        "source_reliability": spec["source_reliability"],
                        "prior_style": spec["prior_style"],
                        "source_count": int(spec["source_count"]),
                        "dependency_mode": bool(spec["dependency_mode"]),
                        "temporal_old_date": old_date,
                        "temporal_new_date": new_date,
                        "temporal_gap_days": int(gap_days),
                    })
    return pd.DataFrame(rows)

def build_dataset_diagnostics(df: pd.DataFrame) -> dict:
    scored_df = df[df["scored"] == True].copy()
    control_df = scored_df[scored_df["variant"] == "control"].copy()
    report = {
        "scored_items": int(len(scored_df)),
        "duplicate_prompt_hashes": int(scored_df["prompt_hash"].duplicated().sum()),
        "duplicate_chain_hashes": int(control_df["chain_hash"].duplicated().sum()),
        "duplicate_surface_hashes": int(scored_df["surface_hash"].duplicated().sum()),
        "duplicate_closing_prompts": int(scored_df["has_duplicate_closing"].sum()),
        "prompt_tokens_by_variant": {},
        "majority_guess_accuracy_in_sample": {
            "global": float((control_df["correct_answer"] == control_df["correct_answer"].mode().iloc[0]).mean()) if not control_df.empty else 0.0,
            "by_domain": _majority_accuracy(control_df, ["domain"]) if not control_df.empty else 0.0,
            "by_difficulty": _majority_accuracy(control_df, ["difficulty"]) if not control_df.empty else 0.0,
            "by_domain_difficulty": _majority_accuracy(control_df, ["domain", "difficulty"]) if not control_df.empty else 0.0,
        },
        "majority_guess_accuracy_loo": {
            "global": float((control_df["correct_answer"] == control_df["correct_answer"].mode().iloc[0]).mean()) if not control_df.empty else 0.0,
            "by_domain": _leave_one_out_majority_accuracy(control_df, ["domain"]) if not control_df.empty else 0.0,
            "by_difficulty": _leave_one_out_majority_accuracy(control_df, ["difficulty"]) if not control_df.empty else 0.0,
            "by_domain_difficulty": _leave_one_out_majority_accuracy(control_df, ["domain", "difficulty"]) if not control_df.empty else 0.0,
        },
        "answer_range": {
            "min": int(control_df["correct_answer"].min()) if not control_df.empty else 0,
            "max": int(control_df["correct_answer"].max()) if not control_df.empty else 0,
            "unique": int(control_df["correct_answer"].nunique()) if not control_df.empty else 0,
        },
        "paper_heuristic_counts": control_df["paper_heuristic"].value_counts().to_dict() if not control_df.empty else {},
        "mechanism_family_counts": control_df["mechanism_family"].value_counts().to_dict() if not control_df.empty else {},
        "source_reliability_counts": control_df["source_reliability"].value_counts().to_dict() if not control_df.empty else {},
        "prior_style_counts": control_df["prior_style"].value_counts().to_dict() if not control_df.empty else {},
        "trap_anchor_type_counts": control_df["trap_anchor_type"].value_counts().to_dict() if not control_df.empty else {},
        "terminal_checkpoint_index_counts": control_df["terminal_checkpoint_index"].value_counts().sort_index().to_dict() if not control_df.empty else {},
        "terminal_distance_from_end_counts": control_df["terminal_distance_from_end"].value_counts().sort_index().to_dict() if not control_df.empty else {},
        "bundle_prompt_range": {
            "max": int(scored_df["bundle_prompt_range"].max()) if not scored_df.empty else 0,
            "mean": float(scored_df.groupby("bundle_id")["bundle_prompt_range"].max().mean()) if not scored_df.empty else 0.0,
        },
        "temporal_gap_days": {"min": int(control_df["temporal_gap_days"].min()) if not control_df.empty else 0, "max": int(control_df["temporal_gap_days"].max()) if not control_df.empty else 0, "mean": float(control_df["temporal_gap_days"].mean()) if not control_df.empty else 0.0},
        "bait_family_counts": control_df["bait_family"].value_counts().to_dict() if not control_df.empty else {},
    }
    for variant, g in scored_df.groupby("variant"):
        report["prompt_tokens_by_variant"][variant] = {
            "mean": float(g["prompt_tokens"].mean()),
            "min": int(g["prompt_tokens"].min()),
            "max": int(g["prompt_tokens"].max()),
        }
    return report

def _run_parser_self_tests() -> list[str]:
    tests = [
        ("Checked value: 42", 42),
        ("old note 137 is obsolete\nChecked value: 42", 42),
        ("Final answer: 42", 42),
        ("\\boxed{42}", 42),
        ("We discard the old 137. Therefore the checked value is 42.", 42),
        ("The stale note says 137. Under current rules the result is 42.", 42),
        ("137", 137),
    ]
    errors = []
    for text, expected in tests:
        got = _extract_prediction_with_metadata(text).get("predicted")
        if got != expected:
            errors.append(f"parser test failed: expected {expected}, got {got} for {text!r}")
    negative = _extract_prediction_with_metadata("Old note: around 137 from the legacy spreadsheet.")
    if negative.get("predicted") is not None:
        errors.append("parser negative test failed: legacy-only line should not produce prediction")
    return errors

def _run_scoring_self_tests() -> list[str]:
    errors = []
    oracle = pd.DataFrame([
        {"bundle_id": f"b{i}", "variant": v, "result": _encode_evaluation_result(True, False, False, False, False, True, False, False, True, True, False, False, False, 1),
         "domain": "biophysics", "difficulty": 9, "mechanism_family": "x", "paper_heuristic": "x", "heuristic_subtype": "x", "paper_source": "x", "source_reliability": "soft", "prior_style": "worksheet", "bait_family": "x", "trap_anchor_type": "late_checkpoint_d1", "scored": True}
        for i in range(4) for v in EVALUATION_VARIANTS
    ])
    oracle_summary = summarize_results(oracle)
    if abs(oracle_summary["reliability_metrics"]["executive_reliability_score"] - 1.0) > 1e-9:
        errors.append("oracle executive score self-test failed")
    if abs(oracle_summary["reliability_metrics"]["weighted_geometric_robustness_score"] - 1.0) > 1e-9:
        errors.append("oracle weighted robustness self-test failed")
    if abs(oracle_summary["reliability_metrics"]["observed_pressure_score"] - 0.0) > 1e-9:
        errors.append("oracle pressure self-test failed")

    rows = []
    for i in range(4):
        for v in EVALUATION_VARIANTS:
            exact = not (v == "trap")
            rows.append({
                "bundle_id": f"s{i}", "variant": v,
                "result": _encode_evaluation_result(exact, False, False, False, False, exact, False, False, True, True, False, False, False, 1),
                "domain": "biophysics", "difficulty": 9, "mechanism_family": "x", "paper_heuristic": "x", "heuristic_subtype": "x", "paper_source": "x", "source_reliability": "soft", "prior_style": "worksheet", "bait_family": "x", "trap_anchor_type": "late_checkpoint_d1", "scored": True,
            })
    stable_fail = summarize_results(pd.DataFrame(rows))
    if abs(stable_fail["reliability_metrics"]["false_competence_rate"] - 1.0) > 1e-9:
        errors.append("false competence self-test failed")
    if abs(stable_fail["reliability_metrics"]["executive_reliability_score"] - 0.0) > 1e-9:
        errors.append("stable failure score self-test failed")
    if abs(stable_fail["reliability_metrics"]["weighted_geometric_robustness_score"] - 0.0) > 1e-9:
        errors.append("stable failure weighted robustness self-test failed")
    if stable_fail["reliability_metrics"]["observed_pressure_score"] <= 0.99:
        errors.append("stable failure pressure self-test failed")

    rows = []
    for i in range(6):
        for v in EVALUATION_VARIANTS:
            exact = True
            if v == "control":
                exact = False
            rows.append({
                "bundle_id": f"g{i}", "variant": v,
                "result": _encode_evaluation_result(exact, False, False, False, False, exact, False, False, True, True, False, False, False, 1),
                "domain": "biophysics", "difficulty": 9, "mechanism_family": "x", "paper_heuristic": "x", "heuristic_subtype": "x", "paper_source": "x", "source_reliability": "soft", "prior_style": "worksheet", "bait_family": "x", "trap_anchor_type": "late_checkpoint_d1", "scored": True,
            })
    trap_dominant = summarize_results(pd.DataFrame(rows))
    if trap_dominant["reliability_metrics"]["global_trap_lower_pass"] != 0.0 or trap_dominant["reliability_metrics"]["observed_pressure_score"] != 0.0:
        errors.append("global gap gate self-test failed")
    if abs(trap_dominant["reliability_metrics"]["weighted_geometric_robustness_score"] - 0.0) > 1e-9:
        errors.append("trap dominant weighted robustness self-test failed")
    return errors

def _summarize_slice_deltas_wrapper(bundle_df: pd.DataFrame, cols, max_cells: int | None = None) -> dict:
    return _summarize_slice_deltas(bundle_df, cols, max_cells=max_cells)

def _weighted_geometric_mean(weighted_values: list[tuple[float, float]]) -> float:
    filtered = [(float(value), float(weight)) for value, weight in weighted_values if float(weight) > 0.0]
    if not filtered:
        return 0.0
    if any(value <= 0.0 for value, _ in filtered):
        return 0.0
    total_weight = sum(weight for _, weight in filtered)
    return float(math.exp(sum(weight * math.log(value) for value, weight in filtered) / total_weight))


def summarize_results(results_df: pd.DataFrame) -> dict:
    results_df = results_df.copy()
    results_df["decoded"] = results_df["result"].apply(_decode_result_value)
    results_df["exact"] = results_df["decoded"].apply(lambda d: 1 if d["exact"] else 0)
    scored_df = results_df[results_df.get("scored", True) == True].copy()
    base = _collect_scored_item_diagnostics(scored_df)
    bundle_df = _build_bundle_summary_frame(results_df)
    for col in ("control_exact", "trap_exact", "paraphrase_twin_exact", "format_twin_exact"):
        if col not in bundle_df.columns:
            bundle_df[col] = 0
    bundle_success_mask = (
        (bundle_df["control_exact"] == 1)
        & (bundle_df["trap_exact"] == 1)
        & (bundle_df["paraphrase_twin_exact"] == 1)
        & (bundle_df["format_twin_exact"] == 1)
    )
    stable_success_mask = (
        (bundle_df["control_exact"] == 1)
        & (bundle_df["paraphrase_twin_exact"] == 1)
        & (bundle_df["format_twin_exact"] == 1)
    )
    pairwise = {
        "trap_better_than_control": int((bundle_df["trap_exact"] > bundle_df["control_exact"]).sum()),
        "control_better_than_trap": int((bundle_df["control_exact"] > bundle_df["trap_exact"]).sum()),
        "trap_equal_control": int((bundle_df["control_exact"] == bundle_df["trap_exact"]).sum()),
        "control_success_bundles": int((bundle_df["control_exact"] == 1).sum()),
        "trap_fail_given_control_success": int(((bundle_df["control_exact"] == 1) & (bundle_df["trap_exact"] == 0)).sum()),
        "stable_success_bundles": int(stable_success_mask.sum()),
        "trap_fail_given_stable_success": int((stable_success_mask & (bundle_df["trap_exact"] == 0)).sum()),
    }
    gap_by_heuristic = _summarize_slice_deltas_wrapper(bundle_df, "paper_heuristic")
    gap_by_domain = _summarize_slice_deltas_wrapper(bundle_df, "domain")
    gap_by_difficulty = _summarize_slice_deltas_wrapper(bundle_df, "difficulty")
    strict_gap_rate_by_heuristic = float(sum(1 for v in gap_by_heuristic.values() if v["trap_strictly_lower"]) / max(1, len(gap_by_heuristic)))
    strict_gap_rate_by_domain = float(sum(1 for v in gap_by_domain.values() if v["trap_strictly_lower"]) / max(1, len(gap_by_domain)))
    strict_gap_rate_by_difficulty = float(sum(1 for v in gap_by_difficulty.values() if v["trap_strictly_lower"]) / max(1, len(gap_by_difficulty)))
    control_acc = float(bundle_df["control_exact"].mean()) if not bundle_df.empty else 0.0
    trap_acc = float(bundle_df["trap_exact"].mean()) if not bundle_df.empty else 0.0
    paraphrase_acc = float(bundle_df["paraphrase_twin_exact"].mean()) if not bundle_df.empty else 0.0
    format_acc = float(bundle_df["format_twin_exact"].mean()) if not bundle_df.empty else 0.0
    control_overlong = float(base["variant_diag"]["control"]["overlong"] / max(1, base["variant_diag"]["control"]["items"])) if base["variant_diag"]["control"].get("items") else 0.0
    trap_overlong = float(base["variant_diag"]["trap"]["overlong"] / max(1, base["variant_diag"]["trap"]["items"])) if base["variant_diag"]["trap"].get("items") else 0.0
    false_comp = float(pairwise["trap_fail_given_stable_success"] / max(1, pairwise["stable_success_bundles"]))
    control_conditioned = float(pairwise["trap_fail_given_control_success"] / max(1, pairwise["control_success_bundles"]))
    stable_trap_success = 0.0 if pairwise["stable_success_bundles"] == 0 else max(0.0, 1.0 - false_comp)
    control_minus_trap_gap = float(control_acc - trap_acc)
    reliability_metrics = {
        "bundle_success_rate": float(bundle_success_mask.mean()) if not bundle_df.empty else 0.0,
        "trap_resistance_rate": trap_acc,
        "surface_consistency_rate": float(stable_success_mask.mean()) if not bundle_df.empty else 0.0,
        "false_competence_rate": false_comp,
        "control_minus_trap_gap": control_minus_trap_gap,
        "strict_gap_rate_by_heuristic": strict_gap_rate_by_heuristic,
        "strict_gap_rate_by_domain": strict_gap_rate_by_domain,
        "strict_gap_rate_by_difficulty": strict_gap_rate_by_difficulty,
        "trap_fail_given_control_success_rate": control_conditioned,
    }
    reliability_metrics["strict_slice_floor"] = min(
        reliability_metrics["strict_gap_rate_by_heuristic"],
        reliability_metrics["strict_gap_rate_by_domain"],
        reliability_metrics["strict_gap_rate_by_difficulty"],
    )
    reliability_metrics["global_trap_lower_pass"] = float(control_minus_trap_gap > 0.0)
    pressure_components = {
        "global_pass": reliability_metrics["global_trap_lower_pass"],
        "strict_slice_floor": reliability_metrics["strict_slice_floor"],
        "gap_strength": max(0.0, min(1.0, control_minus_trap_gap / 0.25)),
        "control_conditioned_strength": max(0.0, min(1.0, control_conditioned / 0.35)),
        "stable_conditioned_strength": max(0.0, min(1.0, false_comp / 0.35)),
        "overlong_specificity": max(0.0, min(1.0, 1.0 - max(0.0, (trap_overlong - control_overlong) - 0.03) / 0.10)),
    }
    reliability_metrics["observed_pressure_score"] = float(
        pressure_components["global_pass"]
        * pressure_components["strict_slice_floor"]
        * pressure_components["gap_strength"]
        * min(pressure_components["control_conditioned_strength"], pressure_components["stable_conditioned_strength"])
        * pressure_components["overlong_specificity"]
    )
    reliability_metrics["executive_reliability_score"] = max(
        0.0,
        reliability_metrics["bundle_success_rate"] * (1.0 - reliability_metrics["false_competence_rate"]),
    )
    reliability_metrics["weighted_geometric_robustness_score"] = _weighted_geometric_mean([
        (control_acc, 1.0),
        (trap_acc, 2.0),
        (paraphrase_acc, 1.0),
        (format_acc, 1.0),
        (stable_trap_success, 2.0),
    ])
    slice_deltas = {
        "paper_heuristic": gap_by_heuristic,
        "mechanism_family": _summarize_slice_deltas_wrapper(bundle_df, "mechanism_family"),
        "source_reliability": _summarize_slice_deltas_wrapper(bundle_df, "source_reliability"),
        "trap_anchor_type": _summarize_slice_deltas_wrapper(bundle_df, "trap_anchor_type"),
        "paper_heuristic_x_source_reliability": _summarize_slice_deltas_wrapper(bundle_df, ["paper_heuristic", "source_reliability"], max_cells=48),
        "paper_heuristic_x_difficulty": _summarize_slice_deltas_wrapper(bundle_df, ["paper_heuristic", "difficulty"], max_cells=48),
    }
    trap_rows = scored_df[scored_df["variant"] == "trap"]
    heuristic_capture = {}
    if not trap_rows.empty:
        for key, g in trap_rows.groupby("paper_heuristic"):
            heuristic_capture[str(key)] = {
                "bait_hit_rate": float(g["decoded"].apply(lambda d: 1 if d["bait_hit"] else 0).mean()),
                "response_has_bait_rate": float(g["decoded"].apply(lambda d: 1 if d["response_has_bait"] else 0).mean()),
                "response_has_terminal_checkpoint_rate": float(g["decoded"].apply(lambda d: 1 if d.get("response_has_terminal_checkpoint") else 0).mean()),
            }
    return {
        **base,
        "bundle_success": int(bundle_success_mask.sum()),
        "bundle_total": int(len(bundle_df)),
        "bpoa": float(bundle_success_mask.mean()) if not bundle_df.empty else 0.0,
        "pairwise": pairwise,
        "reliability_metrics": reliability_metrics,
        "pressure_components": pressure_components,
        "slice_deltas": slice_deltas,
        "heuristic_capture": heuristic_capture,
    }

def validate_dataset(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = build_dataset()
    problems = []
    scored_df = df[df["scored"] == True].copy()
    control_df = scored_df[scored_df["variant"] == "control"].copy()
    if int(scored_df["bundle_id"].nunique()) != 144:
        problems.append(f"bundle count mismatch: {scored_df['bundle_id'].nunique()} expected 144")
    for bundle_id, g in scored_df.groupby("bundle_id"):
        variants = set(g["variant"].tolist())
        if variants != set(EVALUATION_VARIANTS):
            problems.append(f"{bundle_id}: variants={sorted(variants)}")
            continue
        if g["correct_answer"].nunique() != 1:
            problems.append(f"{bundle_id}: multiple correct answers")
        if g["bait_answer"].nunique() != 1:
            problems.append(f"{bundle_id}: multiple bait answers")
        if g["correct_answer"].iloc[0] == g["bait_answer"].iloc[0]:
            problems.append(f"{bundle_id}: correct == bait")
        if int(g["bundle_prompt_range"].max()) > 1:
            problems.append(f"{bundle_id}: bundle prompt range > 1")
    heur_counts = control_df["paper_heuristic"].value_counts().to_dict() if not control_df.empty else {}
    for cell in HEURISTIC_SPECIFICATIONS:
        key = cell["paper_heuristic"]
        expected = int(cell["count_per_domain"]) * len(DOMAIN_NAMES)
        if int(heur_counts.get(key, 0)) != expected:
            problems.append(f"paper heuristic quota mismatch: {key}={heur_counts.get(key, 0)} expected {expected}")
    rel_counts = control_df["source_reliability"].value_counts().to_dict() if not control_df.empty else {}
    for rel, tgt in {"soft": 48, "strong": 48, "none": 48}.items():
        if int(rel_counts.get(rel, 0)) != tgt:
            problems.append(f"source reliability quota mismatch: {rel}={rel_counts.get(rel, 0)} expected {tgt}")
    style_counts = control_df["prior_style"].value_counts().to_dict() if not control_df.empty else {}
    for style in PRIOR_STYLE_NAMES:
        if int(style_counts.get(style, 0)) != 24:
            problems.append(f"prior style quota mismatch: {style}={style_counts.get(style, 0)} expected 24")
    dist_counts = control_df["terminal_distance_from_end"].value_counts().to_dict() if not control_df.empty else {}
    for dist in (1, 2, 3):
        if int(dist_counts.get(dist, 0)) <= 0:
            problems.append(f"missing late checkpoint distance {dist}")
    if int((~scored_df["prompt"].str.contains("Checked value: <integer>", regex=False)).sum()):
        problems.append("missing canonical closing")
    if int(scored_df["has_duplicate_closing"].sum()):
        problems.append("duplicate canonical closing")
    if int(scored_df["prompt_hash"].duplicated().sum()) != 0:
        problems.append("duplicate prompt hash")
    if int(control_df["chain_hash"].duplicated().sum()) != 0:
        problems.append("duplicate chain hash")
    return {
        "items": int(len(df)),
        "scored_items": int(len(scored_df)),
        "bundles": int(scored_df["bundle_id"].nunique()),
        "domains": int(scored_df["domain"].nunique()),
        "all_divergent": bool((scored_df["correct_answer"] != scored_df["bait_answer"]).all()),
        "duplicate_prompt_hashes": int(scored_df["prompt_hash"].duplicated().sum()) if "prompt_hash" in scored_df.columns else 0,
        "duplicate_chain_hashes": int(control_df[control_df["variant"] == "control"]["chain_hash"].duplicated().sum()) if "chain_hash" in control_df.columns else 0,
        "validation_errors": problems[:20],
        "validation_error_count": len(problems),
    }

def _compute_design_metrics(diag: dict) -> dict:
    pb = diag.get("prompt_tokens_by_variant", {})
    control_mean = float(pb.get("control", {}).get("mean", 0.0))
    trap_mean = float(pb.get("trap", {}).get("mean", 0.0))
    para_mean = float(pb.get("paraphrase_twin", {}).get("mean", 0.0))
    fmt_mean = float(pb.get("format_twin", {}).get("mean", 0.0))
    gap = abs(trap_mean - control_mean)
    twin_gap = max(abs(para_mean - control_mean), abs(fmt_mean - control_mean))
    heur_counts = diag.get("paper_heuristic_counts", {})
    dist_counts = diag.get("terminal_distance_from_end_counts", {})
    def norm_entropy(counts: dict) -> float:
        vals = [float(v) for v in counts.values() if float(v) > 0]
        if len(vals) <= 1:
            return 0.0
        total = sum(vals)
        probs = [v / total for v in vals]
        ent = -sum(p * math.log(p, 2) for p in probs if p > 0)
        return float(ent / math.log(len(vals), 2))
    design = {
        "trap_control_prompt_gap_abs": gap,
        "max_twin_prompt_gap_abs": twin_gap,
        "bundle_prompt_range_mean": float(diag.get("bundle_prompt_range", {}).get("mean", 0.0)),
        "bundle_prompt_range_max": float(diag.get("bundle_prompt_range", {}).get("max", 0.0)),
        "heuristic_breadth_score": norm_entropy(heur_counts),
        "distance_breadth_score": norm_entropy(dist_counts),
    }
    design["design_integrity_score"] = float(
        max(0.0, 1.0 - design["trap_control_prompt_gap_abs"] / 6.0)
        * max(0.0, 1.0 - design["max_twin_prompt_gap_abs"] / 3.0)
        * max(0.0, 1.0 - design["bundle_prompt_range_mean"] / 1.0)
        * design["heuristic_breadth_score"]
        * max(0.5, design["distance_breadth_score"])
    )
    return design

def run_preflight_checks(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = build_dataset()
    base = validate_dataset(df)
    diag = build_dataset_diagnostics(df)
    control_mean = diag["prompt_tokens_by_variant"].get("control", {}).get("mean", 0.0)
    trap_mean = diag["prompt_tokens_by_variant"].get("trap", {}).get("mean", 0.0)
    para_mean = diag["prompt_tokens_by_variant"].get("paraphrase_twin", {}).get("mean", 0.0)
    fmt_mean = diag["prompt_tokens_by_variant"].get("format_twin", {}).get("mean", 0.0)
    prompt_balance = {
        "trap_minus_control": float(trap_mean - control_mean),
        "paraphrase_twin_minus_control": float(para_mean - control_mean),
        "format_twin_minus_control": float(fmt_mean - control_mean),
    }
    parser_errors = _run_parser_self_tests()
    score_errors = _run_scoring_self_tests()
    design_metrics = _compute_design_metrics(diag)
    ok = (
        base.get("validation_error_count", 0) == 0
        and not parser_errors
        and not score_errors
        and diag.get("duplicate_closing_prompts", 0) == 0
        and abs(prompt_balance["trap_minus_control"]) <= 6.0
        and abs(prompt_balance["paraphrase_twin_minus_control"]) <= 3.0
        and abs(prompt_balance["format_twin_minus_control"]) <= 3.0
        and design_metrics["bundle_prompt_range_max"] <= 1.0
    )
    return {
        "ok": bool(ok),
        "dataset": base,
        "diagnostics": diag,
        "prompt_balance": prompt_balance,
        "parser_errors": parser_errors,
        "score_errors": score_errors,
        "anchor_discard_prompts": int(df[df["scored"] == True]["prompt"].apply(_prompt_discards_reference).sum()),
        "adaptive_family_weights": FAMILY_WEIGHTS,
        "design_metrics": design_metrics,
    }

@benchmark_task(name="Adversarial Prior Override")
def run_benchmark(llm) -> tuple[int, int]:
    df = build_dataset()
    preflight = run_preflight_checks(df)
    dataset_report = preflight["diagnostics"]
    print(json.dumps(preflight["dataset"], indent=2))
    print("Preflight:")
    print(json.dumps({k: v for k, v in preflight.items() if k != "diagnostics"}, indent=2))
    if not preflight["ok"]:
        raise RuntimeError("Benchmark preflight failed")
    print(f"Items: {len(df)}, Bundles: {df['bundle_id'].nunique()}, Domains: {df['domain'].nunique()}")
    print(f"Divergent: {(df['correct_answer'] != df['bait_answer']).sum()}/{len(df)}")
    print(f"Seed: {int(df['seed'].iloc[0])}")
    print(f"n_jobs: {PARALLEL_JOBS}")
    print(f"request_max_output_tokens: {MAX_OUTPUT_TOKENS}")
    print(f"soft_visible_output_tokens: {SOFT_TOKEN_LIMIT}")
    print(f"debug_log_jsonl: {DEBUG_LOG_FILE or '<off>'}")
    print("Dataset diagnostics:")
    print(json.dumps(dataset_report, indent=2))

    runs = benchmark_item.evaluate(llm=[llm], evaluation_data=df, n_jobs=PARALLEL_JOBS)
    results_df = runs.as_dataframe()
    summary = summarize_results(results_df)
    reliability_metrics = summary["reliability_metrics"]

    print("=" * 72)
    print("Executive Function: Research Reasoning Reliability")
    print("=" * 72)
    print(f"Weighted robustness score:   {reliability_metrics['weighted_geometric_robustness_score']:.4f}")
    print(f"Executive reliability score:  {reliability_metrics['executive_reliability_score']:.4f}")
    print(f"Observed pressure score:     {reliability_metrics['observed_pressure_score']:.4f}")
    print(f"Bundle success rate:         {reliability_metrics['bundle_success_rate']:.4f}")
    print(f"Trap resistance rate:        {reliability_metrics['trap_resistance_rate']:.4f}")
    print(f"Surface consistency rate:    {reliability_metrics['surface_consistency_rate']:.4f}")
    print(f"False competence rate:       {reliability_metrics['false_competence_rate']:.4f}")
    print(f"Control minus trap gap:      {reliability_metrics['control_minus_trap_gap']:.4f}")
    print(f"Strict gap by heuristic:     {reliability_metrics['strict_gap_rate_by_heuristic']:.4f}")
    print(f"Strict gap by domain:        {reliability_metrics['strict_gap_rate_by_domain']:.4f}")
    print(f"Strict gap by difficulty:    {reliability_metrics['strict_gap_rate_by_difficulty']:.4f}")
    print(f"Trap fail | control ok:      {summary['pairwise']['trap_fail_given_control_success']}/{max(1, summary['pairwise']['control_success_bundles'])}")
    print(f"Trap fail | stable success:  {summary['pairwise']['trap_fail_given_stable_success']}/{max(1, summary['pairwise']['stable_success_bundles'])}")
    print(f"Item accuracy:               {summary['item_acc']:.4f}")
    print(f"Bait-hit rate:               {summary['bait_hit_rate']:.4f}")
    print(f"Late-checkpoint capture rate:{summary['response_has_terminal_checkpoint_rate']:.4f}")
    print(f"Parse-fail rate:             {summary['parse_fail_rate']:.4f}")
    print(f"Strict-parser hit rate:      {summary['strict_found_rate']:.4f}")
    print(f"Recovery-parser usage:       {summary['used_recovery_rate']:.4f}")
    print(f"Trap>control bundles:        {summary['pairwise']['trap_better_than_control']}/{summary['bundle_total']}")
    print(f"Control>trap bundles:        {summary['pairwise']['control_better_than_trap']}/{summary['bundle_total']}")

    print("\nVariant breakdown:")
    for variant in EVALUATION_VARIANTS:
        s = summary["variant_stats"][variant]
        acc = s["correct"] / max(1, s["total"])
        d = summary["variant_diag"][variant]
        items = max(1, d["items"])
        print(f"  {variant:17s}: {s['correct']:3d}/{s['total']:3d} = {acc:.3f} | parse_fail={d['parse_fail']/items:.3f} overlong={d['overlong']/items:.3f} bait_hit={d['bait_hit']/items:.3f} resp_bait={d['response_has_bait']/items:.3f} late_checkpoint={d['response_has_terminal_checkpoint']/items:.3f} resp_correct={d['response_has_correct']/items:.3f}")

    print("\nPaper heuristic deltas (worst first):")
    for key, cell in summary["slice_deltas"]["paper_heuristic"].items():
        print(f"  {key:36s}: Δ={cell['trap_minus_control']:+.3f} | strict_lower={int(cell['trap_strictly_lower'])} | trap_fail|stable={cell['trap_fail_given_stable_success']:.3f} | n={cell['n_bundles']}")

    print("\nParser mode breakdown:")
    total_scored = int((results_df.get('scored', True) == True).sum())
    for k in sorted(summary["parser_mode_counts"]):
        v = summary["parser_mode_counts"][k]
        print(f"  {k:16s}: {v:3d}/{total_scored:3d} = {v/max(1,total_scored):.3f}")

    score = int(round(reliability_metrics["weighted_geometric_robustness_score"] * 1_000_000))
    return score, 1_000_000


if getattr(kbench, "llm", None) is not None and hasattr(run_benchmark, "run"):
    result = run_benchmark.run(llm=kbench.llm)