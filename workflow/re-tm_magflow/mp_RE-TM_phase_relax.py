#!/usr/bin/env python3
"""
MP Reference Phase VASP Relaxation Workflow

Identifies missing prototype-lanthanide combinations from the MP phase cache,
generates initial structures via element substitution from existing MP entries,
runs VASP relaxations on HPC, and computes DFT E_hull after completion.

Uses pure MPRelaxSet defaults (no MAGMOM override) to match MP's own DFT settings.

Usage:
    python3 mp_RE-TM_phase_relax.py --output-dir /scratch/$USER/mp_phase_relax
"""

import os
import sys
import json
import time
import types
import argparse
import warnings
import subprocess
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Shim: the legacy MPRester deserialises ComputedEntry objects whose
# @module may reference paths that moved between pymatgen versions.
# Map old paths to the modules that actually exist in this build.
# ---------------------------------------------------------------------------
import pymatgen.entries.computed_entries
import pymatgen.entries.compatibility
sys.modules.setdefault('pymatgen.core.entries',
                       pymatgen.entries.computed_entries)
sys.modules.setdefault('pymatgen.analysis.compatibility',
                       pymatgen.entries.compatibility)

from pymatgen.core import Structure, Composition, Element
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

try:
    from pyxtal import pyxtal
    PYXTAL_AVAILABLE = True
except ImportError:
    PYXTAL_AVAILABLE = False
    print("WARNING: PyXtal not available. Structures will not be symmetrized.")

try:
    from pymatgen.ext.matproj import MPRester
except ImportError:
    MPRester = None

try:
    import requests
except ImportError:
    requests = None

warnings.filterwarnings('ignore', category=UserWarning, message='.*POTCAR data with symbol.*')
warnings.filterwarnings('ignore', message='Using UFloat objects with std_dev==0')

BASE_URL = "https://api.materialsproject.org"


def _get_api_key():
    key = os.environ.get('MP_API_KEY')
    if not key:
        print("  WARNING: MP_API_KEY not set -- cannot query MP API")
    return key

LANTHANIDES = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
               'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']

TM_ELEMENTS = ['Fe', 'Co', 'Ni']

PROTOTYPES = {
    'RETM2_227':  {'ratio': 2.0, 'spg': 227, 'pearson': 'cF24',
                   're_count': 1, 'tm_count': 2},
    'RETM5_191':  {'ratio': 5.0, 'spg': 191, 'pearson': 'hP6',
                   're_count': 1, 'tm_count': 5},
    'R2TM7_166':  {'ratio': 3.5, 'spg': 166, 'pearson': 'hR18',
                   're_count': 2, 'tm_count': 7},
    'R2TM7_194':  {'ratio': 3.5, 'spg': 194, 'pearson': 'hP36',
                   're_count': 2, 'tm_count': 7},
    'R2TM17_166': {'ratio': 8.5, 'spg': 166, 'pearson': 'hR19',
                   're_count': 2, 'tm_count': 17},
    'R2TM17_194': {'ratio': 8.5, 'spg': 194, 'pearson': 'hP38',
                   're_count': 2, 'tm_count': 17},
}

LANTHANIDE_Z = {el: Element(el).Z for el in LANTHANIDES}


def _require_requests():
    if requests is None:
        raise RuntimeError("'requests' package required for MP API calls; "
                           "install it or use a conda env that has it")


def _get_with_retry(url, headers, params, max_retries=3, delay=2.0):
    _require_requests()
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                return resp
            print(f"  WARNING: API returned {resp.status_code}, retry {attempt+1}")
        except requests.exceptions.RequestException as e:
            print(f"  WARNING: request failed ({e}), retry {attempt+1}")
        time.sleep(delay * (attempt + 1))
    return None


def make_struct_id(re_elem, tm_elem, proto_info):
    """Build structure ID: {reduced_formula}_{pearson}_{spg}."""
    re_n = proto_info['re_count']
    tm_n = proto_info['tm_count']
    re_part = re_elem if re_n == 1 else f"{re_elem}{re_n}"
    tm_part = tm_elem if tm_n == 1 else f"{tm_elem}{tm_n}"
    return f"{re_part}{tm_part}_{proto_info['pearson']}_{proto_info['spg']}"


def identify_missing(cache):
    """Return list of (tm, proto_key, re) tuples absent from cache."""
    missing = []
    for tm in TM_ELEMENTS:
        tm_data = cache.get(tm, {})
        for pk in PROTOTYPES:
            proto_data = tm_data.get(pk, {})
            for re in LANTHANIDES:
                if re not in proto_data:
                    missing.append((tm, pk, re))
    return missing


def find_template(cache, tm, proto_key, re_target):
    """Find an existing cache entry for the same prototype.

    Search order:
      1. Same TM system, different RE (only RE substitution needed)
      2. Other TM systems (RE + TM substitution needed)

    Within each TM system, prefer the RE closest in atomic number.
    Returns (re_template, tm_template, material_id) or None.
    """
    target_z = LANTHANIDE_Z[re_target]

    # Priority order: same TM first, then others
    tm_order = [tm] + [t for t in TM_ELEMENTS if t != tm]

    for tm_search in tm_order:
        proto_data = cache.get(tm_search, {}).get(proto_key, {})
        if not proto_data:
            continue
        candidates = [(re, d['material_id']) for re, d in proto_data.items()
                      if not (re == re_target and tm_search == tm)]
        if not candidates:
            continue
        candidates.sort(key=lambda x: abs(LANTHANIDE_Z.get(x[0], 999) - target_z))
        return (candidates[0][0], tm_search, candidates[0][1])

    return None


def fetch_mp_structure(material_id):
    """Fetch crystal structure from MP summary endpoint."""
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("MP_API_KEY environment variable not set")
    headers = {"X-API-KEY": api_key}
    url = f"{BASE_URL}/materials/summary/"
    params = {
        "material_ids": material_id,
        "_fields": "material_id,structure",
    }
    resp = _get_with_retry(url, headers, params)
    if resp is None:
        return None
    data = resp.json().get('data', [])
    if not data:
        return None
    struct_dict = data[0].get('structure')
    if struct_dict is None:
        return None
    return Structure.from_dict(struct_dict)


def make_substituted_structure(template_struct, re_template, re_target,
                               tm_template=None, tm_target=None):
    """Replace re_template with re_target (and optionally tm_template with tm_target)."""
    new_struct = template_struct.copy()
    for i, site in enumerate(new_struct):
        sp = str(site.specie)
        if sp == re_template:
            new_struct.replace(i, re_target)
        elif tm_template and tm_target and sp == tm_template:
            new_struct.replace(i, tm_target)
    return new_struct


def symmetrize_structure(structure):
    """Symmetrize structure using PyXtal with progressive tolerance."""
    if not PYXTAL_AVAILABLE:
        return structure
    tolerances = [5e-2, 1e-2, 1e-3, 1e-4, 1e-5]
    adaptor = AseAtomsAdaptor()
    for tol in tolerances:
        try:
            xtal = pyxtal()
            xtal.from_seed(structure, tol=tol)
            if not xtal.valid:
                continue
            if len(xtal.check_short_distances(r=0.5)) > 0:
                continue
            atoms = xtal.to_ase()
            return adaptor.get_structure(atoms)
        except Exception:
            continue
    print(f"    Warning: Could not symmetrize, using original structure")
    return structure


def create_vasp_inputs(structure, job_dir):
    """Create VASP inputs using MPRelaxSet defaults with symmetrization
    and NBANDS safety for high-ZVAL lanthanide systems."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    structure = symmetrize_structure(structure)

    ncore = 4
    potcar = MPRelaxSet(structure).potcar
    zval = {p.element: p.ZVAL for p in potcar}
    nelect = sum(zval[str(el)] * amt for el, amt in structure.composition.items())
    nions = len(structure)
    nbands = max(int(nelect / 2) + max(nions // 2, 10), int(0.6 * nelect))
    nbands = ((nbands + ncore - 1) // ncore) * ncore

    vis = MPRelaxSet(structure,
                     user_incar_settings={
                         'NBANDS': nbands,
                         'NCORE': ncore,
                         'IBRION': 1,
                         'POTIM': 0.1,
                         'ISMEAR': 0,
                         'SIGMA': 0.05,
                         'NSW': 150,
                         'NELM': 120,
                         'SYMPREC': 1e-4,
                     })
    vis.write_input(job_dir)
    return job_dir


def create_slurm_script(job_dir, job_name):
    """Create SLURM submission script matching magflow_manager.py template."""
    job_dir = Path(job_dir).resolve()
    script_path = job_dir / 'job.sh'

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}_relax
#SBATCH --partition=Apus,Orion
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=str-c[85-97]
#SBATCH --output={job_dir}/vasp_%j.out
#SBATCH --error={job_dir}/vasp_%j.err

# Load modules
module purge
module load intel/mkl/2024.0 intel/2024 intel-mpi/2021.11
ulimit -s unlimited

# Set environment
export OMP_NUM_THREADS=1
export PMG_VASP_PSP_DIR=$HOME/apps/PBE52

# Intel MPI settings for SLURM
if [ -e /opt/slurm/lib/libpmi.so ]; then
  export I_MPI_PMI_LIBRARY=/opt/slurm/lib/libpmi.so
else
  export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so.0
fi
export I_MPI_FABRICS=shm:ofi

# VASP executable (use srun for SLURM-native MPI launching)
VASP_CMD="srun --mpi=pmi2 $HOME/apps/vasp.6.2.1/bin/vasp_std"

# Change to job directory
cd {job_dir}

# Run VASP
echo "Starting VASP MP-phase relaxation"
echo "Working directory: $(pwd)"
echo "VASP command: $VASP_CMD"
echo "Start time: $(date)"

$VASP_CMD

EXIT_CODE=$?

echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

# Check if successful
if [ $EXIT_CODE -eq 0 ]; then
    # Verify critical files for Relax calculation
    if [ -f "CONTCAR" ] && [ -s "CONTCAR" ]; then
        echo "VASP calculation completed successfully"
        echo "Verified CONTCAR exists"
        
        # Clean up large unnecessary files to save disk space
        rm -f CHGCAR CHG WAVECAR WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
        
        touch VASP_DONE
    else
        echo "VASP calculation failed: CONTCAR missing/empty"
        rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
        touch VASP_FAILED
    fi
else
    echo "VASP calculation failed with exit code $EXIT_CODE"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
fi
"""
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def check_electronic_convergence_oszicar(relax_dir):
    """Check electronic convergence from OSZICAR (timeout recovery)."""
    relax_dir = Path(relax_dir)
    oszicar_path = relax_dir / 'OSZICAR'
    incar_path = relax_dir / 'INCAR'

    if not oszicar_path.exists():
        return False, None

    nelm = 60
    if incar_path.exists():
        try:
            with open(incar_path, 'r') as f:
                for line in f:
                    if 'NELM' in line and '=' in line:
                        val = line.split('=')[1].split()[0].strip()
                        nelm = int(val)
                        break
        except Exception:
            pass

    try:
        with open(oszicar_path, 'r') as f:
            lines = [l.rstrip() for l in f.readlines() if l.strip()]

        if len(lines) < 2:
            return False, None

        search_from = len(lines) - 1
        while search_from > 0:
            f_idx = None
            for i in range(search_from, -1, -1):
                if 'F=' in lines[i]:
                    f_idx = i
                    break
            if f_idx is None or f_idx < 1:
                return False, None

            f_line = lines[f_idx]
            try:
                f_pos = f_line.index('F=')
                energy_str = f_line[f_pos + 2:].split()[0]
                total_energy = float(energy_str)
            except (ValueError, IndexError):
                search_from = f_idx - 1
                continue

            scf_line = lines[f_idx - 1]
            parts = scf_line.split()
            if len(parts) >= 2:
                try:
                    e_step = int(parts[1])
                    if e_step < nelm:
                        return True, total_energy
                except ValueError:
                    pass
            search_from = f_idx - 1
        return False, None
    except Exception:
        return False, None


def _alpha_to_mpid(alpha_id):
    """Convert AlphaID (e.g. 'mp-aaaaaaft') to numeric MPID (e.g. 'mp-149')."""
    prefix, code = alpha_id.rsplit('-', 1)
    num = 0
    for ch in code:
        num = num * 26 + (ord(ch) - ord('a'))
    return f"{prefix}-{num}"


def _fetch_gga_elemental_from_tasks(element):
    """Fallback: fetch the lowest GGA energy for an element from MP tasks API.

    Some elements (e.g. Yb) have no GGA thermo entry but DO have GGA
    tasks (Structure Optimization / Static).  Returns (energy_per_atom,
    task_id) or (None, None).
    """
    _require_requests()
    api_key = _get_api_key()
    if not api_key:
        return None, None
    headers = {"X-API-KEY": api_key}
    url = f"{BASE_URL}/materials/tasks/"
    params = {
        "formula": element,
        "_fields": "task_id,task_type,run_type,output",
        "_limit": 100,
    }
    resp = _get_with_retry(url, headers, params)
    if resp is None:
        return None, None

    best_epa, best_tid = None, None
    for d in resp.json().get("data", []):
        if d.get("run_type") != "GGA":
            continue
        ttype = d.get("task_type", "")
        if ttype not in ("Structure Optimization", "Static"):
            continue
        output = d.get("output") or {}
        epa = output.get("energy_per_atom")
        if epa is None:
            continue
        if best_epa is None or epa < best_epa:
            best_epa = epa
            best_tid = d.get("task_id", "task-unknown")
    return best_epa, best_tid


def query_mp_entries_for_chemsys(chemsys):
    """Query MP for GGA/GGA+U entries using legacy MPRester.

    Uses pymatgen.ext.matproj.MPRester.get_entries_in_chemsys() which
    returns ComputedEntry objects for ALL phases (including elemental
    endpoints).  Filters for '-GGA' or '-GGA+U' entry_id suffix and
    uses uncorrected_energy (raw DFT, no anion/composition corrections)
    to match our VASP energies.

    If any elemental endpoint is missing from the thermo entries (e.g. Yb),
    falls back to querying the tasks API for the lowest GGA task energy.
    """
    api_key = _get_api_key()
    if not api_key:
        return []
    if MPRester is None:
        print("  WARNING: pymatgen.ext.matproj.MPRester not available")
        return []

    elements = chemsys.split('-')
    try:
        mpr = MPRester(api_key)
        computed_entries = mpr.get_entries_in_chemsys(elements)
    except Exception as e:
        print(f"  WARNING: MPRester query failed for {chemsys}: {e}")
        return []

    pd_entries = []
    seen = set()
    for comp_entry in computed_entries:
        raw_eid = comp_entry.entry_id
        if isinstance(raw_eid, dict):
            mp_id = _alpha_to_mpid(raw_eid.get('identifier', 'mp-a'))
            suffix = raw_eid.get('suffix', '')
            sep = raw_eid.get('separator', '-')
            entry_id = f"{mp_id}{sep}{suffix}"
        else:
            entry_id = str(raw_eid)
            suffix = entry_id.rsplit('-', 1)[-1] if '-' in entry_id else ''

        if entry_id in seen:
            continue
        if suffix not in ('GGA', 'GGA+U'):
            continue
        seen.add(entry_id)

        try:
            energy = comp_entry.uncorrected_energy
        except AttributeError:
            energy = comp_entry.energy

        pd_entries.append(PDEntry(comp_entry.composition, energy,
                                  name=entry_id))

    # Check for missing elemental endpoints and try tasks API fallback
    present_elems = set()
    for e in pd_entries:
        if len(e.composition.elements) == 1:
            present_elems.add(str(e.composition.elements[0]))

    for el in elements:
        if el in present_elems:
            continue
        print(f"  WARNING: no GGA thermo entry for {el}, "
              f"trying tasks API fallback...")
        epa, tid = _fetch_gga_elemental_from_tasks(el)
        if epa is not None:
            comp = Composition({el: 1})
            pd_entries.append(PDEntry(comp, epa, name=f"{tid}-GGA-task"))
            print(f"    Recovered {el} elemental from task {tid}: "
                  f"{epa:.6f} eV/atom")
        else:
            print(f"    FAILED: no GGA tasks found for {el} either")

    return pd_entries


def compute_dft_ehull(chemsys, vasp_energy_per_atom, composition_dict):
    """Compute E_hull for a relaxed structure against MP reference phases."""
    mp_entries = query_mp_entries_for_chemsys(chemsys)
    if not mp_entries:
        print(f"  WARNING: no MP entries returned for {chemsys}")
        return None

    comp = Composition(composition_dict)
    total_e = vasp_energy_per_atom * comp.num_atoms
    our_entry = PDEntry(comp, total_e, name='relaxed')

    all_entries = mp_entries + [our_entry]

    elem_symbols = set()
    for e in all_entries:
        if len(e.composition.elements) == 1:
            elem_symbols.add(str(e.composition.elements[0]))
    needed = set(str(el) for el in comp.elements)
    missing = needed - elem_symbols
    if missing:
        print(f"  WARNING: cannot build PhaseDiagram for {chemsys} -- "
              f"missing elemental reference(s): {sorted(missing)}")
        return None

    try:
        pd = PhaseDiagram(all_entries)
        return pd.get_e_above_hull(our_entry)
    except Exception as e:
        print(f"  WARNING: PhaseDiagram failed for {chemsys}: {e}")
        return None


class WorkflowDatabase:
    """Simple JSON-based database for tracking MP phase relaxation jobs."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.data = {'structures': {}, 'config': {}}
        self.load()

    def load(self):
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                self.data = json.load(f)

    def save(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.db_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        tmp_path.replace(self.db_path)

    def add_structure(self, struct_id, record):
        self.data['structures'][struct_id] = record
        self.save()

    def update_state(self, struct_id, state, **kwargs):
        if struct_id in self.data['structures']:
            self.data['structures'][struct_id]['state'] = state
            self.data['structures'][struct_id]['last_updated'] = \
                datetime.now().isoformat()
            for key, value in kwargs.items():
                self.data['structures'][struct_id][key] = value
            self.save()

    def get_structure(self, struct_id):
        return self.data['structures'].get(struct_id)

    def get_by_state(self, state):
        return [sid for sid, s in self.data['structures'].items()
                if s['state'] == state]

    def get_running_count(self):
        return sum(1 for s in self.data['structures'].values()
                   if s['state'] == 'RUNNING')

    def get_stats(self):
        states = {}
        for s in self.data['structures'].values():
            states[s['state']] = states.get(s['state'], 0) + 1
        return {
            'total': len(self.data['structures']),
            'states': states,
            'running': self.get_running_count()
        }


class MPPhaseRelaxManager:
    """Manages VASP relaxation of missing MP reference phase prototypes."""

    def __init__(self, db_path, max_concurrent=10, check_interval=60):
        self.db = WorkflowDatabase(db_path)
        self.max_concurrent = max_concurrent
        self.check_interval = check_interval
        self._structure_cache = {}

    def initialize(self, cache, output_dir):
        """Identify missing combos, fetch templates, populate DB."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        missing = identify_missing(cache)
        print(f"Total missing prototype-lanthanide-TM combinations: {len(missing)}")

        skipped_no_template = 0
        skipped_existing = 0
        added = 0

        template_struct_cache = {}

        for tm, pk, re in missing:
            pinfo = PROTOTYPES[pk]
            struct_id = make_struct_id(re, tm, pinfo)

            if self.db.get_structure(struct_id):
                skipped_existing += 1
                continue

            tmpl = find_template(cache, tm, pk, re)
            if tmpl is None:
                skipped_no_template += 1
                print(f"  SKIP {struct_id}: no template available for "
                      f"{pk} in any TM system")
                continue

            re_template, tm_template, template_mpid = tmpl
            if tm_template != tm:
                print(f"  CROSS-TM {struct_id}: using {tm_template} template "
                      f"({re_template}{tm_template} -> {re}{tm})")
            chemsys = '-'.join(sorted([re, tm]))
            relax_dir = str(output_dir / struct_id / 'Relax')

            self.db.add_structure(struct_id, {
                'tm': tm,
                'proto_key': pk,
                're': re,
                're_template': re_template,
                'tm_template': tm_template,
                'template_mpid': template_mpid,
                'chemsys': chemsys,
                'spg': pinfo['spg'],
                'state': 'PENDING',
                'slurm_job_id': None,
                'relax_dir': relax_dir,
                'vasp_energy_per_atom': None,
                'dft_e_hull': None,
                'error': None,
                'last_updated': datetime.now().isoformat(),
            })
            added += 1

        self.db.data['config'] = {
            'max_concurrent': self.max_concurrent,
            'output_dir': str(output_dir),
        }
        self.db.save()

        print(f"\nInitialization summary:")
        print(f"  Added to DB: {added}")
        print(f"  Already in DB: {skipped_existing}")
        print(f"  Skipped (no template): {skipped_no_template}")
        print(f"  Total in DB: {len(self.db.data['structures'])}")

    def _get_structure(self, struct_id):
        """Fetch and cache the substituted structure for a given entry."""
        if struct_id in self._structure_cache:
            return self._structure_cache[struct_id]

        sdata = self.db.get_structure(struct_id)
        if not sdata:
            return None

        template_mpid = sdata['template_mpid']
        re_template = sdata['re_template']
        re_target = sdata['re']
        tm_template = sdata.get('tm_template', sdata['tm'])
        tm_target = sdata['tm']

        if tm_template != tm_target:
            print(f"    Fetching template {template_mpid} "
                  f"({re_template}{tm_template} -> {re_target}{tm_target})...")
        else:
            print(f"    Fetching template {template_mpid} "
                  f"({re_template} -> {re_target})...")
        template_struct = fetch_mp_structure(template_mpid)
        if template_struct is None:
            return None

        structure = make_substituted_structure(
            template_struct, re_template, re_target,
            tm_template if tm_template != tm_target else None,
            tm_target if tm_template != tm_target else None)
        self._structure_cache[struct_id] = structure
        return structure

    def submit_relax(self, struct_id):
        """Submit relaxation job for a structure."""
        sdata = self.db.get_structure(struct_id)
        if not sdata:
            return False

        relax_dir = Path(sdata['relax_dir'])
        print(f"  Submitting: {struct_id}")

        try:
            structure = self._get_structure(struct_id)
            if structure is None:
                raise RuntimeError("Could not fetch/substitute template structure")

            create_vasp_inputs(structure, relax_dir)
            script = create_slurm_script(relax_dir, struct_id)
            job_id = self._submit_job(script)

            self.db.update_state(struct_id, 'RUNNING', slurm_job_id=job_id)
            print(f"    Job ID: {job_id}")
            return True
        except Exception as e:
            print(f"    Error: {e}")
            self.db.update_state(struct_id, 'FAILED', error=str(e))
            return False

    def _submit_job(self, script_path):
        result = subprocess.run(
            ['sbatch', str(script_path)],
            capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split()[-1]
        raise RuntimeError(f"sbatch failed: {result.stderr}")

    def _check_job_status(self, job_id):
        result = subprocess.run(
            ['squeue', '-j', job_id, '-h', '-o', '%T'],
            capture_output=True, text=True)
        if result.stdout.strip():
            return 'RUNNING'
        return 'NOTFOUND'

    def _check_local_status(self, job_dir):
        job_dir = Path(job_dir)
        if (job_dir / 'VASP_DONE').exists():
            return 'DONE'
        if (job_dir / 'VASP_FAILED').exists():
            return 'FAILED'
        return 'UNKNOWN'

    def _parse_vasp_results(self, struct_id):
        """Parse relaxed energy and compute E_hull for a completed structure."""
        sdata = self.db.get_structure(struct_id)
        relax_dir = Path(sdata['relax_dir'])
        vasprun_path = relax_dir / 'vasprun.xml'

        try:
            vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
        except Exception as e:
            return None, None, f"Could not parse vasprun.xml: {e}"

        if not vr.converged_electronic:
            return None, None, "Electronic SCF not converged"

        energy_per_atom = vr.final_energy / len(vr.final_structure)
        comp_dict = dict(vr.final_structure.composition.as_dict())

        print(f"    Energy: {energy_per_atom:.6f} eV/atom, "
              f"computing E_hull...")
        time.sleep(0.5)
        e_hull = compute_dft_ehull(sdata['chemsys'], energy_per_atom, comp_dict)
        if e_hull is not None:
            print(f"    E_hull: {e_hull:.6f} eV/atom")
        else:
            print(f"    E_hull: could not compute")

        return energy_per_atom, e_hull, None

    def update_structure_status(self, struct_id):
        """Check and update status of a structure."""
        sdata = self.db.get_structure(struct_id)
        if not sdata or sdata['state'] != 'RUNNING':
            return

        job_status = self._check_job_status(sdata['slurm_job_id'])
        if job_status != 'NOTFOUND':
            return

        relax_dir = Path(sdata['relax_dir'])
        local_status = self._check_local_status(relax_dir)

        if local_status == 'DONE':
            epa, e_hull, err = self._parse_vasp_results(struct_id)
            if err:
                self.db.update_state(struct_id, 'FAILED', error=err)
                print(f"  {struct_id}: FAILED ({err})")
            else:
                self.db.update_state(struct_id, 'DONE',
                                     vasp_energy_per_atom=epa,
                                     dft_e_hull=e_hull)
                print(f"  {struct_id}: DONE")

        elif local_status == 'FAILED':
            self.db.update_state(struct_id, 'FAILED')
            print(f"  {struct_id}: FAILED")

        else:
            err_files = list(relax_dir.glob('vasp_*.err'))
            is_timeout = False
            if err_files:
                err_file = max(err_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(err_file, 'r') as f:
                        if 'DUE TO TIME LIMIT' in f.read():
                            is_timeout = True
                except Exception:
                    pass

            if is_timeout:
                contcar_path = relax_dir / 'CONTCAR'
                if not contcar_path.exists() or contcar_path.stat().st_size == 0:
                    self.db.update_state(struct_id, 'FAILED',
                                         error='Timed out, CONTCAR missing')
                    print(f"  {struct_id}: FAILED (timeout, no CONTCAR)")
                else:
                    converged, energy = \
                        check_electronic_convergence_oszicar(relax_dir)
                    if converged and energy is not None:
                        natoms = len(Structure.from_file(
                            str(contcar_path)))
                        epa = energy / natoms
                        comp = Structure.from_file(
                            str(contcar_path)).composition.as_dict()
                        e_hull = compute_dft_ehull(
                            sdata['chemsys'], epa, comp)
                        self.db.update_state(
                            struct_id, 'TMOUT',
                            vasp_energy_per_atom=epa,
                            dft_e_hull=e_hull,
                            error='Timed out but electronic converged')
                        print(f"  {struct_id}: TMOUT (usable)")
                    else:
                        self.db.update_state(struct_id, 'FAILED',
                                             error='Timed out, not converged')
                        print(f"  {struct_id}: FAILED (timeout, not converged)")
            else:
                self.db.update_state(struct_id, 'FAILED',
                                     error='Job terminated without marker')
                print(f"  {struct_id}: FAILED (crash)")

    def monitor_and_submit(self):
        """Main monitoring loop."""
        print("\n" + "=" * 70)
        print("Starting MP phase relaxation monitoring loop...")
        print(f"Max concurrent: {self.max_concurrent}")
        print(f"Check interval: {self.check_interval}s")
        print("=" * 70 + "\n")
        sys.stdout.flush()

        while True:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{ts}] Checking job status...")
            sys.stdout.flush()

            for sid in list(self.db.data['structures'].keys()):
                self.update_structure_status(sid)

            running_count = self.db.get_running_count()
            print(f"Currently running: {running_count}/{self.max_concurrent}")

            for sid in list(self.db.data['structures'].keys()):
                if running_count >= self.max_concurrent:
                    break
                sdata = self.db.get_structure(sid)
                if sdata['state'] == 'PENDING':
                    if self.submit_relax(sid):
                        running_count += 1

            stats = self.db.get_stats()
            print("\nStatistics:")
            for state, count in sorted(stats['states'].items()):
                print(f"  {state}: {count}")
            sys.stdout.flush()

            pending = len(self.db.get_by_state('PENDING'))
            if running_count == 0 and pending == 0:
                done = len(self.db.get_by_state('DONE'))
                tmout = len(self.db.get_by_state('TMOUT'))
                failed = len(self.db.get_by_state('FAILED'))
                total = stats['total']
                if done + tmout + failed >= total:
                    print("\n" + "=" * 70)
                    print("All relaxations completed!")
                    print(f"  Done: {done}/{total}")
                    print(f"  Timed out (usable): {tmout}/{total}")
                    print(f"  Failed: {failed}/{total}")
                    print("=" * 70)
                    sys.stdout.flush()
                    break

            print(f"\nSleeping for {self.check_interval}s...")
            sys.stdout.flush()
            time.sleep(self.check_interval)


def _compute_ehull_postprocess(db_path):
    """Post-process: compute dft_e_hull for completed entries missing it.
    Run from a login node with internet access.
    """
    api_key = _get_api_key()
    if not api_key:
        print("ERROR: MP_API_KEY must be set for E_hull computation")
        sys.exit(1)

    db = WorkflowDatabase(db_path)
    targets = []
    for sid, rec in db.data['structures'].items():
        if rec['state'] in ('DONE', 'TMOUT') and \
                rec.get('vasp_energy_per_atom') is not None and \
                rec.get('dft_e_hull') is None:
            targets.append(sid)

    if not targets:
        print("No entries need E_hull computation.")
        return

    print(f"Computing dft_e_hull for {len(targets)} entries...")
    success = 0
    for i, sid in enumerate(targets, 1):
        rec = db.get_structure(sid)
        chemsys = rec['chemsys']
        epa = rec['vasp_energy_per_atom']

        relax_dir = Path(rec['relax_dir'])
        contcar = relax_dir / 'CONTCAR'
        vasprun_path = relax_dir / 'vasprun.xml'

        comp_dict = None
        if vasprun_path.exists():
            try:
                vr = Vasprun(str(vasprun_path), parse_dos=False,
                             parse_eigen=False)
                comp_dict = dict(vr.final_structure.composition.as_dict())
            except Exception:
                pass
        if comp_dict is None and contcar.exists():
            try:
                comp_dict = dict(
                    Structure.from_file(str(contcar)).composition.as_dict())
            except Exception:
                pass
        if comp_dict is None:
            re, tm = rec['re'], rec['tm']
            pinfo = PROTOTYPES[rec['proto_key']]
            comp_dict = {re: pinfo['re_count'], tm: pinfo['tm_count']}

        print(f"  [{i}/{len(targets)}] {sid} ({chemsys}, "
              f"epa={epa:.6f})...", end=" ")
        e_hull = compute_dft_ehull(chemsys, epa, comp_dict)
        if e_hull is not None:
            db.update_state(sid, rec['state'], dft_e_hull=e_hull)
            print(f"E_hull = {e_hull:.6f} eV/atom")
            success += 1
        else:
            print("FAILED")
        time.sleep(0.3)

    print(f"\nDone: {success}/{len(targets)} E_hull values computed.")


def main():
    parser = argparse.ArgumentParser(
        description="MP Reference Phase VASP Relaxation Workflow")
    parser.add_argument(
        '--cache', type=str, default='mp_RE-TM_phase_cache.json',
        help="Path to MP phase cache JSON")
    parser.add_argument(
        '--output-dir', type=str, default='/scratch/$USER/mp_phase_relax',
        help="Output directory for VASP jobs")
    parser.add_argument(
        '--db', type=str, default='mp_RE-TM_phase_flow.json',
        help="JSON database file path")
    parser.add_argument(
        '--max-concurrent', type=int, default=10,
        help="Max concurrent VASP jobs")
    parser.add_argument(
        '--check-interval', type=int, default=60,
        help="Status check interval in seconds")
    parser.add_argument(
        '--init-only', action='store_true',
        help="Only initialize database, don't start monitoring")
    parser.add_argument(
        '--compute-ehull', action='store_true',
        help="Post-processing: compute dft_e_hull for all DONE/TMOUT entries "
             "that have vasp_energy_per_atom but no dft_e_hull. "
             "Run from a login node with internet access.")

    args = parser.parse_args()

    output_dir = Path(os.path.expandvars(args.output_dir)).expanduser()
    db_path = Path(args.db).expanduser()
    if not db_path.is_absolute():
        db_path = output_dir / args.db

    if args.compute_ehull:
        _compute_ehull_postprocess(db_path)
        return

    cache_path = Path(args.cache).expanduser()
    if not cache_path.exists():
        print(f"ERROR: Cache file not found: {cache_path}")
        sys.exit(1)
    with open(cache_path) as f:
        cache = json.load(f)

    manager = MPPhaseRelaxManager(
        db_path=db_path,
        max_concurrent=args.max_concurrent,
        check_interval=args.check_interval)

    manager.initialize(cache, output_dir)

    if args.init_only:
        print("\n" + "=" * 70)
        print("Initialization complete!")
        print(f"Database: {db_path}")
        print("=" * 70)
        return

    try:
        manager.monitor_and_submit()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        print(f"Database saved to: {db_path}")
        print("Resume with same command to continue.")


if __name__ == '__main__':
    main()
