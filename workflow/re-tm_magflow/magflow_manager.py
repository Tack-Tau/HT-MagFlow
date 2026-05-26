#!/usr/bin/env python3
"""
Binary Magnets Coarse Relaxation Workflow - Batch submission with dynamic monitoring
No MongoDB/FireWorks - uses local JSON database for job tracking

Features:
- Single-step coarse VASP relaxation only (no subsequent SPE workflow)
- Dynamic job submission with concurrency control

Note: Pre-screening should be done separately with prescreen.py before running this workflow.
"""

import os
import sys
import json
import time
import argparse
import zipfile
import warnings
import subprocess
import shutil
from pathlib import Path
from io import StringIO
from datetime import datetime

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from pyxtal import pyxtal
    PYXTAL_AVAILABLE = True
except ImportError:
    PYXTAL_AVAILABLE = False
    print("WARNING: PyXtal not available. Structures will not be symmetrized.")

warnings.filterwarnings('ignore', category=UserWarning, message='.*POTCAR data with symbol.*')
warnings.filterwarnings('ignore', message='Using UFloat objects with std_dev==0')



# MAGMOM overrides: antiparallel RE-TM ferrimagnetic convention.
# RE = negative (4f moments), late 3d TM = positive, early 3d TM = small negative.
MAGMOM_OVERRIDE = {
    'Ce': -1.0,
    'Pr': -2.0, 'Nd': -3.0, 'Pm': -4.0, 'Sm': -5.0, 'Eu': -7.0,
    'Gd': -7.0, 'Tb': -6.0, 'Dy': -5.0, 'Ho': -4.0,
    'Er': -3.0, 'Tm': -2.0, 'Yb': -1.0,
    'Ti': -0.5, 'V': -1.0, 'Cr': -2.0,
    'Mn': 0.5,
    'Fe': 2.2, 'Co': 2.0, 'Ni': 2.0,
}


def build_magmom(structure):
    """Build MAGMOM dict for RE-TM ferrimagnetic initial guess.

    When overriding via user_incar_settings, pymatgen does NOT merge with its
    config defaults, so all elements must be included in the returned dict.
    Returns dict (element -> value) if any override is needed,
    None otherwise (use pymatgen defaults).
    """
    elements = [str(el) for el in structure.composition.elements]
    if not any(el in MAGMOM_OVERRIDE for el in elements):
        return None
    result = {}
    for el in elements:
        if el in MAGMOM_OVERRIDE:
            result[el] = MAGMOM_OVERRIDE[el]
        else:
            result[el] = 0.6
    return result


def check_electronic_convergence_oszicar(relax_dir):
    """
    Check electronic convergence from OSZICAR and return energy.

    Searches backwards through ionic steps (F= lines) to find the most recent
    one with converged electronic SCF (iteration count < NELM). Returns the
    energy from that step for reliable timeout recovery.

    Args:
        relax_dir: Path to VASP relaxation directory (contains OSZICAR and INCAR)

    Returns:
        tuple: (converged: bool, total_energy: float or None)
    """
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


class WorkflowDatabase:
    """Simple JSON-based database for tracking job states."""
    
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.data = {'structures': {}, 'config': {}}
        self.load()
    
    def load(self):
        """Load database from JSON file."""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                self.data = json.load(f)
    
    def save(self):
        """Save database to JSON file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.db_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        tmp_path.replace(self.db_path)
    
    def add_structure(self, struct_id, comp_name, struct_idx, base_dir, chemsys=None):
        """Add a new structure to track."""
        self.data['structures'][struct_id] = {
            'composition': comp_name,
            'chemsys': chemsys,
            'structure_idx': struct_idx,
            'state': 'PENDING',
            'relax_job_id': None,
            'relax_dir': str(base_dir / struct_id / 'Relax'),
            'last_updated': datetime.now().isoformat(),
            'error': None
        }
        self.save()
    
    def update_state(self, struct_id, state, **kwargs):
        """Update structure state and additional fields."""
        if struct_id in self.data['structures']:
            self.data['structures'][struct_id]['state'] = state
            self.data['structures'][struct_id]['last_updated'] = datetime.now().isoformat()
            for key, value in kwargs.items():
                self.data['structures'][struct_id][key] = value
            self.save()
    
    def get_structure(self, struct_id):
        """Get structure data."""
        return self.data['structures'].get(struct_id)
    
    def get_by_state(self, state):
        """Get all structures in a specific state."""
        return [sid for sid, sdata in self.data['structures'].items() 
                if sdata['state'] == state]
    
    def get_running_count(self):
        """Count structures currently running."""
        return sum(1 for s in self.data['structures'].values() 
                   if s['state'] == 'RELAX_RUNNING')
    
    def get_stats(self):
        """Get overall statistics."""
        states = {}
        for s in self.data['structures'].values():
            state = s['state']
            states[state] = states.get(state, 0) + 1
        return {
            'total': len(self.data['structures']),
            'states': states,
            'running': self.get_running_count()
        }


class VASPRelaxManager:
    """Manages VASP coarse relaxation job submission and monitoring."""
    
    def __init__(self, db_path, max_concurrent=10, check_interval=60):
        self.db = WorkflowDatabase(db_path)
        self.max_concurrent = max_concurrent
        self.check_interval = check_interval
    
    def read_structures_from_zip(self, zip_path, max_structures=None):
        """Read CIF structures from zip file."""
        structures = []
        with zipfile.ZipFile(zip_path, 'r') as zf:
            cif_files = sorted([f for f in zf.namelist() if f.endswith('.cif')])
            if max_structures:
                cif_files = cif_files[:max_structures]
            
            for cif_file in cif_files:
                try:
                    with zf.open(cif_file) as f:
                        cif_content = f.read().decode('utf-8')
                        parser = CifParser(StringIO(cif_content))
                        structure = parser.parse_structures(primitive=True)[0]
                        structures.append(structure)
                except Exception as e:
                    print(f"  Warning: Could not parse {cif_file}: {e}")
        return structures
    
    def create_vasp_inputs(self, structure, job_dir):
        """
        Create VASP input files for coarse relaxation using pymatgen.
        
        Note: Structure is symmetrized using PyXtal before VASP input generation.
        """
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Symmetrize structure using PyXtal with progressive tolerance
        if PYXTAL_AVAILABLE:
            tolerances = [5e-2, 1e-2, 1e-3, 1e-4, 1e-5]
            symmetrized = False
            for tol in tolerances:
                try:
                    adaptor = AseAtomsAdaptor()
                    xtal = pyxtal()
                    xtal.from_seed(structure, tol=tol)
                    if not xtal.valid:
                        continue
                    if len(xtal.check_short_distances(r=0.5)) > 0:
                        continue
                    atoms = xtal.to_ase()
                    structure = adaptor.get_structure(atoms)
                    symmetrized = True
                    break
                except Exception:
                    continue
            
            if not symmetrized:
                print(f"    Warning: Could not symmetrize structure with tolerances {tolerances}")
                print(f"    Proceeding with original structure...")
        
        incar_settings = {
            'PREC': 'Normal',
            'ALGO': 'Normal',
            'ADDGRID': True,
            'EDIFF': 1e-4,
            'EDIFFG': -0.01,
            'IBRION': 1,
            'ISIF': 3,
            'NELM': 120,
            'NSW': 100,
            'ISMEAR': 1,
            'SIGMA': 0.05,
            'ISPIN': 2,
            'POTIM': 0.2,
            'LREAL': 'Auto',
            'LWAVE': False,
            'LCHARG': False,
            'LAECHG': False,
            'LASPH': True,
            'LORBIT': 11,
            'NCORE': 4,
            'SYMPREC': 1e-5,
        }
        
        # Override MAGMOM for rare earths (pymatgen defaults to 0.6 for most RE)
        magmom = build_magmom(structure)
        if magmom is not None:
            incar_settings['MAGMOM'] = magmom
        
        vis = MPRelaxSet(structure, 
            user_incar_settings=incar_settings,
            user_kpoints_settings={'reciprocal_density': 64}
        )
        
        vis.write_input(job_dir)
        return job_dir
    
    def create_slurm_script(self, job_dir, job_name):
        """Create SLURM submission script for coarse relaxation."""
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
echo "Starting VASP coarse relaxation"
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
        # Clean up large intermediate files to save disk space
        rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
        touch VASP_FAILED
    fi
else
    echo "VASP calculation failed with exit code $EXIT_CODE"
    # Clean up large intermediate files to save disk space
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
fi
"""
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def submit_job(self, script_path):
        """Submit a SLURM job and return job ID."""
        result = subprocess.run(
            ['sbatch', str(script_path)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            return job_id
        else:
            raise RuntimeError(f"sbatch failed: {result.stderr}")
    
    def check_job_status(self, job_id):
        """Check SLURM job status. Returns: RUNNING, COMPLETED, FAILED, or NOTFOUND."""
        result = subprocess.run(
            ['squeue', '-j', job_id, '-h', '-o', '%T'],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            slurm_state = result.stdout.strip()
            if slurm_state in ['RUNNING', 'PENDING', 'CONFIGURING']:
                return 'RUNNING'
            else:
                return 'RUNNING'  # Other states still in queue
        else:
            return 'NOTFOUND'
    
    def check_local_status(self, job_dir):
        """Check local directory for completion markers."""
        job_dir = Path(job_dir)
        if (job_dir / 'VASP_DONE').exists():
            return 'DONE'
        elif (job_dir / 'VASP_FAILED').exists():
            return 'FAILED'
        else:
            return 'UNKNOWN'
    
    def submit_relax(self, struct_id, structure):
        """Submit relaxation job for a structure."""
        sdata = self.db.get_structure(struct_id)
        if not sdata:
            return False
        
        relax_dir = Path(sdata['relax_dir'])
        job_name = struct_id
        
        print(f"  Submitting Relax: {struct_id}")
        
        try:
            self.create_vasp_inputs(structure, relax_dir)
            script = self.create_slurm_script(relax_dir, job_name)
            job_id = self.submit_job(script)
            
            self.db.update_state(struct_id, 'RELAX_RUNNING', relax_job_id=job_id)
            print(f"    Relax job ID: {job_id}")
            return True
        except Exception as e:
            print(f"    Error: {e}")
            self.db.update_state(struct_id, 'RELAX_FAILED', error=str(e))
            return False
    
    def update_structure_status(self, struct_id):
        """Check and update status of a structure."""
        sdata = self.db.get_structure(struct_id)
        if not sdata:
            return
        
        state = sdata['state']
        
        if state != 'RELAX_RUNNING':
            return
        
        job_status = self.check_job_status(sdata['relax_job_id'])
        if job_status != 'NOTFOUND':
            return
        
        local_status = self.check_local_status(sdata['relax_dir'])
        if local_status == 'DONE':
            # Check convergence before marking as done
            vasprun_path = Path(sdata['relax_dir']) / 'vasprun.xml'
            if vasprun_path.exists():
                try:
                    vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
                    if not vr.converged_electronic:
                        self.db.update_state(struct_id, 'RELAX_FAILED', 
                                           error='Electronic SCF not converged')
                        print(f"  {struct_id}: Relax FAILED (electronic not converged)")
                    else:
                        self.db.update_state(struct_id, 'RELAX_DONE')
                        print(f"  {struct_id}: Relax completed (electronic converged)")
                except Exception as e:
                    self.db.update_state(struct_id, 'RELAX_FAILED', 
                                       error=f'Could not check convergence: {e}')
                    print(f"  {struct_id}: Relax FAILED (convergence check error)")
            else:
                self.db.update_state(struct_id, 'RELAX_FAILED', 
                                   error='vasprun.xml not found')
                print(f"  {struct_id}: Relax FAILED (vasprun.xml missing)")
        elif local_status == 'FAILED':
            self.db.update_state(struct_id, 'RELAX_FAILED')
            print(f"  {struct_id}: Relax failed")
        else:
            # Job not in queue and no completion marker - check if timed out
            relax_dir = Path(sdata['relax_dir'])
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
                # Check if electronic converged (using OSZICAR) and CONTCAR exists
                contcar_path = relax_dir / 'CONTCAR'
                
                if not contcar_path.exists() or contcar_path.stat().st_size == 0:
                    self.db.update_state(struct_id, 'RELAX_FAILED',
                                       error='Job timed out, CONTCAR missing/empty')
                    print(f"  {struct_id}: Relax FAILED (timeout, CONTCAR missing)")
                else:
                    converged, _ = check_electronic_convergence_oszicar(relax_dir)
                    if converged:
                        self.db.update_state(struct_id, 'RELAX_TMOUT',
                                           error='Relaxation timed out but electronic converged')
                        print(f"  {struct_id}: Relax TMOUT (electronic converged, usable)")
                    else:
                        self.db.update_state(struct_id, 'RELAX_FAILED',
                                           error='Job timed out, electronic not converged')
                        print(f"  {struct_id}: Relax FAILED (timeout, electronic not converged)")
            else:
                self.db.update_state(struct_id, 'RELAX_FAILED', 
                                   error='Job terminated without completion marker (crash)')
                print(f"  {struct_id}: Relax FAILED (crash)")
    
    def monitor_and_submit(self, structures_dict):
        """Main monitoring loop that checks status and submits new jobs."""
        print("\n" + "="*70)
        print("Starting relaxation monitoring loop...")
        print(f"Max concurrent structures: {self.max_concurrent}")
        print(f"Check interval: {self.check_interval}s")
        print("="*70 + "\n")
        sys.stdout.flush()
        
        while True:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking job status...")
            sys.stdout.flush()
            
            # Update status of all running jobs
            for struct_id in list(self.db.data['structures'].keys()):
                self.update_structure_status(struct_id)
            
            # Check if we can submit new jobs
            running_count = self.db.get_running_count()
            print(f"Currently running: {running_count}/{self.max_concurrent}")
            
            for struct_id in list(self.db.data['structures'].keys()):
                if running_count >= self.max_concurrent:
                    break
                
                sdata = self.db.get_structure(struct_id)
                state = sdata['state']
                structure = structures_dict.get(struct_id)
                
                if not structure:
                    continue
                
                if state == 'PENDING':
                    if self.submit_relax(struct_id, structure):
                        running_count += 1
            
            # Print statistics
            stats = self.db.get_stats()
            print("\nStatistics:")
            for state, count in sorted(stats['states'].items()):
                print(f"  {state}: {count}")
            sys.stdout.flush()
            
            # Check if all done
            pending_count = len(self.db.get_by_state('PENDING'))
            if running_count == 0 and pending_count == 0:
                completed = len(self.db.get_by_state('RELAX_DONE'))
                tmout = len(self.db.get_by_state('RELAX_TMOUT'))
                total = stats['total']
                failed_count = len(self.db.get_by_state('RELAX_FAILED'))
                if completed + tmout + failed_count >= total:
                    print("\n" + "="*70)
                    print("All relaxations completed!")
                    print(f"Successfully relaxed: {completed}/{total}")
                    print(f"Timed out (usable): {tmout}/{total}")
                    print(f"Failed: {failed_count}/{total}")
                    print("="*70)
                    sys.stdout.flush()
                    break
            
            print(f"\nSleeping for {self.check_interval}s...")
            sys.stdout.flush()
            time.sleep(self.check_interval)
    
    def initialize_structures(self, results_dir, output_dir, 
                             max_compositions=None, max_structures=5,
                             prescreen_results=None):
        """Scan results directory and initialize database."""
        results_dir = Path(results_dir)
        output_dir = Path(output_dir)
        
        print("="*70)
        print("Initializing Binary Magnets Coarse Relaxation Workflow")
        print("="*70)
        print(f"Results directory: {results_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Max concurrent: {self.max_concurrent}")
        print(f"Max compositions: {max_compositions or 'all'}")
        print(f"Max structures: {max_structures}")
        
        # Load pre-screening results if provided
        passed_structures = None
        if prescreen_results:
            prescreen_path = Path(prescreen_results)
            if prescreen_path.exists():
                print(f"Pre-screening results: {prescreen_path}")
                with open(prescreen_path, 'r') as f:
                    prescreen_data = json.load(f)
                
                passed_structures = set()
                for result in prescreen_data.get('results', []):
                    if result.get('passed_prescreening', False):
                        passed_structures.add(result['structure_id'])
                
                print(f"Structures passed pre-screening: {len(passed_structures)}")
                print(f"Energy threshold: {prescreen_data['summary']['hull_threshold']} eV/atom")
            else:
                print(f"Warning: Pre-screening file not found: {prescreen_path}")
                print("Will process all structures without filtering")
        else:
            print("No pre-screening filter (will process all structures)")
        
        print("="*70 + "\n")
        
        comp_dirs = sorted(results_dir.glob("*_structures"))
        if max_compositions:
            comp_dirs = comp_dirs[:max_compositions]
        
        structures_dict = {}
        
        for comp_dir in comp_dirs:
            comp_name = comp_dir.name.replace("_structures", "")
            zip_path = comp_dir / "generated_crystals_cif.zip"
            
            if not zip_path.exists():
                print(f"  Skipping {comp_name} (no ZIP file)")
                continue
            
            print(f"Scanning {comp_name}...")
            
            structures = self.read_structures_from_zip(zip_path, max_structures)
            if not structures:
                print(f"  No structures found")
                continue
            
            added_count = 0
            for idx, structure in enumerate(structures, 1):
                struct_id = f"{comp_name}_s{idx:03d}"
                
                # Skip if not in passed structures (when filtering is enabled)
                if passed_structures is not None and struct_id not in passed_structures:
                    continue
                
                structures_dict[struct_id] = structure
                
                if struct_id not in self.db.data['structures']:
                    elements = sorted([str(el) for el in structure.composition.elements])
                    chemsys = '-'.join(elements)
                    
                    self.db.add_structure(
                        struct_id, comp_name, idx,
                        output_dir / comp_name,
                        chemsys=chemsys
                    )
                    added_count += 1
            
            if added_count > 0:
                print(f"  Added {added_count} structures")
        
        # Load structures from database that aren't in structures_dict yet
        # This handles resume scenarios where structures exist in DB but weren't loaded from ZIP
        print("\nChecking database for additional structures...")
        loaded_from_contcar = 0
        skipped_count = 0
        
        for struct_id, sdata in self.db.data['structures'].items():
            if struct_id in structures_dict:
                continue  # Already loaded from ZIP
            
            # Try to load from Relax/CONTCAR for structures that have been processed
            if sdata['state'] not in ['PENDING', 'RELAX_RUNNING']:
                relax_dir = Path(sdata['relax_dir'])
                contcar_path = relax_dir / 'CONTCAR'
                
                if contcar_path.exists():
                    try:
                        structure = Structure.from_file(str(contcar_path))
                        structures_dict[struct_id] = structure
                        loaded_from_contcar += 1
                    except Exception as e:
                        print(f"  Warning: Could not load {struct_id} from CONTCAR: {e}")
                        skipped_count += 1
                else:
                    skipped_count += 1
        
        if loaded_from_contcar > 0:
            print(f"  Loaded {loaded_from_contcar} structures from CONTCAR files (resume)")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} structures (no CONTCAR available)")
        
        self.db.data['config'] = {
            'max_concurrent': self.max_concurrent,
            'results_dir': str(results_dir),
            'output_dir': str(output_dir),
            'max_structures': max_structures
        }
        self.db.save()
        
        print(f"\nTotal structures ready for workflow: {len(structures_dict)}")
        
        # Report structures in database but not in structures_dict
        missing_count = len(self.db.data['structures']) - len(structures_dict)
        if missing_count > 0:
            print(f"  Note: {missing_count} structures in database but not in structures_dict")
            print(f"        (These will be skipped during monitoring)")
        
        return structures_dict


def main():
    parser = argparse.ArgumentParser(
        description="Binary Magnets Coarse Relaxation - Batch submission with monitoring (relaxation only, no SPE)"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help="MatterGen results directory"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/scratch/$USER/VASP_JOBS',
        help="Output directory for VASP jobs"
    )
    parser.add_argument(
        '--db',
        type=str,
        default='workflow.json',
        help="JSON database file path"
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=10,
        help="Max concurrent structures running"
    )
    parser.add_argument(
        '--max-compositions',
        type=int,
        default=None,
        help="Max compositions to process"
    )
    parser.add_argument(
        '--max-structures',
        type=int,
        default=5,
        help="Max structures per composition"
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=60,
        help="Status check interval in seconds"
    )
    parser.add_argument(
        '--init-only',
        action='store_true',
        help="Only initialize database, don't start monitoring"
    )
    parser.add_argument(
        '--prescreen-results',
        type=str,
        default=None,
        help="Path to prescreening_stability.json (filters structures by energy_above_hull)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    db_path = Path(args.db).expanduser()
    
    if not db_path.is_absolute():
        db_path = output_dir / args.db
    
    # Create workflow manager
    manager = VASPRelaxManager(
        db_path=db_path,
        max_concurrent=args.max_concurrent,
        check_interval=args.check_interval
    )
    
    # Initialize structures
    structures_dict = manager.initialize_structures(
        results_dir=results_dir,
        output_dir=output_dir,
        max_compositions=args.max_compositions,
        max_structures=args.max_structures,
        prescreen_results=args.prescreen_results
    )
    
    if args.init_only:
        print("\n" + "="*70)
        print("Initialization complete!")
        print(f"Database: {db_path}")
        print("="*70)
        return
    
    # Start monitoring and submission loop
    try:
        manager.monitor_and_submit(structures_dict)
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        print(f"Database saved to: {db_path}")
        print("Resume with same command to continue.")


if __name__ == '__main__':
    main()
