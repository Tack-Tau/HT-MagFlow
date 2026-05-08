#!/usr/bin/env python3
"""
CeFe Derivatives VASP Workflow Manager - 3-step progressive relaxation
No MongoDB/FireWorks - uses local JSON database for job tracking

Features:
- 3-step progressive VASP relaxation for accurate energies
- Simplified workflow (no SC/PARCHG/ELF - only relaxation for energy_above_hull analysis)
- Custom INCAR settings for CeFe system (ISPIN=2, ISMEAR=1)
- LDAU handled by pymatgen defaults (only for O/F-containing systems, MP-compatible)
- Dynamic job submission with concurrency control

Based on refine_electrideflow.py but tailored for CeFe derivatives.
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
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
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



# MAGMOM overrides for elements with wrong pymatgen neutral-element defaults.
MAGMOM_OVERRIDE = {
    'Pr': 2.0, 'Nd': 3.0, 'Pm': 4.0, 'Sm': 5.0,
    'Gd': 7.0, 'Tb': 6.0, 'Dy': 5.0, 'Ho': 4.0,
    'Er': 3.0, 'Tm': 2.0, 'Yb': 1.0,
    'Co': 2.0,
}


def build_magmom(structure):
    """Build MAGMOM dict with correct rare earth magnetic moments.
    
    When overriding via user_incar_settings, pymatgen does NOT merge with its
    config defaults, so all elements must be included in the returned dict.
    Returns dict (element -> value) if any override is needed,
    None otherwise (use pymatgen defaults).
    """
    elements = [str(el) for el in structure.composition.elements]
    if not any(el in MAGMOM_OVERRIDE for el in elements):
        return None
    pmg_defaults = {
        'Ce': 5, 'Eu': 10, 'Fe': 5, 'Ni': 5,
        'Mn': 5, 'Cr': 5, 'V': 5, 'Mo': 5, 'W': 5,
    }
    result = {}
    for el in elements:
        if el in MAGMOM_OVERRIDE:
            result[el] = MAGMOM_OVERRIDE[el]
        elif el in pmg_defaults:
            result[el] = pmg_defaults[el]
        else:
            result[el] = 0.6
    return result



class WorkflowDatabase:
    """
    Simple JSON-based database for tracking job states.
    
    Job Status Design for 3-Step Relaxation:
    ----------------------------------------
    - PENDING: Not yet submitted
    - RELAX_RUNNING: 3-step relaxation job is running
    - RELAX_DONE: All 3 steps completed successfully (final state)
    - RELAX_TMOUT: Step 3 timed out but produced CONTCAR (usable, final state)
    - RELAX_FAILED: Job failed at any step
    """
    
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
        running_states = ['RELAX_RUNNING']
        return sum(1 for s in self.data['structures'].values() 
                   if s['state'] in running_states)
    
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


class VASPWorkflowManager:
    """Manages VASP job submission and monitoring with batch control."""
    
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
    
    def create_vasp_inputs(self, structure, job_dir, job_type='relax', step=1):
        """
        Create VASP input files for 3-step progressive relaxation.
        
        Progressive relaxation parameters for CeFe derivatives:
        - Step 1: ISIF=2, EDIFFG=-0.05, IBRION=2  (quick ionic relaxation, fixed cell)
        - Step 2: ISIF=3, EDIFFG=-0.02, IBRION=1  (full relaxation with robust RMM-DIIS)
        - Step 3: ISIF=3, EDIFFG=-0.01, IBRION=1  (final high-precision, robust optimizer)
        
        LDAU is handled by pymatgen defaults (anion-dependent, MP-compatible).
        MAGMOM overridden for rare earths that pymatgen maps to 0.6 by default.
        Note: Structure is symmetrized using PyXtal with progressive tolerance.
        """
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        
        if job_type != 'relax':
            raise ValueError(f"Only 'relax' job_type supported in CeFe workflow, got: {job_type}")
        
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
        
        # Progressive relaxation parameters for each step
        if step == 1:
            ediffg = -0.05
            isif = 2
            ibrion = 2
            potim = 0.3
        elif step == 2:
            ediffg = -0.02
            isif = 3
            ibrion = 1
            potim = 0.2
        elif step == 3:
            ediffg = -0.01
            isif = 3
            ibrion = 1
            potim = 0.1
        else:
            raise ValueError(f"Invalid step number: {step}. Must be 1, 2, or 3.")
        
        incar_settings = {
            'PREC': 'Normal',
            'ALGO': 'Normal',
            'ADDGRID': True,
            'ENCUT': 520,
            'EDIFF': 1e-4,
            'EDIFFG': ediffg,
            'IBRION': ibrion,
            'ISIF': isif,
            'NELM': 120,
            'NSW': 100,
            'ISMEAR': 1,
            'SIGMA': 0.05,
            'ISPIN': 2,
            'POTIM': potim,
            'LREAL': 'Auto',
            'LWAVE': False,
            'LCHARG': False,
            'LAECHG': False,
            'LASPH': True,
            'LORBIT': 11,
            'NCORE': 4,
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
    
    def create_slurm_script(self, job_dir, job_name, job_type='relax'):
        """
        Create SLURM submission script for 3-step progressive relaxation.
        
        Progressive Relaxation Strategy:
        --------------------------------
        - Step 1: ISIF=2, EDIFFG=-0.02, IBRION=2, POTIM=0.3 (quick ionic, CG)
        - Step 2: ISIF=3, EDIFFG=-0.01, IBRION=1, POTIM=0.2 (full relax, RMM-DIIS)
        - Step 3: ISIF=3, EDIFFG=-0.005, IBRION=1, POTIM=0.1 (tight, RMM-DIIS)
        
        Timeout/Failure Handling:
        -------------------------
        1. Before each step: Save POSCAR-{1,2,3} for debugging
        2. Steps 1-2: Timeout OK if CONTCAR exists + electronic converged (continue)
        3. Step 3: Timeout → RELAX_TMOUT marker (CONTCAR usable)
        4. Any VASP failure → VASP_FAILED marker (check POSCAR-* count to identify step)
        5. Exit codes 140, 143 indicate SLURM timeout
        """
        job_dir = Path(job_dir).resolve()
        script_path = job_dir / 'job.sh'
        
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}_{job_type}
#SBATCH --partition=Apus,Orion
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=str-c[85-95]
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

# VASP executable
VASP_CMD="srun --mpi=pmi2 $HOME/apps/vasp.6.2.1/bin/vasp_std"

# Change to job directory
cd {job_dir}

"""
        
        # Relax job type - only job type used in CeFe workflow
        if job_type == 'relax':
            script += f"""
# Run 3 consecutive VASP relaxation steps with timeout/failure handling
echo "Starting 3-step VASP relaxation for CeFe derivatives"
echo "Working directory: $(pwd)"
echo "VASP command: $VASP_CMD"
echo "Start time: $(date)"

# Relaxation step 1
echo ""
echo "========================================"
echo "Relaxation Step 1/3"
echo "  ISIF=2, EDIFFG=-0.02, IBRION=2, POTIM=0.3"
echo "========================================"

# Save initial POSCAR for debugging
cp POSCAR POSCAR-1

# Use INCAR-1 for step 1
cp INCAR-1 INCAR

$VASP_CMD

EXIT_CODE=$?
echo "Step 1 exit code: $EXIT_CODE"

# Check if SLURM timeout occurred
if [ $EXIT_CODE -eq 140 ] || [ $EXIT_CODE -eq 143 ]; then
    echo "WARNING: Step 1 timed out (exit code $EXIT_CODE)"
    if [ -f "CONTCAR" ] && [ -s "CONTCAR" ]; then
        echo "CONTCAR exists, proceeding to step 2 with partial relaxation"
    else
        echo "ERROR: Step 1 timed out without producing CONTCAR"
        rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
        touch VASP_FAILED
        exit 1
    fi
elif [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Relaxation step 1 failed with exit code $EXIT_CODE"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
    exit 1
fi

# Verify CONTCAR from step 1
if [ ! -f "CONTCAR" ] || [ ! -s "CONTCAR" ]; then
    echo "ERROR: CONTCAR missing/empty after step 1"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
    exit 1
fi

# Check electronic convergence from OSZICAR (search backwards for last
# converged ionic step, handles killed-mid-electronic-SCF case)
echo ""
echo "Checking electronic convergence (OSZICAR)..."
if [ ! -f "OSZICAR" ] || [ ! -s "OSZICAR" ]; then
    echo "ERROR: OSZICAR not found - cannot verify convergence"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
    exit 1
fi

NELM_VAL=$(grep -m1 'NELM' INCAR | awk -F'=' '{{print $2}}' | awk '{{print $1}}')
NELM_VAL=${{NELM_VAL:-60}}
CONVERGED=0
F_LINES=$(grep -n "F=" OSZICAR | tac)
if [ -n "$F_LINES" ]; then
    while IFS=: read -r FNUM REST; do
        SCF_NUM=$((FNUM - 1))
        [ "$SCF_NUM" -lt 1 ] && continue
        ESTEP=$(sed -n "${{SCF_NUM}}p" OSZICAR | awk '{{print $2}}')
        if [ "$ESTEP" -lt "$NELM_VAL" ] 2>/dev/null; then
            echo "  Electronic SCF converged in step 1 (e-steps: $ESTEP < NELM=$NELM_VAL)"
            CONVERGED=1
            break
        fi
    done <<< "$F_LINES"
fi
if [ "$CONVERGED" -ne 1 ]; then
    echo "ERROR: No ionic step with converged electronic SCF found in step 1"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
    exit 1
fi

echo "Step 1 completed (exit code: $EXIT_CODE)"
echo "Copying CONTCAR -> POSCAR for step 2"
cp CONTCAR POSCAR

# Clean large intermediate files between steps
rm -f WAVECAR CHGCAR CHG WFULL TMPCAR AECCAR* 2>/dev/null

# Relaxation step 2
echo ""
echo "========================================"
echo "Relaxation Step 2/3"
echo "  ISIF=3, EDIFFG=-0.01, IBRION=1 (RMM-DIIS), POTIM=0.2"
echo "========================================"

# Save POSCAR for debugging (should be identical to step 1 CONTCAR)
cp POSCAR POSCAR-2

# Use INCAR-2 for step 2
cp INCAR-2 INCAR

$VASP_CMD

EXIT_CODE=$?
echo "Step 2 exit code: $EXIT_CODE"

# Check if SLURM timeout occurred
if [ $EXIT_CODE -eq 140 ] || [ $EXIT_CODE -eq 143 ]; then
    echo "WARNING: Step 2 timed out (exit code $EXIT_CODE)"
    if [ -f "CONTCAR" ] && [ -s "CONTCAR" ]; then
        echo "CONTCAR exists, proceeding to step 3 with partial relaxation"
    else
        echo "ERROR: Step 2 timed out without producing CONTCAR"
        rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
        touch VASP_FAILED
        exit 1
    fi
elif [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Relaxation step 2 failed with exit code $EXIT_CODE"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
    exit 1
fi

# Verify CONTCAR from step 2
if [ ! -f "CONTCAR" ] || [ ! -s "CONTCAR" ]; then
    echo "ERROR: CONTCAR missing/empty after step 2"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
    exit 1
fi

# Check electronic convergence from OSZICAR (search backwards for last
# converged ionic step, handles killed-mid-electronic-SCF case)
echo ""
echo "Checking electronic convergence (OSZICAR)..."
if [ ! -f "OSZICAR" ] || [ ! -s "OSZICAR" ]; then
    echo "ERROR: OSZICAR not found - cannot verify convergence"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
    exit 1
fi

NELM_VAL=$(grep -m1 'NELM' INCAR | awk -F'=' '{{print $2}}' | awk '{{print $1}}')
NELM_VAL=${{NELM_VAL:-60}}
CONVERGED=0
F_LINES=$(grep -n "F=" OSZICAR | tac)
if [ -n "$F_LINES" ]; then
    while IFS=: read -r FNUM REST; do
        SCF_NUM=$((FNUM - 1))
        [ "$SCF_NUM" -lt 1 ] && continue
        ESTEP=$(sed -n "${{SCF_NUM}}p" OSZICAR | awk '{{print $2}}')
        if [ "$ESTEP" -lt "$NELM_VAL" ] 2>/dev/null; then
            echo "  Electronic SCF converged in step 2 (e-steps: $ESTEP < NELM=$NELM_VAL)"
            CONVERGED=1
            break
        fi
    done <<< "$F_LINES"
fi
if [ "$CONVERGED" -ne 1 ]; then
    echo "ERROR: No ionic step with converged electronic SCF found in step 2"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
    exit 1
fi

echo "Step 2 completed (exit code: $EXIT_CODE)"
echo "Copying CONTCAR -> POSCAR for step 3"
cp CONTCAR POSCAR

# Clean large intermediate files between steps
rm -f WAVECAR CHGCAR CHG WFULL TMPCAR AECCAR* 2>/dev/null

# Relaxation step 3
echo ""
echo "========================================"
echo "Relaxation Step 3/3"
echo "  ISIF=3, EDIFFG=-0.005, IBRION=1 (RMM-DIIS), POTIM=0.1"
echo "========================================"

# Save POSCAR for debugging (should be identical to step 2 CONTCAR)
cp POSCAR POSCAR-3

# Use INCAR-3 for step 3 (final high-precision)
cp INCAR-3 INCAR

$VASP_CMD

EXIT_CODE=$?
echo "Step 3 exit code: $EXIT_CODE"

# Check if SLURM timeout occurred in final step
if [ $EXIT_CODE -eq 140 ] || [ $EXIT_CODE -eq 143 ]; then
    echo "WARNING: Step 3 (final) timed out (exit code $EXIT_CODE)"
    if [ -f "CONTCAR" ] && [ -s "CONTCAR" ]; then
        echo "CONTCAR exists - marking as RELAX_TMOUT"
        echo "Partial relaxation completed, may proceed to analysis"
        # Mark as timeout instead of failure, clean up large files
        echo "Cleaning up large intermediate files..."
        rm -f CHGCAR CHG WAVECAR WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
        touch RELAX_TMOUT
        touch VASP_DONE
        exit 0
    else
        echo "ERROR: Step 3 timed out without producing CONTCAR"
        rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
        touch VASP_FAILED
        exit 1
    fi
elif [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Relaxation step 3 failed with exit code $EXIT_CODE"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
    exit 1
fi

# Verify CONTCAR from step 3
if [ ! -f "CONTCAR" ] || [ ! -s "CONTCAR" ]; then
    echo "ERROR: CONTCAR missing/empty after step 3"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
    exit 1
fi

# Check electronic convergence from OSZICAR (search backwards for last
# converged ionic step, handles killed-mid-electronic-SCF case)
echo ""
echo "Checking electronic convergence (OSZICAR)..."
if [ ! -f "OSZICAR" ] || [ ! -s "OSZICAR" ]; then
    echo "ERROR: OSZICAR not found - cannot verify convergence"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
    exit 1
fi

NELM_VAL=$(grep -m1 'NELM' INCAR | awk -F'=' '{{print $2}}' | awk '{{print $1}}')
NELM_VAL=${{NELM_VAL:-60}}
CONVERGED=0
F_LINES=$(grep -n "F=" OSZICAR | tac)
if [ -n "$F_LINES" ]; then
    while IFS=: read -r FNUM REST; do
        SCF_NUM=$((FNUM - 1))
        [ "$SCF_NUM" -lt 1 ] && continue
        ESTEP=$(sed -n "${{SCF_NUM}}p" OSZICAR | awk '{{print $2}}')
        if [ "$ESTEP" -lt "$NELM_VAL" ] 2>/dev/null; then
            echo "  Electronic SCF converged in step 3 (e-steps: $ESTEP < NELM=$NELM_VAL)"
            CONVERGED=1
            break
        fi
    done <<< "$F_LINES"
fi
if [ "$CONVERGED" -ne 1 ]; then
    echo "ERROR: No ionic step with converged electronic SCF found in step 3"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
    touch VASP_FAILED
    exit 1
fi

echo ""
echo "========================================"
echo "All 3 Relaxation Steps Completed Successfully"
echo "========================================"
echo "End time: $(date)"
echo "Final structure saved in CONTCAR"

# Clean up large unnecessary files to save disk space
echo ""
echo "Cleaning up large intermediate files..."
rm -f CHGCAR CHG WAVECAR WFULL AECCAR* TMPCAR PROCAR 2>/dev/null
echo "Cleanup complete - kept: POSCAR, CONTCAR, INCAR, KPOINTS, POTCAR, OUTCAR, OSZICAR, vasprun.xml, POSCAR-*"

touch VASP_DONE
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
            # Check if it's a timeout completion
            if (job_dir / 'RELAX_TMOUT').exists():
                return 'TIMEOUT'
            return 'DONE'
        elif (job_dir / 'VASP_FAILED').exists():
            return 'FAILED'
        else:
            return 'UNKNOWN'
    
    def submit_relax(self, struct_id, structure):
        """
        Submit relaxation job for a structure.
        
        Generates 3 INCAR files (INCAR-1, INCAR-2, INCAR-3) with progressive parameters:
        - INCAR-1: ISIF=2, EDIFFG=-0.02, IBRION=2, POTIM=0.3
        - INCAR-2: ISIF=3, EDIFFG=-0.01, IBRION=1, POTIM=0.2
        - INCAR-3: ISIF=3, EDIFFG=-0.005, IBRION=1, POTIM=0.1
        """
        sdata = self.db.get_structure(struct_id)
        if not sdata:
            return False
        
        relax_dir = Path(sdata['relax_dir'])
        job_name = struct_id
        
        print(f"  Submitting Relax: {struct_id}")
        
        try:
            for step_num in [1, 2, 3]:
                self.create_vasp_inputs(structure, relax_dir, 'relax', step=step_num)
                shutil.copy2(relax_dir / 'INCAR', relax_dir / f'INCAR-{step_num}')
            
            script = self.create_slurm_script(relax_dir, job_name, 'relax')
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
        
        if state == 'RELAX_RUNNING':
            job_status = self.check_job_status(sdata['relax_job_id'])
            if job_status == 'NOTFOUND':
                local_status = self.check_local_status(sdata['relax_dir'])
                relax_dir = Path(sdata['relax_dir'])
                
                if local_status == 'DONE':
                    # Successfully completed all 3 steps
                    vasprun_path = relax_dir / 'vasprun.xml'
                    if vasprun_path.exists():
                        try:
                            vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
                            if not vr.converged_electronic:
                                self.db.update_state(struct_id, 'RELAX_FAILED', 
                                                   error='Electronic SCF not converged in final step')
                                print(f"  {struct_id}: Relax FAILED (electronic not converged)")
                            else:
                                self.db.update_state(struct_id, 'RELAX_DONE')
                                print(f"  {struct_id}: 3-step relaxation completed")
                        except Exception as e:
                            self.db.update_state(struct_id, 'RELAX_FAILED', 
                                               error=f'Convergence check error: {e}')
                    else:
                        self.db.update_state(struct_id, 'RELAX_FAILED', 
                                           error='vasprun.xml not found')
                
                elif local_status == 'TIMEOUT':
                    # Step 3 timed out but CONTCAR exists (marked by SLURM script)
                    self.db.update_state(struct_id, 'RELAX_TMOUT',
                                       error='Step 3 timed out but produced CONTCAR')
                    print(f"  {struct_id}: Relax TMOUT (step 3 timeout, CONTCAR available)")
                
                elif local_status == 'FAILED':
                    # VASP_FAILED marker exists - determine which step failed
                    poscar_files = sorted(relax_dir.glob('POSCAR-*'))
                    num_steps = len(poscar_files)
                    if num_steps > 0:
                        error_msg = f'VASP failed at step {num_steps}'
                        print(f"  {struct_id}: Relax FAILED (failed at step {num_steps}/3)")
                    else:
                        error_msg = 'VASP failed (unknown step)'
                        print(f"  {struct_id}: Relax FAILED")
                    self.db.update_state(struct_id, 'RELAX_FAILED', error=error_msg)
                
                else:
                    # Job not in queue and no completion marker - crashed
                    self.db.update_state(struct_id, 'RELAX_FAILED', 
                                       error='Job terminated without completion marker (crash)')
                    print(f"  {struct_id}: Relax FAILED (crash)")
    
    def monitor_and_submit(self, structures_dict):
        """Main monitoring loop that checks status and submits new jobs."""
        print("\n" + "="*70)
        print("Starting workflow monitoring loop...")
        print(f"Max concurrent structures: {self.max_concurrent}")
        print(f"Check interval: {self.check_interval}s")
        print("="*70 + "\n")
        sys.stdout.flush()
        
        while True:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking job status...")
            sys.stdout.flush()
            
            for struct_id in list(self.db.data['structures'].keys()):
                self.update_structure_status(struct_id)
            
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
            
            stats = self.db.get_stats()
            print("\nStatistics:")
            for state, count in sorted(stats['states'].items()):
                print(f"  {state}: {count}")
            sys.stdout.flush()
            
            pending_count = len(self.db.get_by_state('PENDING'))
            if running_count == 0 and pending_count == 0:
                completed = len(self.db.get_by_state('RELAX_DONE'))
                tmout = len(self.db.get_by_state('RELAX_TMOUT'))
                total = stats['total']
                failed_count = len(self.db.get_by_state('RELAX_FAILED'))
                if completed + tmout + failed_count >= total:
                    print("\n" + "="*70)
                    print("All workflows completed!")
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
        """Scan results directory and initialize database for 3-step relaxation workflow."""
        results_dir = Path(results_dir)
        output_dir = Path(output_dir)
        
        print("="*70)
        print("Initializing CeFe Derivatives VASP Workflow (3-step Relaxation)")
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
        description="CeFe Derivatives VASP Workflow Manager - 3-step progressive relaxation for energy_above_hull analysis"
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
    manager = VASPWorkflowManager(
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

