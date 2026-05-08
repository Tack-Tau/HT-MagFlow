# Ce-Fe Derivative Structures for Screening

Generated using `gen_CeFe_dev.py` from Ce-Fe prototype structures.

## Generation Summary

- **Prototypes**: 2 structures (`Ce10Fe52.cif`, `Ce2Fe14B.cif`)
- **A-type substitution** (Ce → 14 rare earths): Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu
- **B-type substitution** (Fe → 9 transition metals): Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn
- **Total structures**: 252 (126 per prototype = 14 × 9)

## Structure Organization

Each composition has its own directory:
```
Ce2Co14B_structures/
  generated_crystals_cif.zip    # Contains Ce2Co14B_s001.cif

Pr5Ti26_structures/
  generated_crystals_cif.zip    # Contains Pr5Ti26_s001.cif
```

This format is compatible with VASPflow pre-screening workflow.

## Running Pre-screening on HPC

### 1. Transfer to HPC

```bash
# From local machine
scp -r CeFe_derivatives_results your_cluster:/scratch/$USER/

# Or use rsync for faster transfer
rsync -avz --progress CeFe_derivatives_results/ your_cluster:/scratch/$USER/CeFe_derivatives_results/
```

### 2. Run MatterSim Pre-screening

```bash
# SSH to HPC
ssh your_cluster
cd /scratch/$USER

# Submit pre-screening job
bash run_prescreen.sh \
  --results-dir ./CeFe_derivatives_results \
  --output-dir ./VASP_JOBS_CeFe \
  --device cuda \
  --hull-threshold 0.05 \
  --max-structures 0

# Monitor progress
squeue -u $USER | grep prescreen
tail -f prescreen_*.out

# Check results after completion
cat VASP_JOBS_CeFe/prescreening_stability.json | jq '.summary'
```

### 3. Run VASP Workflow (on passed structures)

```bash
# After pre-screening completes
bash run_workflow.sh \
  --results-dir ./CeFe_derivatives_results \
  --output-dir ./VASP_JOBS_CeFe \
  --max-concurrent 20 \
  --max-structures 0
```

### 4. Analyze for Electrides

```bash
# After ELF calculations complete
bash submit_analysis.sh \
  --vasp-jobs ./VASP_JOBS_CeFe \
  --output electride_CeFe.csv
```

## Expected Pre-screening Time

- **Per structure**: ~10-30s (CPU) or ~5-10s (GPU)
- **Total (252 structures)**: 
  - GPU: ~20-40 minutes
  - CPU: ~1-2 hours
- **MP cache**: First query per chemical system takes 2-5 min, then cached

## Chemical Systems

The 252 structures span multiple chemical systems:
- **A-Ti-B** systems (14 compositions from Ce2Fe14B prototype)
- **A-V-B** systems (14 compositions)
- ... up to **A-Zn-B**
- **A-Ti** systems (14 compositions from Ce10Fe52 prototype)
- **A-V** systems (14 compositions)
- ... up to **A-Zn**

Each unique chemical system requires one MP query (cached afterward).

## Regeneration

To regenerate with different parameters:

```bash
# Local machine
cd $HOME/SOFT/vaspflow

source ~/.bashrc

conda activate vaspflow

# Activate pymatgen environment
python3 gen_CeFe_dev.py \
  --prototype-dir ./ \
  --output-dir ./CeFe_derivatives_results

# Test mode (5 structures per prototype)
python3 gen_CeFe_dev.py --test

# Full options
python3 gen_CeFe_dev.py --help
```

## Directory Structure

```
CeFe_derivatives_results/
├── Ce2Co14B_structures/
│   └── generated_crystals_cif.zip
├── Ce2Cr14B_structures/
│   └── generated_crystals_cif.zip
├── ...
├── Lu5Zn26_structures/
│   └── generated_crystals_cif.zip
└── README.md (this file)
```

## Validation

All generated structures:
- Follow reduced formula convention (e.g., Ce10Fe52 → Ce5Fe26)
- Are packaged in ZIP files compatible with VASPflow
- Maintain original crystal structure (only element substitution)
- Include proper CIF formatting with symmetry information

## Notes

- The `Ce2Fe14B` prototype contains Boron (B), which is preserved in all derivatives
- Formula reduction is automatic (e.g., Pr10Ti52 → Pr5Ti26)
- Element substitution is global (all Ce → A, all Fe → B)
- Original space group symmetry is maintained

