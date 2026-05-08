#!/bin/bash

echo "structure_id,mattersim_e_hull,dft_e_hull,mattersim_energy_per_atom,vasp_energy_per_atom,spg_num" > candidates.csv

while read -r id _ spg; do
    n=${spg#\#}

    mh=$(jq -r --arg id "$id" '.matched_structures[] | select(.structure_id==$id) | .mattersim_e_hull' hull_comparison.json)
    dh=$(jq -r --arg id "$id" '.matched_structures[] | select(.structure_id==$id) | .dft_e_hull' hull_comparison.json)

    me=$(jq -r --arg id "$id" '.results[] | select(.structure_id==$id) | .mattersim_energy_per_atom' prescreening_stability.json)
    ve=$(jq -r --arg id "$id" '.results[] | select(.structure_id==$id) | .vasp_energy_per_atom' dft_stability_results.json)

    echo "$id,$mh,$dh,$me,$ve,$n"
done < candidates_id_spg_list >> candidates.csv
