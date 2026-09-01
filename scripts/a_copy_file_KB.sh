#!/bin/bash
wsi_path="/mnt/wwn-0x50014ee2c13881aa-part1/patients_1_89/Patient_*"


for patient_dir in $wsi_path; do
    if [ -d "$patient_dir" ]; then        
    echo "folder $patient_dir"
    #cp
    #process
    # rm 
    dest_folder="data/WSIs/PB_2"
    mkdir $dest_folder
    cp -r $patient_dir $dest_folder
    echo "process"
    #docker exec eve_2 .venv/bin/python brouillons/hello_world.py
    .venv/bin/python src/expert_tumor_extraction.py
    #### completer le process



    rm -r $dest_folder
    fi
done

