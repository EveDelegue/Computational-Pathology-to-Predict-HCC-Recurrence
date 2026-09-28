#!/bin/bash
wsi_path="/mnt/backup/edelegue/WSIs/hospitals/PB/Patient_*"
docker start eve_5
docker exec eve_5 pip install -r requirements_cells.txt
docker exec eve_5 pip install -e .

for patient_dir in $(ls -d $wsi_path | sort -r); do
    if [ -d "$patient_dir" ]; then        
    echo "folder $patient_dir"
    #cp
    #process
    # rm 
    dest_folder="data/WSIs/PB2"
    mkdir $dest_folder
    cp -r $patient_dir $dest_folder
    echo "process"
    #docker exec eve_2 .venv/bin/python brouillons/hello_world.py
    docker exec eve_5 python src/expert_cell_extraction.py
    #### completer le process



    rm -r $dest_folder
    fi
done

