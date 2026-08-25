#!/bin/bash
wsi_path="/mnt/backup/aziz_chaari/asadraoui_data_backups/backup_HM/Patient_*"


for patient_dir in $wsi_path; do
    if [ -d "$patient_dir" ]; then        
    echo "folder $patient_dir"
    #cp
    #process
    # rm 
    dest_folder="data/WSIs/HM"
    mkdir $dest_folder
    cp -r $patient_dir $dest_folder
    echo "process"
    #docker exec eve_2 .venv/bin/python brouillons/hello_world.py
    docker exec eve_2 .venv/bin/python src/expert_feature_extraction.py
    #### completer le process



    rm -r $dest_folder
    fi
done

