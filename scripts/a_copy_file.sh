#!/bin/bash
wsi_path="/mnt/backup/aziz_chaari/asadraoui_data_backups/backup_BJ/Patient_*"
docker start eve_5
docker exec eve_5 pip install -e .

for patient_dir in $wsi_path; do
    if [ -d "$patient_dir" ]; then        
    echo "folder $patient_dir"
    #cp
    #process
    # rm 
    dest_folder="data/WSIs/BJ"
    mkdir $dest_folder
    cp -r $patient_dir $dest_folder
    echo "process"
    #docker exec eve_2 .venv/bin/python brouillons/hello_world.py
    docker exec eve_5 python src/expert_feature_extraction.py
    #### completer le process



    rm -r $dest_folder
    fi
done

