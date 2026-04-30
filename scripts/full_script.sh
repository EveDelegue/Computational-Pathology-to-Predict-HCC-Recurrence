#!/bin/bash

python src/STEP0_create_directories.py

wsi_path="data/WSIs/*"
# list hospitals
for wsi in $wsi_path; do
    # extract hospital name
    hospital=${wsi: 10}
    echo " processing hospital $hospital"
    # if there are patients from the hospital
    if find $wsi -mindepth 1 -maxdepth 1 | read; then
    # list patients
    wsi_list="$wsi/*"
    for patient_file in $wsi_list; do
    # extract patient name
        patient=${patient_file#*/*/*/}
        echo " patient $patient "
        # if there are files in the patients folder
        if find $patient_file -mindepth 1 -maxdepth 1 | read; then
        # select the mrxs ones
        patient_file_list="$patient_file/*.mrxs"
        for slide in $patient_file_list; do
            # extract slide name
            slide_path=${slide#*/*/*/*/}
            slide_name=${slide_path%.*}
            # run patching
            echo "patching $slide_name"
            python src/STEP1_gen_patches_from_WSI.py --slide_name "$slide" --verbose True
            # find the patched data
            patched_path="data/patches/${slide_name}_$hospital"
            # run tumor detection
            echo "detecting tumor from $patched_path"
            python src/STEP2_detect_tumor_from_WSI.py --slide_name "$patched_path" --verbose True
            # run nucleus detection
            echo "detecting nucleus from $patched_path"
            python src/STEP4.py --slide_name "$patched_path" --verbose True
            # deleting the patches folder
            rm -r $patched_path
            # run inflammation detection
            patched_path_bis="data/patches_bis/${slide_name}_$hospital"
            echo "detecting inflammatory cells on $patched_path_bis"
            python src/STEP3_detect_inflammatory_cells.py --slide_name "$patched_path_bis" --verbose True
            # deleting the patches bis folder
            rm -r $patched_path_bis
            # deleting the inflam dats folder
            rm -r checkpoints/inflam_dats/${slide_name}_${hospital}_raw.pt
        done
        else 
            echo "$patient empty"
        fi
    done
    else
        echo "$hospital empty"

    fi
    
done

python src/STEP5.py
python src/STEP6.py
python src/STEP7.py
python src/STEP8.py
