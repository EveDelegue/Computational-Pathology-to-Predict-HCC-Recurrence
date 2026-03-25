
python src/STEP0_create_directories.py

wsi_path="data/WSIs/PB/*/*.mrxs"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/STEP1_gen_patches_from_WSI.py --slide_name "$wsi"
done

wsi_path="data/patches/*"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/STEP2_detect_tumor_from_WSI.py --slide_name "$wsi"
    echo "detecting inflammatory cells on $wsi"
    python src/STEP3_detect_inflammatory_cells.py --slide_name "$wsi"
    echo "detecting nucleus features on $wsi"
    python src/STEP4.py --slide_name "$wsi"
done

python src/STEP5.py
python src/STEP6.py
python src/STEP7.py
python src/STEP8.py