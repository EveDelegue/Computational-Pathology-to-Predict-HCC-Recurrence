python src/STEP0_create_directories.py

wsi_path="data/WSIs/PB/*/*.mrxs"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/STEP1_gen_patches_from_WSI.py --slide_name "$wsi"
done


wsi_path="data/WSIs/HM/*/*.ndpi"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/STEP1_gen_patches_from_WSI.py --slide_name "$wsi"
done


wsi_path="data/WSIs/BJ/*/*.svs"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/STEP1_gen_patches_from_WSI.py --slide_name "$wsi"
done
