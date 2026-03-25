
wsi_path="data/patches/*"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/STEP2_detect_tumor_from_WSI.py --slide_name "$wsi"
done