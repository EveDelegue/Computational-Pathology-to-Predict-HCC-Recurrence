
wsi_path="data/patches/*"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/STEP3_detect_inflammatory_cells_2.py --slide_name "$wsi"
done