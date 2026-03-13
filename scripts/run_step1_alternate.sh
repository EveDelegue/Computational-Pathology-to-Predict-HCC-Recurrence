python src/STEP0_create_directories.py

wsi_path="data/WSIs/PB/*/*.mrxs"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/alternate_STEP1_gen_patches_from_WSI.py --slide_name "$wsi"
    wsi_path2="data/patches/*"
    for wsi2 in $wsi_path2; do
        echo "encoding $wsi2"
        python src/alternate_STEP2_foundation_model.py --slide_name "$wsi2" --model "gigapath" --xlsx "data/Label_slides.xlsx"
    done
done

wsi_path="data/WSIs/HM/*/*.ndpi"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/alternate_STEP1_gen_patches_from_WSI.py --slide_name "$wsi"
    wsi_path2="data/patches/*"
    for wsi2 in $wsi_path2; do
        echo "encoding $wsi2"
        python src/alternate_STEP2_foundation_model.py --slide_name "$wsi2" --model "gigapath" --xlsx "data/Label_slides.xlsx"
    done
done

wsi_path="data/WSIs/BJ/*/*.svs"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/alternate_STEP1_gen_patches_from_WSI.py --slide_name "$wsi"
    wsi_path2="data/patches/*"
    for wsi2 in $wsi_path2; do
        echo "encoding $wsi2"
        python src/alternate_STEP2_foundation_model.py --slide_name "$wsi2" --model "gigapath" --xlsx "data/Label_slides.xlsx"
    done
done

