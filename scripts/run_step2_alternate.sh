echo "fitting some classifiers after the foundation model"
python src/alternate_STEP3_bis_classifier.py

echo "cross-validating a linear layer on top of the foundation model"
python src/alternate_STEP3_training_final_layer.py --l1 1/76 --step 'cross_val'

echo "training a linear layer on top of the foundation model on all the data"
python src/alternate_STEP3_training_final_layer.py --l1 1/76 --step 'train_all'