mkdir -p foo /wdata/logs

rm /wdata/test_pred_4k_double -r -f
echo "Predicting NNs"
python predict_double.py "$@" | tee /wdata/logs/predict_double.out

echo "Generating JSON"
python generate_json.py "$@" | tee /wdata/logs/generate_json.out

echo "Tracking buildings"
python track_buildings.py "$@" | tee /wdata/logs/track_buildings.out

echo "Creating submission"
python create_submission.py "$@" | tee /wdata/logs/create_submission.out

echo "Submission created!"