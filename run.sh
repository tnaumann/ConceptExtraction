#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


python "$DIR/code/train.py" -t data/concept_assertion_relation_training_data/beth/txt/record-13.txt -c data/concept_assertion_relation_training_data/beth/concept/record-13.con -m models/run_models/run.model
python "$DIR/code/predict.py" -i data/concept_assertion_relation_training_data/beth/txt/record-13.txt -o data/test_predictions -m models/run_models/run.model
python "$DIR/code/evaluate.py" -t data/concept_assertion_relation_training_data/beth/txt/record-13.txt -c data/test_predictions/record-13.con -r data/concept_assertion_relation_training_data/beth/concept/record-13.con


