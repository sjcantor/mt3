#!bin/bash

# ============ Inference using pretrained model ============
# Download checkpoint and inference
CHECKPOINT_PATH="CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
wget -O $CHECKPOINT_PATH "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
MODEL_TYPE="Note_pedal"


# Dataset
DATASET_DIR="../thesis-testing/data"


WORKSPACE="./workspaces/piano_transcription"

# Pack audio files to hdf5 format for training
python3 utils/features.py pack_maestro_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE


# ============ Evaluate (optional) ============
# Inference probability for evaluation

python3 pytorch/calculate_score_for_paper.py infer_prob --workspace=$WORKSPACE --model_type='Note_pedal' --checkpoint_path=$CHECKPOINT_PATH --augmentation='none' --dataset='rousseau' --split='test'

# Calculate metrics
python3 pytorch/calculate_score_for_paper.py calculate_metrics --workspace=$WORKSPACE --model_type='Note_pedal' --augmentation='none' --dataset='rousseau' --split='test'
