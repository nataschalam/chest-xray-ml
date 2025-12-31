# Chest X-Ray Disease Classification with VGG16
This repository contains a PyTorch-based deep learning pipeline for binary disease detection from chest X-ray (CXR) images using transfer learning with VGG16. The model classifies X-ray images into: (1) No Finding (healthy) or (2) Disease present (any pathology)

The project includes dataset preprocessing, stratified data splitting, model training with early stopping, comprehensive evaluation metrics, and visual diagnostics.

## Data
- This project uses a freely-accessible chest x-ray dataset provided by the NIH Clinical Center: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Please refer to the following paper for more information about the dataset and benchmark performances on the dataset: https://arxiv.org/abs/1705.02315
- The ChestX-ray dataset consists of 112,120 frontal chest X-ray images from 30,805 unique patients
- Each image is annotated with up to 14 disease labels, which include: Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Mass, and Hernia.

## Methodology / Project Overview
1. Download, extract and create merged chest X-ray dataset
2. Compute dataset-specific normalisation statistics (mean & standard deviation)
3. Apply data augmentation and preprocessing
4. Train a VGG16-based binary classifier
5. Validate using multiple performance metrics (accuracy, precision, F1-score, recall, ROC-AUC, FPR, FNR)
6. Evaluate the final model on a held-out test set
7. Save trained weights, metrics, and plots

## Outputs & Visualisations
The training script automatically saves:

Model
- Best-performing model weights (based on validation accuracy)

Plots
- Training vs validation accuracy
- Training vs validation loss
- Confusion matrix (test set)
- ROC curve (test set)

CSV Files
- Final validation metrics
- Per-epoch training & validation metrics
- Test set performance metrics

## Results / Key Findings
- The trained VGG-16 model achieves an accuracy of 0.71, AUC of 0.76, precision of 0.68, F1-Score of 0.69.
- In a real-world NHS setting, low precision would be unsustainable and would likely drive wait-times in A&E
- Future improvements need to focus on better label curation and making a balanced dataset to future explore the potential of multi-label classification 

## Future Improvements
- Optimise decision threshold (default is 0.5) to improve recall (lower false negative/capture more true positive)
- K-fold cross validation on subset of dataset

## How to Run Analysis
1. Run the script called: `image_download.ipynb`
2. Run the script called `gz_extraction.ipynb`
3. Run the script called `creating_merged_dataset.ipynb`
4. Run the script called `vgg_model.ipynb`
