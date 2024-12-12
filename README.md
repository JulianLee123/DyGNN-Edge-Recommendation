# [Connecting Text with Graph Structure: Deep Learning Algorithms for Textural Relation Generation on Dynamic Text-Attributed Graphs](https://www.overleaf.com/read/wbrkbjvfktqx#e90573)

## Usage

The code was run on GPUs through Google Colab. The proposed models can be run as follows:
1. In the project's root folder, create a directory named 'Code' to store this repo, and create an empty directory named 'Processed_Datasets'.
2. Download the desired dataset from the Dynamic Text-Attributed Graph Benchmark release [here](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk?usp=sharing). Store the downloaded datasets in the 'Processed_Datasets' folder. 
2. To train and test the supervised dynamic graph algorithm, run 'Run_DTGB_Train_Edge_Retrieval.ipynb' in the 'Code' folder. This notebook also includes data analysis.
3. To train and test the unsupervised dynamic graph algorithm, run 'Run_DTGB_Train_Edge_Classification.ipynb' in the 'Code' folder. 

## Acknowledgements
This project was advised by PhD student Jialin Chen and Professor Rex Ying. Datasets are pulled from the [DGTB paper](https://arxiv.org/abs/2406.12072).
