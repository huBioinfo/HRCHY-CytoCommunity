# HRCHY-CytoCommunity



## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Update Log](#update-log)
- [Maintainers](#maintainers)
- [Citation](#citation)


## Overview

<div align=center><img src="https://github.com/wzk610/HRCHY-CytoCommunity/blob/main/support/Schematic.png" width="650" height="900" alt="pipline"/></div>  


Diverse cell types within a tissue assemble into multicellular structures to shape the functions of the tissue. These structural modules typically comprise specialized consist of subunits, each performing unique roles. Uncovering these hierarchical multicellular structures holds significant importance for gaining deep insights into the assembly principles from individual cells to the entire tissue. However, the methods available for identifying hierarchical tissue structures are quite limited and have several limitations as below.

(1) May not be suitable for single-cell-resolution spatial omics data with limited number of gene or protein expression features available. Instead, cell phenotypes typically serve as better initial cell features for learning tissue structures using such data type.

(2) The identified hierarchical tissue structures may not cover all cells within the dataset.

(3) There may not be a clearly nested relationship between the identified different levels of tissue structures.

(4) Cannot correctly identify tissue structures with spatial discontinuous distribution.

Building upon our recently established tissue structure identification framework CytoCommunity (https://github.com/huBioinfo/CytoCommunity), we developed **HRCHY-CytoCommunity**, which utilized a graph neural network (GNN) model to identify hierarchical tissue structures on single-cell spatial maps. HRCHY-CytoCommunity models the identification of hierarchical tissue structures as a MinCut-based hierarchical community detection problem, offering several advantages:

(1) HRCHY-CytoCommunity identifies hierarchical tissue structures from a cellular-based perspective, making it suitable for single-cell-resolution spatial omics data, while ensuring that the hierarchical structures cover all cells within the data.

(2) By leveraging differentiable graph pooling and graph pruning, HRCHY-CytoCommunity is capable of simultaneously identifying tissue structures of various hierarchical levels at multiple resolutions and exhibiting clearly nested relationship between them.

(3) HRCHY-CytoCommunity possesses the ability to discover structures with spatial discontinuous distribution.

(4) HRCHY-CytoCommunity employs a hierarchical majority voting strategy to ensure the robustness of the result, while maintaining the unambiguously nested relationship between the hierarchical tissue structures.

(5) HRCHY-CytoCommunity utilizes an additional cell-type enrichment-based clustering module to generate a unified set of nested multicellular structures across all tissue samples, thereby addressing the issue of cross-sample comparative analysis.


## Installation

### Hardware requirement 

Memory: 16G or more

Storage: 8GB or more

### Software requirement

Conda version: 4.12.0

Python version: 3.10.6

R version: 4.1.3

Clone this repository and cd into it as below.
```
git clone https://github.com/wzk610/HRCHY-CytoCommunity.git
cd HRCHY-CytoCommunity
```
#### For Windows

1. Create a new conda environment using the environment.yml file with the following commands:

    ```bash
    conda env create -f environment.yml
    ```

2. Install R and the diceR package with the following command:

    ```bash
    conda activate HRCHY-CytoCommunity
    R.exe
    > install.packages("diceR")
    ```

#### For Linux

1. Create a new conda environment using the environment_linux.yml file with the following commands:
   
    ```bash
    conda env create -f environment_linux.yml
    ```

2. Install R and the diceR package with the following command:
    
    ```bash
    conda activate HRCHY-CytoCommunity
    conda install R
    R
    > install.packages("diceR")
    ```

The whole installation should take around 20 minutes.

    
## Usage

You can reproduce hierarchical tissue structure assignments of the human triple-negative breast cancer MIBI-TOF dataset shown in the HRCHY-CytoCommunity paper using the commands below. The associated code scripts and example input data can be found under the directory "Tutorial/".

### Prepare input data

The input data includes five types of files:

(1) Cell type label files for each image named "[image name]_CellTypeLabel.txt".

(2) Cell spatial coordinate files for each image named "[image name]_Coordinates.txt".

(3) Index files for each image named "[image name]_GraphIndex.txt".

(4) Node attribute files for each image named "[image name]_NodeAttr.txt".

(5) An image name list file named "ImageNameList.txt".

These example input files can be found under the directory "Tutorial/TNBC_MIBI-TOF_Input/".

### Run the following steps in Windows Powershell or Linux Bash shell:

#### 1. Use Step1 to construct an undirected KNN graph.

```bash
conda activate HRCHY-CytoCommunity
cd Tutorial
python Step1_Construct_KNNgraph.py
```
&ensp;&ensp;**Hyperparameters**
- InputFolderName: The folder name of the input dataset.
- KNN_K: The K value used in the construction of the undirected KNN graph for each image. This value can be empirically set to the integer closest to the square root of the average number of cells in the images in the dataset.

#### 2. Use Step2 to convert the input data to the standard format required by Torch.

```bash
python Step2_DataImport.py
```
&ensp;&ensp;**Hyperparameters**
- Max_Nodes: This number must be higher than the largest number of cells in each image in the dataset.
- InputFolderName: The folder name of the input dataset, consistent with Step1.

#### 3. Use Step3 to perform soft hierarchical tissue structure assignment learning.

```bash
python Step3_HierarchicalTissueStructureLearning.py
```
&ensp;&ensp;**Hyperparameters**
- Max_Nodes: This number must be higher than the largest number of cells in each image in the dataset, consistent with Step2.
- Image_Index: The index of the image which you want to identify hierarchical tissue structures.
- Num_Run: How many times to run the soft hierarchical tissue structure assignment learning module in order to obtain robust results. [Default=20]
- Num_Epoch: The number of training epochs. [Default=10000]
- Num_Fine: The maximum number of fine-grained tissue structures expected to identify.
- Num_Coarse: The maximum number of coarse-grained tissue structures expected to identify.
- Alpha: The weight parameter used to balance fine-grained and coarse-grained loss. [Default=0.9]
- Edge_Pruning: The hyperparameter used to prune the edges of the coarsened graph. [Default=0.2]
- Embedding_Dimension: The dimension of the embedding features. [Default=128]
- Learning_Rate: This parameter determines the step size at each iteration while moving toward a minimum of a loss function. [Default=1E-5]
- Beta: Threshold parameter used to filter results. [Default=0.7 for coarse-grained structures with large scale differences, 0.2 for coarse-grained structures with small scale differences]

#### 4. Use Step4 to perform hierarchical tissue structure ensemble.

```bash
Rscript Step4_HierarchicalTissueStructureEnsemble.R
```
&ensp;&ensp;**Hyperparameters**
- Num_Run: How many times to run the soft hierarchical tissue structure assignment learning module in order to obtain robust results, consistent with Step3. [Default=20]
- Num_Cell:  The number of cells in the image.
- Num_Fine: The maximum number of fine-grained tissue structures expected to identify, consistent with Step3.
- Num_Coarse: The maximum number of coarse-grained tissue structures expected to identify, consistent with Step3.

#### 5. Use Step5 to visualize single-cell spatial maps colored according to cell types and final hierarchical tissue structures.

```bash
python Step5_ResultVisualization.py
```
&ensp;&ensp;**Hyperparameter**
- InputFolderName: The folder name of the input dataset, consistent with Step1.

The total runtime is approximately 12 hours.


## Update Log


## Maintainers
Zekun Wang (22031212461@stu.xidian.edu.cn)

Lin Gao (lgao@mail.xidian.edu.cn)

Yuxuan Hu (huyuxuan@xidian.edu.cn)


## Citation
