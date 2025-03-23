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

(1) Using gene expression features as input, requiring a large number of features, which may not be applicable to single-cell spatial omics data.

(2) The identified hierarchical tissue structures may not cover all cells within the dataset.

(3) There may not be a clearly nested relationship between the identified different levels of structures.

(4) Cannot correctly identify tissue structures with spatial discontinuous distribution.

Building upon the recently established tissue structure identification framework CytoCommunity (https://github.com/huBioinfo/CytoCommunity), we developed **HRCHY-CytoCommunity**, which utilized a graph neural network (GNN) model to identify hierarchical tissue structures on single-cell spatial maps. HRCHY-CytoCommunity models the identification of hierarchical tissue structures as a MinCut-based hierarchical community detection problem, offering several advantages:

(1) HRCHY-CytoCommunity identifies hierarchical tissue structures from a cellular-based perspective, making it suitable for single-cell spatial omics data with a limited number of features, while ensuring that the hierarchical structures cover all cells within the data.

(2) By leveraging differentiable graph pooling and graph pruning, HRCHY-CytoCommunity is capable of simultaneously identifying tissue structures of various hierarchical levels at multiple resolutions and exhibiting clear nested relationship between them.

(3) HRCHY-CytoCommunity possesses the ability to discover structures with spatial discontinuous distribution.

(4) HRCHY-CytoCommunity employs a hierarchical majority voting strategy to ensure the robustness of the result, while maintaining the unambiguously nested relationship between the hierarchical tissue structures.

(5) HRCHY-CytoCommunity utilizes an additional cell-type enrichment-based clustering module to generate a unified set of nested multicellular structures across all tissue samples, thereby addressing the issue of cross-sample comparative analysis.



