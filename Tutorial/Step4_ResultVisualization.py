import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Hyperparameter
InputFolderName = "./MIBI_TNBC_KNNgraph_Input/"

# Import graph index.
GraphIndex_filename = "./Run1/GraphIdx.csv"
graph_index = np.loadtxt(GraphIndex_filename, dtype='int', delimiter="\t")

# Import region name list.
Region_filename = InputFolderName + "ImageNameList.txt"
region_name_list = pd.read_csv(
        Region_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["Image"],  # set our own names for the columns
    )

# Import target graph x/y coordinates.
region_name = region_name_list.Image[graph_index]
GraphCoord_filename = InputFolderName + region_name + "_Coordinates.txt"
x_y_coordinates = pd.read_csv(
        GraphCoord_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["x_coordinate", "y_coordinate"],
    )
target_graph_map = x_y_coordinates


# Import cell type label.
CellType_filename = InputFolderName + region_name + "_CellTypeLabel.txt"
cell_type_label = pd.read_csv(
        CellType_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["cell_type"],  # set our own names for the columns
    )
# Add cell type labels to target graph x/y coordinates.
target_graph_map["CellType"] = cell_type_label.cell_type

#!!! Add consensus cluster labels to target graph x/y coordinates.
target_graph_map["PredictedLabel1"] = np.loadtxt("ConsensusLabel_MajorityVoting_Fine.csv", dtype='int', delimiter=",")
target_graph_map["PredictedLabel2"] = np.loadtxt("ConsensusLabel_MajorityVoting_Coarse.csv", dtype='int', delimiter=",")
# Converting integer list to string list for making color scheme discrete.
target_graph_map.PredictedLabel1 = target_graph_map.PredictedLabel1.astype(str)
target_graph_map.PredictedLabel2 = target_graph_map.PredictedLabel2.astype(str)


#-----------------------------------------Generate plots-------------------------------------------------#
dict_color_Structure1 = {"1": "#5a3b1c", "2": "#939396", "3": "#2c663b", "4": "#d63189", "5": "#54a9dd",
                   "6": "#813188", "7": "Orange", "8": "#231f1f", "9": "#912b61",  "10": "#dc143c",
                   "11": "#ba55d3", "12": "#7b68ee", "13": "#00008b", "14": "#cd853f", "15": "#5f9eA0"}
dict_color_Structure2 = {"1": "#7fc97f", "2": "#beaed4"}

Structure_MajorityVoting_fig1 = sns.lmplot(x="x_coordinate", y="y_coordinate", data=target_graph_map, fit_reg=False, hue='PredictedLabel1', legend=False, palette=dict_color_Structure1, scatter_kws={"s": 10.0})

Structure_MajorityVoting_fig1.set(xticks=[]) #remove ticks and also tick labels.
Structure_MajorityVoting_fig1.set(yticks=[])
Structure_MajorityVoting_fig1.set(xlabel=None) #remove axis label.
Structure_MajorityVoting_fig1.set(ylabel=None)
Structure_MajorityVoting_fig1.despine(left=True, bottom=True) #remove x(bottom) and y(left) axis.


Structure_MajorityVoting_fig1.add_legend(label_order=["1", "2", "3", "4", "5", "6", "7", "8", "9",
                                                "10", "11", "12", "13", "14", "15"])
for lh in Structure_MajorityVoting_fig1._legend.legendHandles:
    #lh.set_alpha(1)
    lh._sizes = [15]   # You can also use lh.set_sizes([15])
#plt.show()
# Save the figure.
Structure_fig_filename1 = "./Fine-grained_Tissue_Structure.pdf"
Structure_MajorityVoting_fig1.savefig(Structure_fig_filename1)


Structure_MajorityVoting_fig2 = sns.lmplot(x="x_coordinate", y="y_coordinate", data=target_graph_map, fit_reg=False, hue='PredictedLabel2', legend=False, palette=dict_color_Structure2, scatter_kws={"s": 10.0})

Structure_MajorityVoting_fig2.set(xticks=[]) #remove ticks and also tick labels.
Structure_MajorityVoting_fig2.set(yticks=[])
Structure_MajorityVoting_fig2.set(xlabel=None) #remove axis label.
Structure_MajorityVoting_fig2.set(ylabel=None)
Structure_MajorityVoting_fig2.despine(left=True, bottom=True) #remove x(bottom) and y(left) axis.


Structure_MajorityVoting_fig2.add_legend(label_order = ["1", "2"])
for lh in Structure_MajorityVoting_fig2._legend.legendHandles:
    #lh.set_alpha(1)
    lh._sizes = [15]   # You can also use lh.set_sizes([15])
#plt.show()
# Save the figure.
Structure_fig_filename2 = "./Coarse-grained_Tissue_Structure.pdf"
Structure_MajorityVoting_fig2.savefig(Structure_fig_filename2)


# Plot x/y map with "CellType" coloring.
dict_color_CellType = {"CD4T": "#fee08b", "B": "Red", "DC": "Black", "CD8T": "MediumBlue", "CD11c-high": "Purple", "MF_1": "#00A087",
                       "MF/Glia": "#1F77B4", "NK": "#a50026", "Treg": "#FF7F0E", "Other": "#9467BD", "MF_2": "#2CA02C",
                       "Neutrophil": "#8C564B", "Epithelial": "#E377C2", "Mesenchymal/SMA": "#7F7F7F", "Tumor/Keratin": "#543005", "Tumor/EGFR": "#BCBD22",
                       "Endothelial/Vim": "#17BECF"}
CellType_fig = sns.lmplot(x="x_coordinate", y="y_coordinate", data=target_graph_map, fit_reg=False, hue='CellType', legend=False, palette=dict_color_CellType, scatter_kws={"s": 10.0})

CellType_fig.set(xticks=[]) #remove ticks and also tick labels.
CellType_fig.set(yticks=[])
CellType_fig.set(xlabel=None) #remove axis label.
CellType_fig.set(ylabel=None)
CellType_fig.despine(left=True, bottom=True) #remove x(bottom) and y(left) axis.

CellType_fig.add_legend(label_order = ["CD4T", "B", "DC", "CD8T", "CD11c-high", "MF_1",
                                       "MF/Glia", "NK", "Treg", "Other", "MF_2",
                                       "Neutrophil", "Epithelial", "Mesenchymal/SMA", "Tumor/Keratin", "Tumor/EGFR", "Endothelial/Vim"])
for lh in CellType_fig._legend.legendHandles:
    #lh.set_alpha(1)
    lh._sizes = [15]   # You can also use lh.set_sizes([15])
# Save the figure.
CellType_fig_filename = "./CellType.pdf"
CellType_fig.savefig(CellType_fig_filename)


# Export dataframe: "target_graph_map".
TargetGraph_dataframe_filename = "./TargetGraphDF.csv"
target_graph_map.to_csv(TargetGraph_dataframe_filename, na_rep="NULL", index=False) #remove row index.


