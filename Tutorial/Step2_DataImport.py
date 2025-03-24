import torch
import numpy
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset


# Hyperparameters
Max_Nodes = 8300   #This number must be higher than the largest number of cells in each image in the studied dataset.
InputFolderName = "./TNBC_MIBI-TOF_Input/"


## Construct ordinary Python list to hold all input graphs.
Region_filename = InputFolderName + "ImageNameList.txt"
region_name_list = pd.read_csv(
        Region_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["Image"],  # set our own names for the columns
    )

data_list = []
for i in range(0, len(region_name_list)):
    region_name = region_name_list.Image[i]

    # Import network topology.
    EdgeIndex_filename = InputFolderName + region_name + "_EdgeIndex.txt"
    edge_ndarray = numpy.loadtxt(EdgeIndex_filename, dtype = 'int64', delimiter = "\t")
    edge_index = torch.from_numpy(edge_ndarray)

    # Import node attribute.
    NodeAttr_filename = InputFolderName + region_name + "_NodeAttr.txt"
    x_ndarray = numpy.loadtxt(NodeAttr_filename, dtype='float32', delimiter="\t")  #should be float32 not float or float64.
    x = torch.from_numpy(x_ndarray)

    # Import graph index.
    GraphIndex_filename = InputFolderName + region_name + "_GraphIndex.txt"
    graph_index = numpy.loadtxt(GraphIndex_filename, dtype = 'int', delimiter="\t")
    graph_idx = torch.from_numpy(graph_index)
    
    data = Data(x=x, edge_index=edge_index.t().contiguous(), graph_idx=graph_idx)
    data_list.append(data)


## Define "Dataset" class based on ordinary Python list.
class SpatialOmicsImageDataset(InMemoryDataset):                                         
    def __init__(self, root, transform=None, pre_transform=None):
        super(SpatialOmicsImageDataset, self).__init__(root, transform, pre_transform)  
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SpatialOmicsImageDataset.pt']                                           

    def download(self):
        pass
    
    def process(self):
        # Read data_list into huge `Data` list.
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

## Create an object of this "Dataset" class.
dataset = SpatialOmicsImageDataset('./', transform=T.ToDense(Max_Nodes))


