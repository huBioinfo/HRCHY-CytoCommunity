import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGraphConv, dense_mincut_pool
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T

import os
import numpy
import datetime
import csv
import shutil


# Hyperparameters
Max_Nodes = 8300  #This number must be higher than the largest number of cells in each image in the studied dataset.

Num_Run = 20
Num_Epoch = 10000
Num_Fine = 15
Num_Coarse = 2
alpha = 0.9
cut_off = 0.2

Num_Dimension = 128
LearningRate = 0.00001

## Load dataset from constructed Dataset.
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

dataset = SpatialOmicsImageDataset('./', transform=T.ToDense(Max_Nodes))


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=Num_Dimension):
        super(Net, self).__init__()

        self.conv1 = DenseGraphConv(in_channels, hidden_channels)
        num_cluster1 = Num_Fine   #This is a hyperparameter.
        self.pool1 = Linear(hidden_channels, num_cluster1)
        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
        num_cluster2 = Num_Coarse
        self.pool2 = Linear(hidden_channels, num_cluster2)

    def forward(self, x, adj, mask=None):

        x = F.relu(self.conv1(x, adj, mask))
        s = self.pool1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        ClusterAssignTensor_1 = s
        adj = torch.where(adj > cut_off, 1.0, 0.0)
        ClusterAdjTensor_1 = adj

        y = F.relu(self.conv3(x, adj))
        z = self.pool2(y)
        y, adj, mc2, o2 = dense_mincut_pool(y, adj, z, mask=None)

        ClusterAssignTensor_2 = z
        ClusterAdjTensor_2 = adj

        return F.log_softmax(x, dim=-1), mc1, o1, mc2, o2, ClusterAssignTensor_1, ClusterAdjTensor_1, ClusterAssignTensor_2, ClusterAdjTensor_2


def train(epoch):
    model.train()
    loss_all = 0
    loss_1 = 0
    loss_2 = 0
    loss_3 = 0
    loss_4 = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, mc_loss1, o_loss1, mc_loss2, o_loss2, _, _, _, _ = model(data.x, data.adj, data.mask)
        loss = alpha * (mc_loss1 + o_loss1) + (1 - alpha) * (mc_loss2 + o_loss2)
        loss.backward()
        loss_all += loss.item()
        loss_1 += mc_loss1.item()
        loss_2 += o_loss1.item()
        loss_3 += mc_loss2.item()
        loss_4 += o_loss2.item()
        optimizer.step()

    return loss_all, loss_1, loss_2, loss_3, loss_4


normal_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  #Image index
train_index = normal_index[0:1]  #Extract a single graph.

train_dataset = dataset[train_index]
train_loader = DenseDataLoader(train_dataset, batch_size=1)
all_sample_loader = DenseDataLoader(train_dataset, batch_size=1)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

run_number = 1
while run_number <= Num_Run:  #Generate multiple independent runs for ensemble.

    print(f"This is Run{run_number:02d}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, dataset.num_classes).to(device)  #Initializing the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate)
    
    RunFolderName = "Run" + str(run_number)
    os.makedirs(RunFolderName)  #Creating the Run folder.
    filename_0 = RunFolderName + "/Epoch_TrainLoss.csv"
    headers_0 = ["Epoch", "TrainLoss", "MincutLoss1", "OrthoLoss1", "MincutLoss2", "OrthoLoss2"]
    with open(filename_0, "w", newline='') as f0:
        f0_csv = csv.writer(f0)
        f0_csv.writerow(headers_0)

    previous_loss = float("inf")  #Initialization.
    num_epoch = 0
    for epoch in range(1, Num_Epoch+1):  #Specify the number of epoch in each independent run.

        num_epoch = num_epoch + 1

        train_lo = train(epoch)
        train_loss = train_lo[0]
        train_loss1 = train_lo[1]
        train_loss2 = train_lo[2]
        train_loss3 = train_lo[3]
        train_loss4 = train_lo[4]

        with open(filename_0, "a", newline='') as f0:
            f0_csv = csv.writer(f0)
            f0_csv.writerow([epoch, train_loss, train_loss1, train_loss2, train_loss3, train_loss4])
        
        if epoch == 200 and (train_loss4 > 0.7):
            print(f"Type1: {train_loss1:.4f} {train_loss2:.4f} {train_loss3:.4f} {train_loss4:.4f}")
            break

        if epoch == 1000 and (train_loss2 > 1.2 or train_loss4 > 0.7):
            print(f"Type2: {train_loss1:.4f} {train_loss2:.4f} {train_loss3:.4f} {train_loss4:.4f}")
            break

        if epoch == 2000 and (train_loss2 > 1.1 or train_loss4 > 0.7):
            print(f"Type3: {train_loss1:.4f} {train_loss2:.4f} {train_loss3:.4f} {train_loss4:.4f}")
            break

        else:
            previous_loss = train_loss

    if num_epoch < 2500:
        shutil.rmtree(RunFolderName)
        continue

    print(f"Final train loss is {train_loss:.6f}")

    if train_loss >= -0.3  or train_loss4 > 0.2:
        print(f"Type4: {train_loss1:.4f} {train_loss2:.4f} {train_loss3:.4f} {train_loss4:.4f}")
        shutil.rmtree(RunFolderName)
        continue

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    for EachData in all_sample_loader:
        EachData = EachData.to(device)
        TestModelResult = model(EachData.x, EachData.adj, EachData.mask)

        ClusterAssignMatrix1 = TestModelResult[5][0, :, :]
        ClusterAssignMatrix1 = torch.softmax(ClusterAssignMatrix1, dim=-1)
        ClusterAssignMatrix1 = ClusterAssignMatrix1.detach().numpy()
        filename1 = RunFolderName + "/ClusterAssignMatrix_Fine.csv"
        numpy.savetxt(filename1, ClusterAssignMatrix1, delimiter=',')

        ClusterAdjMatrix1 = TestModelResult[6][0, :, :]
        ClusterAdjMatrix1 = ClusterAdjMatrix1.detach().numpy()
        filename2 = RunFolderName + "/ClusterAdjMatrix_Fine.csv"
        numpy.savetxt(filename2, ClusterAdjMatrix1, delimiter=',')

        ClusterAssignMatrix2 = TestModelResult[7][0, :, :]
        ClusterAssignMatrix2 = torch.softmax(ClusterAssignMatrix2, dim=-1)
        ClusterAssignMatrix2 = ClusterAssignMatrix2.detach().numpy()
        filename3 = RunFolderName + "/ClusterAssignMatrix_Coarse.csv"
        numpy.savetxt(filename3, ClusterAssignMatrix2, delimiter=',')

        ClusterAdjMatrix2 = TestModelResult[8][0, :, :]
        ClusterAdjMatrix2 = ClusterAdjMatrix2.detach().numpy()
        filename4 = RunFolderName + "/ClusterAdjMatrix_Coarse.csv"
        numpy.savetxt(filename4, ClusterAdjMatrix2, delimiter=',')

        NodeMask = EachData.mask
        NodeMask = numpy.array(NodeMask)
        filename5 = RunFolderName + "/NodeMask.csv"
        numpy.savetxt(filename5, NodeMask.T, delimiter=',', fmt='%i')

        GraphIdxArray = numpy.array(EachData.graph_idx.view(-1))
        filename6 = RunFolderName + "/GraphIdx.csv"
        numpy.savetxt(filename6, GraphIdxArray, delimiter=',', fmt='%i')

    run_number = run_number + 1

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


