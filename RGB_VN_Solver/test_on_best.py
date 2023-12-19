import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from torch.utils.data import Dataset

print("start best")
# from train import PointnetModel,PC_test_loader,PC_test_dataset
import random

import plyfile

torch.backends.cudnn.deterministic = True
visualization = 'circular'
coloring = 'uniform'
model_size = 'medium'
tv = 3
node_size = 0.5
edge_width = 0.1
df_time = pd.DataFrame(columns = ['run', 'time'])
vertical = 1
resolution = 0.35
if tv == 1:
    tn_s = 80
    val_s = 20
if tv == 2:
    tn_s = 160
    val_s = 40
if tv == 3:
    tn_s = 800
    val_s = 200

tt_s = 500

test_size = 'small'
resize = False
test_dir = None
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.DataFrame(columns=['seed', 'AUC', 'Acc', 'F1', 'Recall'])
# visualization= coloring + '_color_' + visualization
plce_graphs_non_hamiltonian_dir = 'Point_cloud_' + coloring + '_color_' + visualization+'_non_hamiltonian_'+str(model_size)+'/'
plce_graphs_hamiltonian_dir = 'Point_cloud_' + coloring + '_color_' + visualization+'_hamiltonian_'+str(model_size)+'/'
# random.seed(seed)
all_point_Clouds = os.listdir(plce_graphs_non_hamiltonian_dir)+os.listdir(plce_graphs_hamiltonian_dir)
# all_point_Clouds = sorted(all_point_Clouds, key=sort_key)
random.shuffle(all_point_Clouds)
pcloud_graph_data = []
label = {}
print("lengths",len(all_point_Clouds),len(os.listdir(plce_graphs_non_hamiltonian_dir)),len(os.listdir(plce_graphs_hamiltonian_dir)))
for file in all_point_Clouds:
    if file.endswith(".ply"):
        if 'non_hamiltonian' in file:
            label[file]=0
        else:
            label[file]=1

for s in [3, 7, 11, 13, 29]:
    # test_dir=None
    if resize == True:
        test_dir = visualization + '/data_transformed_' + visualization + '_' + model_size + '_' + str(s) + '_' + str(
            tn_s) + '_' + str(tt_s) + '_' + str(val_s) + '/test'

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    else:
        if visualization == 'ellipse':
            test_dir = str(vertical).replace('.', 'p') + '_' + coloring + '_color_' + visualization + '/data_' + str(
                vertical).replace('.', 'p') + '_' + coloring + '_' + visualization + '_' + model_size + '_' + str(
                s) + '_' + str(tn_s) + '_' + str(tt_s) + '_' + str(val_s) + '/test'

        if node_size != 0.5:
            test_dir = 'node_size_'+ str(node_size).replace('.', 'p') +'_' + coloring + '_color_' + visualization + '/data_node_size_'+ str(node_size).replace('.', 'p') +'_' + coloring + '_' + visualization + '_' + model_size + '_' + str(
                s) + '_' + str(tn_s) + '_' + str(tt_s) + '_' + str(val_s) + '/test'
        if visualization == 'spiral':
            test_dir = str(resolution).replace('.', 'p') + '_sp_' + coloring + '_color_' + visualization + '/data_' + str(resolution).replace('.', 'p') + '_sp_' + coloring + '_' + visualization + '_' + model_size + '_' + str(
                s) + '_' + str(tn_s) + '_' + str(tt_s) + '_' + str(val_s) + '/test'

        if test_dir == None:
            test_dir = 'PointCloud_' + coloring + '_color_' + visualization + '/data_' + coloring + '_' + visualization + '_' + model_size + '_' + str(s) + '_' + str(tn_s) + '_' + str(tt_s) + '_' + str(val_s) + '/test'
        if coloring == 'gray':
            transform = transforms.Compose([
                transforms.Grayscale(),
                # transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    batch_size = 32
    classes = ["hamiltonian", "non_hamiltonian"]
    pcloud_graph_data1 = []

    class PointCloudDataset(Dataset):
        def __init__(self, data, labels,transform=None):
            # self.data = data  # List of point clouds
            # self.labels = labels  # List of corresponding labels
            self.transform = transform
            for each in classes:
                plce_graphs = os.path.join(data,each)
                data_dir =os.listdir(plce_graphs)
                for point_cloud_file in data_dir:
                    ply_data = plyfile.PlyData.read(os.path.join(plce_graphs,point_cloud_file))
                    vertices = ply_data['vertex']
                    graph_points = [[vertex['x'], vertex['y'], vertex['z'], vertex['Node']] for vertex in vertices]
                    pcloud_graph_data1.append((graph_points,label[point_cloud_file]))
                    self.file = graph_points
                    self.label= labels[point_cloud_file]
        def __len__(self):
            return len(pcloud_graph_data1)

        def __getitem__(self, idx):
            point_cloud = pcloud_graph_data1[idx][0]
            label = pcloud_graph_data1[idx][1]
            if self.transform:
                point_cloud = self.transform(point_cloud)
            return point_cloud, label

    def custom_collate_fn1(batch, num_points=64):
        # Extract the point clouds and labels from the batch
        point_clouds, labels = zip(*batch)

        # Create a tensor to store the point clouds in the batch
        batch_size = len(batch)
        # num_features = len(point_clouds[0][0][0])  # Assuming all point clouds have the same number of features
        point_cloud_tensor = torch.zeros(batch_size, 4,num_points)

        # Iterate over the point clouds in the batch and make sure each has num_points
        for i in range(batch_size):
            # Select num_points randomly from the original point cloud
            original_points = point_clouds[i]
            if len(original_points) < num_points:
                selected_points = original_points + [[0.0] * 4] * (num_points - len(original_points))
            else:
                selected_points = point_clouds[i][:num_points]
            selected_points = np.array(selected_points).T

            # Fill the tensor with the selected points
            point_cloud_tensor[i,:len(selected_points), : ] = torch.tensor(selected_points)
        # Convert the labels to a tensor
        label_tensor = torch.tensor(labels)

        return point_cloud_tensor, label_tensor

    PC_test_dataset = PointCloudDataset(test_dir, label)
    PC_test_loader = torch.utils.data.DataLoader(PC_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: custom_collate_fn1(batch, num_points=1024))

    # POINTNET model
    class TNet(nn.Module):
        def __init__(self):
            super(TNet, self).__init__()
            self.conv1 = nn.Conv1d(4, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 16)  # 4x4 matrix for transformation

        def forward(self, x):
            batch_size = x.size(0)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)

            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)

            # Add an identity matrix to the transformation matrix
            # identity = torch.eye(3, dtype=x.dtype, device=x.device).view(1, 9).repeat(batch_size, 1)
            # x = x + identity

            x = x.view(-1, 4, 4)
            return x

    # PointNet classification module
    class PointNetCls(nn.Module):
        def __init__(self, num_classes=2):
            super(PointNetCls, self).__init__()
            self.input_transform = TNet()
            self.feature_transform = TNet()
            self.conv1 = nn.Conv1d(4,64,1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            batch_size = x.size(0)
            k=x
            # Input transformation
            trans = self.input_transform(x)
            x = torch.bmm(trans,x)
            x=self.conv1(x)
            x = torch.relu(x)

            x=self.conv2(x)
            x = torch.relu(x)

            # Feature transformation
            trans = self.feature_transform(k)
            x = torch.bmm(trans,k)
            x=self.conv1(x)
            x = torch.relu(x)

            x=self.conv2(x)
            x = torch.relu(x)
            x = torch.relu(self.conv3(x))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)

            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)

            return x

    # Initialize the PointNet model
    best_PointnetModel = PointNetCls(num_classes=2)
    PointnetModel = nn.DataParallel(best_PointnetModel)  # Utilize multiple GPUs
    PointnetModel = PointnetModel.to(device)
    # print("pointnet",PointnetModel)
    if resize == True:
        PointnetModel.load_state_dict(torch.load(
            'best_model_transformed_' + visualization + '_' + model_size + '_' + str(s) + '_' + str(tn_s) + '_' + str(tt_s) + '_' + str(
                val_s) + '.pt'))
    else:
        if node_size != 0.5:

            PointnetModel.load_state_dict(torch.load(
                    'best_model_node_size_'+ str(node_size).replace('.', 'p') +'_' + coloring + '_' + visualization + '_' + model_size + '_' + str(
                        s) + '_' + str(tn_s) + '_' + str(tt_s) + '_' + str(val_s) + '.pt'))

        elif visualization == 'ellipse':
            PointnetModel.load_state_dict(torch.load(
                'best_model_'+ str(vertical).replace('.', 'p') + '_' + coloring + '_' + visualization + '_' + model_size + '_' + str(
                    s) + '_' + str(
                    tn_s) + '_' + str(tt_s) + '_' + str(val_s) + '.pt'))

        elif visualization == 'spiral':
            PointnetModel.load_state_dict(torch.load(
                'best_model_' + str(resolution).replace('.', 'p') + '_sp_' + coloring + '_' + visualization + '_' + model_size + '_' + str(
                    s) + '_' + str(tn_s) + '_' + str(tt_s) + '_' + str(val_s) + '.pt'))


        else:
            state_dir = torch.load(
                'PC_best_model_' + coloring + '_' + visualization + '_' + model_size + '_' + str(s) + '_' + str(
                    tn_s) + '_' + str(tt_s) + '_' + str(val_s) + '.pt')
            PointnetModel.load_state_dict(state_dir, strict=False)
    # Evaluate the best PointnetModel on the test set
    PointnetModel.eval()
    test_correct = 0
    i = 0
    with torch.no_grad():
        all_preds = []
        all_labels = []
        test_correct = 0
        tp = 0
        fn = 0
        tt = 0
        for i, data in enumerate(PC_test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            ts = time.time()
            outputs = PointnetModel(inputs)

            probs = torch.softmax(outputs, dim=1)  # Obtain probabilities using softmax
            te = time.time()
            td = te - ts
            new_row_time = pd.DataFrame({'run': [i],
                            'time': [float(td)]})
            df_time = df_time._append(new_row_time, ignore_index=True)
            tt += td
            _, preds = torch.max(outputs, 1)
            all_preds.extend(probs[:, 1].cpu().numpy())  # Use the probability of the positive class

            all_labels.extend(labels.cpu().numpy())
            test_correct += torch.sum(preds == labels.data)
            tp += torch.sum((preds == 1) & (labels == 1))
            fn += torch.sum((preds == 0) & (labels == 1))
        # print("len",PC_test_dataset,test_correct,"test","lables",all_labels,"preds",all_preds,"tp",tp,"fn",fn)
        print(len(PC_test_dataset))
        test_acc = test_correct.double() / len(PC_test_dataset)
        auc = roc_auc_score(all_labels, all_preds)
        f1 = f1_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
        recall = tp.double() / (tp.double() + fn.double())
        test_acc = test_acc.cpu()
        recall = recall.cpu()

        new_row = pd.Series({'seed': s, 'AUC': auc, 'Acc': test_acc, 'F1': f1, 'Recall': recall})

        # Add row using loc indexer
        df.loc[len(df.index)] = [s, auc, test_acc, f1, recall]

        print(f"seed:{s}, Test Acc: {test_acc:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print (f'Time: {tt},tv:{tv}')
df_time.to_csv('time_pointnet.csv')

statistics = df.agg(['mean', 'std'])

# Print the mean and standard deviation of each column
for column in df.columns:
    mean_value = round(statistics.loc['mean', column], 2)
    std_value = round(statistics.loc['std', column], 2)
    print(f"Column '{column}':")
    print(f"  Mean ± Std: {mean_value} ± {std_value}\n")
