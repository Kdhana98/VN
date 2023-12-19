import os
import random
import re
import shutil

import numpy as np
import numpy.random
import pandas as pd
import plyfile
import torch
# from plyfile import PlyData
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from pyntcloud import PyntCloud
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from torch.utils.data import Dataset

torch.backends.cudnn.deterministic = True


def sort_key(string):
    # Extract numeric portion of the string using regular expression
    numeric_part = re.search(r'\d+', string).group()
    # Convert the numeric part to integer for comparison
    return int(numeric_part)


# choose among [3, 7, 11, 13, 29]
seed = 29
# set seed for reproducibility
random.seed(seed)
# select among random, ellipse, random
visualization = 'circular'
# select between medium and small
model_size = 'medium'
# select between random or uniform
coloring = 'uniform'
# select  among 1, 2, or 3 for the train/val splits of 80/20, 160/40, or 800/200
tv = 3
node_size = 0.5
edge_width = 0.1
vertical = 1
resolution = 0.35

if tv == 1:
    train_size = 80
    val_size = 20
if tv == 2:
    train_size = 160
    val_size = 40
if tv == 3:
    train_size = 800
    val_size = 200

test_size = 500
size = 224
resize = False
params = None

if 'ellipse' in visualization:
    if params != None:
        params = str(vertical).replace('.', 'p') + '_' + params
        visualization = str(vertical).replace('.', 'p') + '_' + visualization
    else:
        params = str(vertical).replace('.', 'p') + '_' + coloring + "_" + visualization + "_" + str(
            model_size) + "_" + str(seed) + '_' + str(
            train_size) + "_" + str(test_size) + "_" + str(val_size)
        visualization = str(vertical).replace('.', 'p') + '_' + coloring + '_color_' + visualization

if node_size != 0.5:
    if params != None:
        params = 'node_size_' + str(node_size).replace('.', 'p') + '_' + params
        visualization = 'node_size_' + str(node_size).replace('.', 'p') + '_' + visualization
    else:
        params = 'node_size_' + str(node_size).replace('.', 'p') + '_' + coloring + "_" + visualization + "_" + str(
            model_size) + "_" + str(seed) + '_' + str(
            train_size) + "_" + str(test_size) + "_" + str(val_size)
        visualization = 'node_size_' + str(node_size).replace('.', 'p') + '_' + coloring + '_color_' + visualization

if edge_width != 0.1:
    if node_size == 0.5 and params == None:
        params = 'edge_width_' + str(edge_width).replace('.', 'p') + '_' + coloring + "_" + visualization + "_" + str(
            model_size) + "_" + str(
            seed) + '_' + str(
            train_size) + "_" + str(test_size) + "_" + str(val_size)
        visualization = 'edge_width_' + str(edge_width).replace('.', 'p') + '_' + coloring + '_color_' + visualization
    else:
        params = 'edge_width_' + str(edge_width).replace('.', 'p') + '_' + params
        visualization = 'edge_width_' + str(edge_width).replace('.', 'p') + '_' + visualization

if visualization == 'spiral':

    params = str(resolution).replace('.', 'p') + '_sp_' + coloring + "_" + visualization + "_" + str(
        model_size) + "_" + str(
        seed) + '_' + str(
        train_size) + "_" + str(test_size) + "_" + str(val_size)
    visualization = str(resolution).replace('.', 'p') + '_sp_' + coloring + '_color_' + visualization

elif params == None:
    params = coloring + "_" + visualization + "_" + str(model_size) + "_" + str(seed) + '_' + str(
        train_size) + "_" + str(test_size) + "_" + str(val_size)
    visualization = coloring + '_color_' + visualization
# params = visualization+"_"+str(model_size)+"_"+str(seed)+'_'+str(train_size)+"_"+str(test_size)+"_"+str(val_size)

# Define pointcloud directories
plce_graphs_non_hamiltonian_dir = 'Point_cloud_' + visualization + '_non_hamiltonian_' + str(model_size) + '/'
plce_graphs_hamiltonian_dir = 'Point_cloud_' + visualization + '_hamiltonian_' + str(model_size) + '/'
random.seed(seed)
all_point_Clouds = os.listdir(plce_graphs_non_hamiltonian_dir) + os.listdir((plce_graphs_hamiltonian_dir))
all_point_Clouds = sorted(all_point_Clouds, key=sort_key)
random.shuffle(all_point_Clouds)
pcloud_graph_data = []
label = {}

for file in all_point_Clouds:
    if file.endswith(".ply"):
        if 'non_hamiltonian' in file:
            label[file] = 0
            class_label = "non_hamiltonian"
            ply_data = plyfile.PlyData.read(os.path.join(plce_graphs_non_hamiltonian_dir, file))
            cloud = PyntCloud.from_file(os.path.join(plce_graphs_non_hamiltonian_dir, file))
        else:
            label[file] = 1
            class_label = "hamiltonian"
            ply_data = plyfile.PlyData.read(os.path.join(plce_graphs_hamiltonian_dir, file))
            cloud = PyntCloud.from_file(os.path.join(plce_graphs_hamiltonian_dir, file))
        vertices = ply_data['vertex']

        graph_points_coord = [[vertex['x'], vertex['y'], vertex['z'], vertex['Node']] for vertex in vertices]
        pcloud_graph_data.append((graph_points_coord, label[file]))

# Create train, validation, and test directories

os.makedirs('PointCloud_' + visualization + '/data_' + str(params) + '/train/hamiltonian', exist_ok=True)
os.makedirs('PointCloud_' + visualization + '/data_' + str(params) + '/val/hamiltonian', exist_ok=True)
os.makedirs('PointCloud_' + visualization + '/data_' + str(params) + '/test/hamiltonian', exist_ok=True)

os.makedirs('PointCloud_' + visualization + '/data_' + str(params) + '/train/non_hamiltonian', exist_ok=True)
os.makedirs('PointCloud_' + visualization + '/data_' + str(params) + '/val/non_hamiltonian', exist_ok=True)
os.makedirs('PointCloud_' + visualization + '/data_' + str(params) + '/test/non_hamiltonian', exist_ok=True)

if os.path.exists('data_' + str(params) + '_saved_models'):
    # Remove the directory and all its contents
    shutil.rmtree('data_' + str(params) + '_saved_models')

# Create the new directory
os.makedirs('PointCloud_' + visualization + '/data_' + str(params) + '_saved_models', exist_ok=True)
# Split the sampled images into train, validation, and test sets
random.shuffle(all_point_Clouds)

for i, file in enumerate(all_point_Clouds[:train_size]):
    try:
        shutil.copy(os.path.join(plce_graphs_non_hamiltonian_dir, file),
                    os.path.join('PointCloud_' + visualization + '/data_' + str(params) + '/train/non_hamiltonian',
                                 file))
    except:
        shutil.copy(os.path.join(plce_graphs_hamiltonian_dir, file),
                    os.path.join('PointCloud_' + visualization + '/data_' + str(params) + '/train/hamiltonian', file))
for i, file in enumerate(all_point_Clouds[train_size:train_size + val_size]):
    try:
        shutil.copy(os.path.join(plce_graphs_non_hamiltonian_dir, file),
                    os.path.join('PointCloud_' + visualization + '/data_' + str(params) + '/val/non_hamiltonian', file))
    except:
        shutil.copy(os.path.join(plce_graphs_hamiltonian_dir, file),
                    os.path.join('PointCloud_' + visualization + '/data_' + str(params) + '/val/hamiltonian', file))
for i, file in enumerate(all_point_Clouds[train_size + val_size:train_size + val_size + test_size]):
    try:
        shutil.copy(os.path.join(plce_graphs_non_hamiltonian_dir, file),
                    os.path.join('PointCloud_' + visualization + '/data_' + str(params) + '/test/non_hamiltonian',
                                 file))
    except:
        shutil.copy(os.path.join(plce_graphs_hamiltonian_dir, file),
                    os.path.join('PointCloud_' + visualization + '/data_' + str(params) + '/test/hamiltonian', file))

seed_model = 23
torch.manual_seed(seed_model)
torch.cuda.manual_seed(seed_model)
numpy.random.seed(seed_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        k = x
        # Input transformation
        trans = self.input_transform(x)
        x = torch.bmm(trans, x)
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = torch.relu(x)

        # Feature transformation
        trans = self.feature_transform(k)
        x = torch.bmm(trans, k)
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Initialize the PointNet model
PointnetModel = PointNetCls(num_classes=2)
PointnetModel.to(device)
# Print the model architecture
print(PointnetModel)

if resize == True:
    # Set up data loaders
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
if coloring == 'gray':
    transform = transforms.Compose([
        transforms.Grayscale(),
        # transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
else:
    transform = transforms.Compose([
        # transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

classes = ["hamiltonian", "non_hamiltonian"]
pcloud_graph_data1 = []


class PointCloudDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        # self.data = data  # List of point clouds
        # self.labels = labels  # List of corresponding labels
        self.transform = transform
        for each in classes:
            plce_graphs = os.path.join(data, each)
            data_dir = os.listdir(plce_graphs)
            for point_cloud_file in data_dir:
                ply_data = plyfile.PlyData.read(os.path.join(plce_graphs, point_cloud_file))
                vertices = ply_data['vertex']
                graph_points = [[vertex['x'], vertex['y'], vertex['z'], vertex['Node']] for vertex in vertices]
                pcloud_graph_data1.append((graph_points, label[point_cloud_file]))
                self.file = graph_points
                self.label = labels[point_cloud_file]

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
    point_cloud_tensor = torch.zeros(batch_size, 4, num_points)

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
        point_cloud_tensor[i, :len(selected_points), :] = torch.tensor(selected_points)
    # Convert the labels to a tensor
    label_tensor = torch.tensor(labels)

    return point_cloud_tensor, label_tensor


PC_train_dataset = PointCloudDataset('PointCloud_' + visualization + '/data_' + str(params) + '/train/', label)
PC_val_dataset = PointCloudDataset('PointCloud_' + visualization + '/data_' + str(params) + '/val/', label)
PC_test_dataset = PointCloudDataset('PointCloud_' + visualization + '/data_' + str(params) + '/test/', label)

batch_size = 32

PC_train_loader = torch.utils.data.DataLoader(PC_train_dataset, batch_size=batch_size, shuffle=True,
                                              collate_fn=lambda batch: custom_collate_fn1(batch, num_points=512))
PC_val_loader = torch.utils.data.DataLoader(PC_val_dataset, batch_size=batch_size, shuffle=False,
                                            collate_fn=lambda batch: custom_collate_fn1(batch, num_points=512))
PC_test_loader = torch.utils.data.DataLoader(PC_test_dataset, batch_size=batch_size, shuffle=False,
                                             collate_fn=lambda batch: custom_collate_fn1(batch, num_points=512))
# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(PointnetModel.parameters(), lr=0.01)

# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.09)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training loop
num_epochs = 100  # Adjust as needed
best_val_f1 = 0.0
early_stopping_patience = 8
early_stopping_counter = 0
metrics_df = pd.DataFrame(columns=['Epoch', 'AUC', 'Accuracy', 'F1', 'Recall'])

print("Started Training")

# Training loop
for epoch in range(num_epochs):
    epoch = epoch + 1
    print(f"Epoch {epoch}/{num_epochs}")
    print('-' * 10)
    # Training phase
    PointnetModel.train()
    train_loss = 0.0
    train_correct = 0

    for i, data in enumerate(PC_train_loader):
        inputs, labels1 = data
        inputs, labels1 = inputs.to(device), labels1.to(device)
        optimizer.zero_grad()

        outputs = PointnetModel(inputs)
        loss = criterion(outputs, labels1)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_loss += loss.item() * len(inputs[1])
        train_correct += torch.sum(preds == labels1)

    train_loss = train_loss / len(PC_train_dataset)
    train_acc = train_correct.double() / len(PC_train_dataset)
    # Save the model after each epoch
    model_path = os.path.join('PointCloud_' + visualization + '/data_' + str(params) + '_saved_models',
                              f'model_epoch_{epoch}.pt')
    torch.save(PointnetModel.state_dict(), model_path)
    # Validation phase
    PointnetModel.eval()
    val_loss = 0.0
    val_correct = 0

    y_true_list = []
    y_pred_list = []
    y_pred_prob_list = []

    with torch.no_grad():
        for i, data in enumerate(PC_val_loader):
            inputs, labels1 = data
            inputs, labels1 = inputs.to(device), labels1.to(device)

            outputs = PointnetModel(inputs)
            loss = criterion(outputs, labels1)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * len(inputs[1])
            val_correct += torch.sum(preds == labels1)

            labels_np = labels1.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # Append true labels and predicted probabilities to the lists
            y_true_list.extend(labels_np.tolist())
            y_pred_list.extend(preds_np.tolist())
            y_pred_prob_list.extend(outputs[:, 1].cpu().detach().numpy().tolist())

        val_loss = val_loss / len(PC_val_dataset)
        val_acc = val_correct.double() / len(PC_val_dataset)
        val_acc = val_acc.cpu().numpy()
        # Calculate metrics
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        y_pred_prob = np.array(y_pred_prob_list)
        # y_scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y_true, y_pred_prob)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        # Print training and validation metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} |f1: {f1:.4f} | AUC: {auc:.4f} | Recall: {recall:.4f}")

        # Check for early stopping
        if f1 > best_val_f1:
            best_val_f1 = f1
            early_stopping_counter = 0

            # Save the best model
            torch.save(PointnetModel.state_dict(), 'PC_best_model_' + str(params) + '.pt')

            print("Best model saved!")

        else:
            early_stopping_counter += 1

        # Add metrics to the dataframe
        metrics_df.loc[epoch] = [epoch, auc, val_acc, f1, recall]

        # Check if early stopping criteria are met
        if early_stopping_counter >= early_stopping_patience:
            if best_val_f1 == 0:
                torch.save(PointnetModel.state_dict(), 'PC_best_model_' + str(params) + '.pt')
                # continue
            print("Early stopping!")
            break
    scheduler.step()

# Testing Accuracy
PointnetModel.eval()
running_loss = 0.0
y_pred1 = 0
total_samples = 0
y_true_list1 = []
y_pred_list1 = []
y_pred_prob_list1 = []
# test loop
for i, data in enumerate(PC_test_loader):
    inputs, labels1 = data
    inputs, labels1 = inputs.to(device), labels1.to(device)
    optimizer.zero_grad()

    # Forward pass
    outputs: object = PointnetModel(inputs)
    loss = criterion(outputs, labels1.data)

    # Backpropagation and optimization
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    _, predicted = torch.max(outputs, 1)
    y_pred1 += torch.sum(predicted == labels1.data)
    total_samples += labels1.size(0)

    # Convert labels and predictions to numpy arrays
    labels_np = labels1.cpu().numpy()
    preds_np = predicted.cpu().numpy()

    # Append true labels and predicted probabilities to the lists
    y_true_list1.extend(labels_np.tolist())
    y_pred_list1.extend(preds_np.tolist())
    y_pred_prob_list1.extend(outputs[:, 1].cpu().detach().numpy().tolist())

# Convert the lists to numpy arrays
y_true_test = np.array(y_true_list1)
y_pred_test = np.array(y_pred_list1)
y_pred_prob_test = np.array(y_pred_prob_list1)

test_acc = y_pred1.double() / len(PC_test_dataset)
test_acc = test_acc.cpu().numpy()
# y_true = test_dataset.targets
# y_scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
auc = roc_auc_score(y_true_test, y_pred_prob_test)
# print("test",y_true_test,y_pred_test)
f1 = f1_score(y_true_test, y_pred_test)
recall = recall_score(y_true_test, y_pred_test)
# metrics_df.loc[epoch] = [epoch, auc, test_acc, f1, recall]
print(f"Test Acc: {test_acc:.4f} | AUC: {auc:.4f} | f1: {f1} | Recall: {recall:4f}")
print(params)
