#####################################################################################
########################  FED-CVaR-AVG  FashionMnist ################################
import json 
import time 
from tqdm import trange
import sys
import ray
import os
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.datasets import VisionDataset
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import math
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader , Dataset
from torch.utils.tensorboard import SummaryWriter


############################################################################################################################
############################################################################################################################
##### Define the hyperparameters
# Learning rate for theta
learning_rate = 0.01
# Learning rate for t
learning_rate_t = 0.0005  
# Batch Size
batch_size = 128
# Number of users
K = 30
# Chooce the percentage of users that will take exclusively some patterns (according to the paper is the quantity (100-r)% of users )
percentage =   K * 0.1 # Example: move the last  (K * 0.1 ) % of least values
# Patterns at the most frequent users
label_subset_1 = [0,1,2,3,4,5,6,7]
# Patterns at the less frequent users
label_subset_2 = [8,9]
# Number of classes in the dataset
number_classes = len(label_subset_1+label_subset_2)
# Number of global epochs
num_epochs = 6000
# Number of local epochs
H = 20 
# Factor b helps to track the users the server see (it is not important for the algorithm)
beta = 0.9
# Number of users each time the server lets to connect to the server
users_per_round = 1
# Value of CVaR parameter a
cvar_alpha_list = [0.2] #[ 1, 0.3, 0.2, 0.1 ]
# Value of gamma parameter 
gamma_list = [ 0. ]  #[  0.1 , 0. ]         #[0.3, 0.2] # [1. , 0.5, 0.4 ] #, 0.1 , 0. ]
# How many time we repeat the experiment
Repeat_experiment = [0,1,2,3,4]
# initialize the list of t
t_final_list = []
# Hidden size of CNN
hidden_size=(120, 84)
# Conv size of CNN
conv_size =(6,16)
# Kernel size of CNN
kernel_size = 5 # 3 , # 5
# padding size of CNN
padding=  2 #1 , # 2

## (Both two lines below are useful only for fast results)
# Number of classes we reduce 
reduce_classes = (0,1,2,3,4,5,6,7,8,9)
# Percentage of the reduced class
reduce_to_ratio = 1.


############################################################################################################################
############################################################################################################################

############################################################################################################################
############################################################################################################################
##################### Generate user probabilities based on which RAM will let each user to connect to the server #########################
# Make an array with the  users id
users = np.arange(0, K)
# Number of less probable users (according to the paper is the quantity (100-M)% of users )
k = int( percentage )   

# Set randmomly probabilities to users 
probabilities = np.random.rand(K)
# Be sure that these probabiblities have sum=1
probabilities = probabilities / probabilities.sum()
# Convert the array of user probabilities to tensor
user_probs = torch.from_numpy(probabilities).float()


#### Plot the bar chart showing how the distribution of users
# plt.bar(users, user_probs, align='center', color='blue', alpha=0.6)
# plt.xlabel('Users')
# plt.ylabel('Frequency')
# plt.xticks(users)
# plt.title('Distribution of Users')
# plt.show()

########### Use torch.topk() to find the 10% less often users #######
# Find the less k probable users
smallest_values, indices = torch.topk(user_probs, k=k, largest=False)

# Print the 10 smallest values and their indices
# print("Smallest values:", smallest_values)
# print("Indices:", indices)
# Original list

user_probs = user_probs.tolist()
# Calculate the number of elements to move
num_elements = int(len(user_probs) * percentage)

# Sort the list in ascending order
sorted_list =  sorted(user_probs, reverse=True)

# Get the last a% of least values
values_to_move = sorted_list[:num_elements]

# Create a new list by excluding the values to move
user_probs = [x for x in user_probs if x not in values_to_move]

# Append the values to move at the end of the new list
user_probs.extend(values_to_move)

#### Plot the bar chart showing how the distribution of users in a descending order
# plt.bar(users, user_probs, align='center', color='blue', alpha=0.6)
# plt.xlabel('Users')
# plt.ylabel('Frequency')
# plt.xticks(users)
# plt.title('Distribution of Users')
# plt.show()

# Check if gpu is available, else we use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)
############################################################################################################################
############################################################################################################################

############################################################################################################################
############################################################################################################################
# create imbalance for some patterns ( This used only for fast experiments, not important for the algorithm)
def create_imbalance(dataset, reduce_classes=(0,), reduce_to_ratio=.2):
    data = dataset.data
    label = dataset.targets

    reduce_mask = torch.zeros(data.shape[0], dtype=torch.bool)
    for reduce_class in reduce_classes:
        reduce_mask = torch.logical_or(reduce_mask, label == reduce_class)
    preserve_mask = torch.logical_not(reduce_mask)

    label_reduce = label[reduce_mask]
    len_reduce = label_reduce.shape[0]
    label_reduce = label_reduce[:max(1, int(len_reduce * reduce_to_ratio))]
    label_preserve = label[preserve_mask]

    label = torch.cat([label_reduce, label_preserve], dim=0)

    preserve_mask_np = preserve_mask.numpy()
    reduce_mask_np = reduce_mask.numpy()

    data_reduce = data[reduce_mask_np]
    data_reduce = data_reduce[:max(1, int(len_reduce * reduce_to_ratio))]
    data_preserve = data[preserve_mask_np]

    data = np.concatenate([data_reduce, data_preserve], axis=0)

    remain_len = label.shape[0]

    rand_index = torch.randperm(remain_len)
    rand_index_np = rand_index.numpy()

    dataset.data = data[rand_index_np]
    dataset.targets = label[rand_index]

    return dataset
################################################################################################################################################
################################################################################################################################################

################################################################################################################################################
################################################################################################################################################
# pick Fashion Mnist dataset
dataset = "fashion-mnist"
dataset_train = datasets.FashionMNIST(root='datasets/' + dataset, download=True, transform= transforms.ToTensor())
dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets))
dataset_test = datasets.FashionMNIST(root='datasets/' + dataset, train=False, download=True ,transform= transforms.ToTensor() )
dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
n_classes = 10
n_channels = 1
img_size = 28

# ( This used only for fast experiments, not important for the algorithm, if you need faster experiments pick reduce_to_ratio -> 0. )
imb_dataset_train =  create_imbalance(dataset_train, reduce_classes=reduce_classes, reduce_to_ratio=reduce_to_ratio)
# counts = torch.bincount(imb_dataset_train.targets)
# print(counts)

# Create the subset dataset based on the patterns that most often users will have
idx_subset_train_1 = torch.where(torch.isin(dataset_train.targets, torch.tensor(label_subset_1)))
dataset_train_subset_1= datasets.FashionMNIST(root='datasets/' + dataset, train=True, download=True ,transform= transforms.ToTensor() )
dataset_train_subset_1.targets = imb_dataset_train.targets[idx_subset_train_1]
dataset_train_subset_1.data = imb_dataset_train.data[idx_subset_train_1]

# Create the subset dataset based on the patterns that less often users will have
idx_subset_train_2 = torch.where(torch.isin(dataset_train.targets, torch.tensor(label_subset_2)))
dataset_train_subset_2= datasets.FashionMNIST(root='datasets/' + dataset, train=False, download=True ,transform= transforms.ToTensor() )
dataset_train_subset_2.targets = imb_dataset_train.targets[idx_subset_train_2]
dataset_train_subset_2.data = imb_dataset_train.data[idx_subset_train_2]


# Function for Splitting data to users
def split_data_to_users(dataset, n_workers, homo_ratio):
    data = dataset.data
    data = data.numpy() if torch.is_tensor(data) is True else data
    label = dataset.targets

    n_data = data.shape[0]

    n_homo_data = int(n_data * homo_ratio)

    n_homo_data = n_homo_data - n_homo_data % n_workers
    n_data = n_data - n_data % n_workers

    if n_homo_data > 0:
        data_homo, label_homo = data[0:n_homo_data], label[0:n_homo_data]
        data_homo_list, label_homo_list = np.split(data_homo, n_workers), label_homo.chunk(n_workers)

    if n_homo_data < n_data:
        data_hetero, label_hetero = data[n_homo_data:n_data], label[n_homo_data:n_data]
        label_hetero_sorted, index = torch.sort(label_hetero)
        data_hetero_sorted = data_hetero[index]

        data_hetero_list, label_hetero_list = np.split(data_hetero_sorted, n_workers), label_hetero_sorted.chunk(
            n_workers)

    if 0 < n_homo_data < n_data:
        data_list = [np.concatenate([data_homo, data_hetero], axis=0) for data_homo, data_hetero in
                        zip(data_homo_list, data_hetero_list)]
        label_list = [torch.cat([label_homo, label_hetero], dim=0) for label_homo, label_hetero in
                        zip(label_homo_list, label_hetero_list)]
    elif n_homo_data < n_data:
        data_list = data_hetero_list
        label_list = label_hetero_list
    else:
        data_list = data_homo_list
        label_list = label_homo_list
    
    return data_list , label_list

# Split Data to the most often users
data_list_subset_1, label_list_subset_1 = split_data_to_users(dataset = dataset_train_subset_1 , n_workers = K - k , homo_ratio=1.)
# Split Data to the less often users
data_list_subset_2, label_list_subset_2 = split_data_to_users(dataset = dataset_train_subset_2 , n_workers = k , homo_ratio=1.)

### Store the data of each user into lists
data_users ,labels_users = [], []

# Store the input X
data_users = [ torch.tensor( data_list_subset_1[i] ) for i in range(K - k)  ]
data_users.extend(  torch.tensor(data_list_subset_2[i])  for i in range(k)  )

# Store the labels Y
labels_users = [ label_list_subset_1[i].clone().detach() for i in range(K - k)]
labels_users.extend(  label_list_subset_2[i].clone().detach()  for i in range(k) )

#########################################################################################################
#########################################################################################################
###### For testing at training set #######
general_train_dataset = dataset_train_subset_1 + dataset_train_subset_2

x_general_train_tensor_1 = torch.tensor(dataset_train_subset_1.data)
y_general_train_tensor_1 = dataset_train_subset_1.targets


x_general_train_tensor_2 = torch.tensor(dataset_train_subset_2.data)
y_general_train_tensor_2 = dataset_train_subset_2.targets


x_general_train_tensor_1_2 = torch.cat([x_general_train_tensor_1, x_general_train_tensor_2], dim=0)
y_general_train_tensor_1_2 = torch.cat([y_general_train_tensor_1, y_general_train_tensor_2], dim=0)

general_train_dataset = TensorDataset(x_general_train_tensor_1_2,y_general_train_tensor_1_2)

general_train_loader_subset = DataLoader(general_train_dataset, batch_size=batch_size, shuffle=True)
################################################################################################################################################
################################################################################################################################################

################################################################################################################################################
################################################################################################################################################
###### For testing at test set #######
idx_subset_test = torch.where(torch.isin(dataset_test.targets, torch.tensor(label_subset_1 + label_subset_2)))
dataset_test_subset= datasets.FashionMNIST(root='datasets/' + dataset, train=False, download=True ,transform= transforms.ToTensor()  )


dataset_test_subset = create_imbalance(dataset_test_subset, reduce_classes = reduce_classes , reduce_to_ratio = reduce_to_ratio  )

dataset_test_subset.targets = dataset_test.targets[idx_subset_test]
dataset_test_subset.data = dataset_test.data[idx_subset_test]

x_test_tensor = dataset_test_subset.data.clone().detach()
y_test_tensor = dataset_test_subset.targets

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset_test_subset, batch_size=batch_size, shuffle=False)
#########################################################################################################
#########################################################################################################

#########################################################################################################
#########################################################################################################
######## Genearate trainloaders ##########
users_dataloaders = []
for user in range(K):
    x_train_tensor = data_users[user].clone().detach().float() / 255.0 # Normalize input data #data_users[user].clone().detach().float()
    y_train_tensor = labels_users[user].clone().detach().long()
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )
    users_dataloaders.append(train_loader)
################################################################################################################################################
################################################################################################################################################


################################################################################################################################################
################################################################################################################################################
#### Define the RAM model, it takes as input the K users with the corresponding probabilities, and the number of the allowed users at each global round
def RAM(K,user_probs,users_per_round):
    user_index = random.choices(range(K), weights=user_probs, k=users_per_round)[0]
    return user_index 
################################################################################################################################################
################################################################################################################################################


################################################################################################################################################
################################################################################################################################################
####### Define the Convolutional Neural Network
    
class CNNFashion_Mnist(nn.Module):
    def __init__(self, n_class, conv_size,kernel_size , padding):
        super(CNNFashion_Mnist, self).__init__()
        self.activation = F.leaky_relu
        self.layer1 = nn.Conv2d(1, conv_size[0], kernel_size=kernel_size, padding=padding)
        self.layer2 = nn.Conv2d(conv_size[0] ,conv_size[1], kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2)     
        self.fc1 = nn.Linear(conv_size[1] * 7 * 7, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], n_class)
    
    def forward(self, x):
        x = self.pool( self.activation( self.layer1(x) ) )
        x = self.pool( self.activation( self.layer2(x) ) )
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
################################################################################################################################################
################################################################################################################################################

################################################################################################################################################
################################################################################################################################################
###### Define the basic training function 

# @ray.remote(num_gpus=0.285,max_calls=0)
def train_Fashion_mnist(user, global_model , device , global_t, H , train_loader):
    # For each user 
    # Copy the global model transmitted by the server
    f_local = copy.deepcopy(global_model)
    f_local.requires_grad_(True)
    
    # define optimizer for theta
    optimizer = optim.SGD(f_local.parameters(), lr = learning_rate )
    # convert t into tensor
    global_t = global_t.clone().detach().requires_grad_(True)
    # define optimizer for t
    optimizer_t = torch.optim.SGD([global_t], lr=learning_rate_t  )

    # for each local epoch
    for _ in range(H):
        # Pass through each batch of data
        for i, (batch_x, batch_y) in enumerate(train_loader):
            # Pass each batch through the device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Forward pass
            output = f_local(batch_x.unsqueeze(1))
            # loss empirical loss
            train_loss = criterion( output , batch_y )
            # generalized risk-aware local loss 
            cvar_loss = (1-gamma) * global_t + ( ( (1-gamma)/cvar_alpha ) * torch.relu(train_loss - global_t) ) + gamma * train_loss
            # Backward pass and optimization for theta and t
            optimizer.zero_grad()
            optimizer_t.zero_grad()
            cvar_loss.backward(retain_graph = True)
            optimizer.step()
            optimizer_t.step()
    # Each user returns the to RAM the updated theta and t (essential for the algorithm), it also send the local training loss, for evaluation reasons (not essential for the algorithm)
    return f_local.state_dict() ,global_t.item(), train_loss.item()
################################################################################################################################################
################################################################################################################################################



################################################################################################################################################
################################################################################################################################################
########## Starting the training process ##########
# For extensive experiments we repeat each experiment as the lenght of: Repeat_experiment
for repeat in Repeat_experiment:
    # pick parameter gamma
    gamma = gamma_list[0]
    # pick parameter a of CVaR
    cvar_alpha = cvar_alpha_list[0]
    
    ############################################################################################################################
    ############################################################################################################################
    # Store the value of hyper-parameters into dictionary
    config = { 'learning_rate': learning_rate ,
            'learning_rate_t' : learning_rate_t,
            'batch_size': batch_size,
            'number_classes' : number_classes ,
            'K' : K,
            'label_subset_1' : label_subset_1,
            'label_subset_2' : label_subset_2,
            'num_epochs' : num_epochs,
            'H' : H , 
            'beta' : beta,
            'user_probs' : user_probs, 
            'users_per_round' : users_per_round,
            'gamma' : gamma , 
            'cvar_alpha_list' : cvar_alpha_list,
            'cvar_alpha' : cvar_alpha,
            'repeat' : repeat
            }
    
    # Store the experiment setup
    # define the path to store
    path = ""
    experiment_setup = f"{path}_[{len(label_subset_1 + label_subset_2)}]_classes_users_[{K}]/cvar_[{cvar_alpha}]/_gamma_[{gamma}]_learning_rate_[{learning_rate}]_learning_rate_t_[{learning_rate_t}]_local_epochs_[{H}]_batch_size_[{batch_size}]_percentage_[{percentage}]_fast_example_reduce_to_ratio_[{reduce_to_ratio}]_conv_size_[{conv_size}]_kernel_size_[{kernel_size}]_padding_[{padding}]_2_classes_repeat_[{repeat}]"    
    save_dir = 'simulations/%s' % (experiment_setup)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #################################################################
    #################################################################
    # The steps are useeful for plotting (Not essential for the algorithm)
    # Retrieve stored checkpoint if necessary (Not essential for the algorithm)
    checkpoint_model_PATH = f'checkpoint_model_cvar_[{cvar_alpha}]/%s'  % (experiment_setup)

    if not os.path.exists(checkpoint_model_PATH):
        os.makedirs(checkpoint_model_PATH)

    tb_file = save_dir + f'/{time.time()}'
    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)

    ##### Plot apriori users' probabilities #####
    plt.bar(users, user_probs, align='center', color='blue', alpha=0.6)
    plt.savefig(f'bar_plot_cvar_[{cvar_alpha}].png')

    image = plt.imread(f'bar_plot_cvar_[{cvar_alpha}].png')
    image = np.uint8(image * 255)  # Convert image to uint8
    image = np.transpose(image, (2, 0, 1))  # Transpose dimensions to CHW format
    writer.add_image(f'bar_plot_cvar_[{cvar_alpha}]', image)
    ############################################################################################################################
    ############################################################################################################################

    ############################################################################################################################
    ############################################################################################################################

    # Call the model
    global_model = CNNFashion_Mnist(n_class = number_classes, conv_size = conv_size , kernel_size= kernel_size , padding= padding)
    
    # Pass the model through the device
    global_model.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    ############################################################################################################################
    ############################################################################################################################

    ############################################################################################################################
    ############################################################################################################################
    # Initialize the parameters theta and t
    # Initialize the parameter theta 
    global_theta = global_model.state_dict()
    # Initialize the parameter t 
    global_t = torch.tensor(0. ,requires_grad=True)

    # Make lists for the users thetas
    user_weights = [ global_theta.copy() for _ in range(K) ] 
    # Make lists for the users t
    t_list = [ torch.tensor(  0.  , requires_grad=True) for _ in range(K)  ]
    train_loss_list = [ global_t for _ in range(K) ]
    train_loss_list_users = [ 0 for i in range(K)]
    
    ##############################################
    # The next two commants are important only for plotting (Not essential for the algorithm) 
    sampling_counts, a_0  = [0] * K , 1/K
    a_i = [0] * K
    ##############################################

    # Initialize the necessary lists
    train_losses = []
    test_losses = []
    test_accuracies = []
    t_list_track = []
    ############################################################################################################################
    ############################################################################################################################

    ############################################################################################################################
    ############################################################################################################################
    ### Start the training loop
    # ray.init(log_to_driver=False)
    # For each global epoch
    for epoch in trange(1,num_epochs+1):
        # Set the global model into training mode
        global_model.train()

        # Server Ask from RaM to return a user id, and its parameter
        user_index  = RAM(K,user_probs,users_per_round)

        # The server, based on the allowed user_index, braodcasts to the users the parameters theta and t of the allowed user_index and they start the local training
        # After finishing the training the users send to the RAM their updates
        global_theta , global_t , train_loss_list = train_Fashion_mnist( user_index, global_model, device, global_t, H , users_dataloaders[user_index] ) 

        #################################################################
        #################################################################
        # Update the appropriate lists in order to visualize the parameters 
        global_t = torch.tensor(global_t,requires_grad=True)
        t_list_track.append(global_t.item())
        writer.add_scalar("Global t ", global_t.item(), epoch)
        
        ###########################################################################
        #################### Counter of the user Availability #####################
        sampling_counts[user_index] += 1
        n_i = sampling_counts[user_index]
        n = epoch
        fraction_n_i_n = n_i / n if n != 0 else 1.0

        for i in range(K):
            a_i[i] = (1 - beta) * a_0 + beta * sampling_counts[i]/n
        
        if round(sum(a_i)) != 1:
            print("fraction_n_i_n",fraction_n_i_n)
            print("a_i:",a_i)
            sys.exit("List a_i has not sum equal to 1")
        
        # calculate the percentages
        total = sum(a_i)
        percentages = [(f / total) * 100 for f in a_i]

        # create a pie chart using matplotlib
        labels = ['User {}'.format(i) for i in range(len(a_i))]
        fig, ax = plt.subplots()
        ax.pie(percentages, labels=labels, autopct='%1.1f%%')

        # log the pie chart to Tensorboard
        writer.add_figure('Pie Chart', fig)
        #################################################################
        #################################################################
        
        
        ###########################################################################
        ###########################################################################
        # update the global model 
        global_model.load_state_dict(global_theta)

        ###########################################################################
        ###########################################################################
        # The next steps help for evaluating the algorithm the training and testing loss (the next steps are not necessary for the algorithm)
        train_loss_list_users = []
        for i in range(K):
            for x,y in users_dataloaders[i]:
                x, y = x.to(device), y.to(device)

                output = global_model(x.unsqueeze(1))
                f_i = criterion(output, y)
            train_loss_list_users.append(f_i.item())
        
        for i in range(K):
            writer.add_scalar(f"Training loss vs global epoch/user_{i}", train_loss_list_users[i], epoch)
        
        
        test_loss = 0
        correct = 0.
        with torch.no_grad():
            global_model.eval() 
            test_loss , correct , n_data = 0. , 0. , 0.
            for test_batch_x, test_batch_y in test_loader:
                test_batch_x, test_batch_y = test_batch_x.to(device), test_batch_y.to(device)

                # y_pred_test = global_model(test_batch_x.unsqueeze(1))
                y_pred_test = global_model(test_batch_x)
                test_loss += criterion(y_pred_test, test_batch_y)
                pred = y_pred_test.data.max(1, keepdim=True)[1]
                correct += pred.eq(test_batch_y.data.view_as(pred)).sum()

            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            writer.add_scalar("Overall Test Loss", test_loss, epoch)
            test_accuracies.append((100. * correct / len(test_loader.dataset) ).item())
            writer.add_scalar("Test Accuracy", (100. * correct / len(test_loader.dataset) ).item() , epoch)

            # Compute the accuracy for the test data with label i
            correct_hist = torch.zeros(number_classes).to(device)
            label_hist = torch.zeros(number_classes).to(device)
            for data, label in test_loader:
                data, label = data.to(device) , label.to(device)
                
                label_hist += torch.histc(label, number_classes, max=number_classes)
                f_data = global_model(data)
                pred = f_data.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_index = pred.eq(label.view_as(pred)).squeeze()
                label_correct = label[correct_index]
                correct_hist += torch.histc(label_correct, number_classes, max=number_classes)

            correct_rate_hist = correct_hist / label_hist
            test_accuracy = [correct_rate_hist.cpu().numpy()][0] 
            # print("t_accuracy:",t_accuracy)

            for i in range(number_classes):
                writer.add_scalar(f"class-wise correct rate vs global epoch/test/class_{i}", test_accuracy[i], epoch)

            for general_train_data, general_train_label in general_train_loader_subset:
                general_train_data, general_train_label = general_train_data.to(device) , general_train_label.to(device)
                general_train_data = general_train_data.float()

                f_data = global_model(general_train_data.unsqueeze(1))
                general_train_loss = criterion( f_data , general_train_label )
            
            general_train_loss /= len(general_train_loader_subset.dataset)
            writer.add_scalar("Overall Training Loss", general_train_loss, epoch)
            ###########################################################################
            ###########################################################################
            
######################################################################################################################################################
######################################################################################################################################################
# To visualize the experiments you should use tensorboard.
# The commands below can help
## tensorboard --logdir tensorboard --logdir simulations/ the name of the path you have defined above /
