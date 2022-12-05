#!/usr/bin/env python
# coding: utf-8

# # Assignment Summary

# **Denoising autoencoders**: We will evaluate denoising autoencoders applied to the MNIST dataset.
# 
# * Obtain (or write! but this isn't required) a pytorch/tensorflow/etc. code for a denoising autoencoder. Train this autoencoder on the MNIST dataset. Use only the MNIST training set. You should use at least three layers in the encoder and in the decoder.
# * We now need to determine how well this autoencoder works. For each image in the MNIST test dataset, compute the residual error of the autoencoder. This is the difference between the true image and the reconstruction of that image by the autoencoder. It is an image itself. Prepare a figure showing the mean residual error, and the first five principal components. Each is an image. You should preserve signs (i.e. the mean residual error may have negative as well as positive entries). The way to show these images most informatively is to use a mid gray value for zero, then darker values for more negative image values and lighter values for more positive values. The scale you choose matters. You should show
#     * mean and five principal components on the same gray scale for all six images, chosen so the largest absolute value over all six images is full dark or full light respectively and
#     * mean and five principal components on a scale where the gray scale is chosen for each image separately.
# 
# **Variational autoencoders**: We will evaluate variational autoencoders applied to the MNIST dataset.
#   * Obtain (or write! but this isn't required) a pytorch/tensorflow/etc. code for a variational autoencoder. Train this autoencoder on the MNIST dataset. Use only the MNIST training set.
#   * We now need to determine how well the codes produced by this autoencoder can be interpolated.
#     * For 10 pairs of MNIST test images of the same digit, selected at random, compute the code for each image of the pair. Now compute 7 evenly spaced linear interpolates between these codes, and decode the result into images. Prepare a figure showing this interpolate. Lay out the figure so each interpolate is a row. On the left of the row is the first test image; then the interpolate closest to it; etc; to the last test image. You should have a 10 rows and 9 columns of images.
#     * For 10 pairs of MNIST test images of different digits, selected at random, compute the code for each image of the pair. Now compute 7 evenly spaced linear interpolates between these codes, and decode the result into images. Prepare a figure showing this interpolate. Lay out the figure so each interpolate is a row. On the left of the row is the first test image; then the interpolate closest to it; etc; to the last test image. You should have a 10 rows and 9 columns of images.

# **Hints and References**: For the denoising autoencoder, there is an abundance of code online should you choose to obtain one. It may be a good practice to also implement this part from scratch and test what you learned in the CNN assignment. All you have to do is define a network with two groups of layers:
#   * *Encoder Layers*: This part must take an image and produce a low-dimensional "code" of the image. Therefore, the architecture of the netwok must be narrowing down. Let's call this function $f^{\text{encoder}}$.
#   * *Decoder Layers*: This part must take a low-dimensional "code" of the image and produce the original image. Therefore, the architecture of the netwok must be expanding. Let's call this function $f^{\text{decoder}}$.
#   
# All you have to do is to try and write some code to minimize the following loss:
# 
# $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \|x_i - f^{\text{decoder}}(f^{\text{encoder}}(x_i))\|_2^2$$
# 
# You may pick any architecture that works as long as it has three layes. The MNIST data has 784 pixels. Therefore, a fully connected network which takes 784 reshaped dimensions to $h_1$ dimensions, then to $h_2$ dimensions, and finally to $h_3$ dimensions is an excellent starting point for an encoder. A vast range of choices can work for these three numbers, but just to give you an idea about their plausible range of values, $h_1$ could be in the order of hundreds, $h_2$ could be in order of tens (or at most a few hundreds), and $h_3$ is supposed to be a low-dimension (preferrably under 10 or at most 20).
# 
# You can reverse the encoder architecture, to obtain a decoder, and then stack an SGD optimizer on top with default hyper-parameters to train your denoising autoencoder. You must be familiar with the rest of the concepts from earlier assignments such as multi-dimensional scalings and PCA. You also would need to write some basic code to visualize using matplotlib, PIL, etc.
# 
# For VAEs, you may also be able to implement everything from scratch once you review the material. However, there are a lot of resources and examples for implementing VAEs, and here we share a few of them:
# 
#   1. Pytorch Tutorials has an example for training VAEs at https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py
#   2. Another pytorch example for VAEs can be found at https://github.com/pytorch/examples/blob/master/vae/main.py
#   3. Pyro is a library for bayesian optimization and is based on pytorch, which has a detailed tutorial on how to train VAEs with some high-level story of the math involved https://pyro.ai/examples/vae.html
#   4. BoTorch is another bayesian optimization library based based on pytorch and has some tutorials for implementing VAEs https://botorch.org/tutorials/vae_mnist
#   5. If you're a tensorflow fan, you may find some tutorial at https://www.tensorflow.org/probability/examples/Probabilistic_Layers_VAE or  https://www.tensorflow.org/tutorials/generative/cvae
#   6. Keras fans can also see https://keras.io/examples/generative/vae/
#   7. etc.
#   
# The MNIST data is provided at `../VAE-lib/data_mnist` so that you could use the `torchvision` API for loading the data just like the previous assignment.

# **Important Note**: This assignment will not be automatically graded and is optional. Therefore, do not expect meaninful grades to be published upon or after submission. However, please make sure to submit your work if you expect it to be reviewed by the instructors for any reason. We will consider the latest submission of your work. 
# 
# Any work that is not submitted will not be viewed by us.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from tensorflow import keras
import numpy as np
import torch
import botorch
import pyro
import matplotlib.pyplot as plt


# In[ ]:





# # Denoising autoencoders

# In[2]:


import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch
import torch.optim as optim
from torch.autograd import Variable


# In[3]:



# Load the dataset, add gaussian,poisson,speckle
'''

    'gauss'     Gaussian-distributed additive noise.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.

'''
    
# Define a function that adds each noise when called from main function

def add_noise(img,noise_type="gaussian"):
  
  row,col=28,28
  img=img.astype(np.float32)
  
  if noise_type=="gaussian":
    mean=0
    var=10
    sigma=var**.5
    noise=np.random.normal(-5.9,5.9,img.shape)
    noise=noise.reshape(row,col)
    img=img+noise
    return img

  if noise_type=="speckle":
    noise=np.random.randn(row,col)
    noise=noise.reshape(row,col)
    img=img+img*noise
    return img


# In[4]:


# Load the dataset from keras
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
print("Train dataset:{}\nTest dataset:{}".format(len(xtrain),len(xtest)))


# In[5]:


# Split the 60k train dataset into 3 sets each given one type of each noise.
# Shuffle for better generalization
# https://sofiadutta.github.io/datascience-ipynbs/pytorch/Denoising-Autoencoder.html
# https://www.analyticsvidhya.com/blog/2021/07/image-denoising-using-autoencoders-a-beginners-guide-to-deep-learning-project/
# https://medium.com/@connectwithghosh/denoising-images-using-an-autoencoder-using-tensorflow-in-python-1e2e62932837
# https://medium.com/@connectwithghosh/simple-autoencoder-example-using-tensorflow-in-python-on-the-fashion-mnist-dataset-eee63b8ed9f1


noises=["gaussian","speckle"]
noise_ct=0
noise_id=0
traindata=np.zeros((60000,28,28))

for idx in tqdm(range(len(xtrain))):
  
  if noise_ct<(len(xtrain)/2):
    noise_ct+=1
    traindata[idx]=add_noise(xtrain[idx],noise_type=noises[noise_id])
    
  else:
    print("\n{} noise addition completed to images".format(noises[noise_id]))
    noise_id+=1
    noise_ct=0

print("\n{} noise addition completed to images".format(noises[noise_id])) 


noise_ct=0
noise_id=0
testdata=np.zeros((10000,28,28))

for idx in tqdm(range(len(xtest))):
  
  if noise_ct<(len(xtest)/2):
    noise_ct+=1
    x=add_noise(xtest[idx],noise_type=noises[noise_id])
    testdata[idx]=x
    
  else:
    print("\n{} noise addition completed to images".format(noises[noise_id]))
    noise_id+=1
    noise_ct=0


print("\n{} noise addition completed to images".format(noises[noise_id]))    


# In[6]:


# visualize the noisy images along with their original versions

f, axes=plt.subplots(2,2)

#showing images with gaussian noise
axes[0,0].imshow(xtrain[0],cmap="gray")
axes[0,0].set_title("Original Image")
axes[1,0].imshow(traindata[0],cmap='gray')
axes[1,0].set_title("Noised Image")

#showing images with speckle noise
axes[0,1].imshow(xtrain[25000],cmap='gray')
axes[0,1].set_title("Original Image")
axes[1,1].imshow(traindata[25000],cmap="gray")
axes[1,1].set_title("Noised Image")


# In[7]:


# Create dataset which includes both clean and noisy images


class noisedDataset(Dataset):
  
  def __init__(self,datasetnoised,datasetclean,labels,transform):
    self.noise=datasetnoised
    self.clean=datasetclean
    self.labels=labels
    self.transform=transform
  
  def __len__(self):
    return len(self.noise)
  
  def __getitem__(self,idx):
    xNoise=self.noise[idx]
    xClean=self.clean[idx]
    y=self.labels[idx]
    
    if self.transform != None:
      xNoise=self.transform(xNoise)
      xClean=self.transform(xClean)
      
    
    return (xNoise,xClean,y)


# In[8]:


tsfms=transforms.Compose([
    transforms.ToTensor()
])

trainset=noisedDataset(traindata,xtrain,ytrain,tsfms)
testset=noisedDataset(testdata,xtest,ytest,tsfms)


# In[9]:


# Create the trainloaders and testloaders.
# Transform the images using standard lib functions

batch_size=32

trainloader=DataLoader(trainset,batch_size=32,shuffle=True)
testloader=DataLoader(testset,batch_size=1,shuffle=True)


# In[10]:


# Define the autoencoder model.


class denoising_model(nn.Module):
  def __init__(self):
    super(denoising_model,self).__init__()
    self.encoder=nn.Sequential(
                  nn.Linear(28*28,256),
                  nn.ReLU(True),
                  nn.Linear(256,128),
                  nn.ReLU(True),
                  nn.Linear(128,64),
                  nn.ReLU(True)
        
                  )
    
    self.decoder=nn.Sequential(
                  nn.Linear(64,128),
                  nn.ReLU(True),
                  nn.Linear(128,256),
                  nn.ReLU(True),
                  nn.Linear(256,28*28),
                  nn.Sigmoid(),
                  )
    
 
  def forward(self,x):
    x=self.encoder(x)
    x=self.decoder(x)
    
    return x


# In[11]:


# Check whether cuda is available and choose device accordingly
if torch.cuda.is_available()==True:
  device="cuda:0"
else:
  device ="cpu"

  
model=denoising_model().to(device)
criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,weight_decay=1e-5)


epochs=120
l=len(trainloader)
losslist=list()
epochloss=0
running_loss=0
for epoch in range(epochs):
  
  print("Entering Epoch: ",epoch)
  for dirty,clean,label in tqdm((trainloader)):
    
    
    dirty=dirty.view(dirty.size(0),-1).type(torch.FloatTensor)
    clean=clean.view(clean.size(0),-1).type(torch.FloatTensor)
    dirty,clean=dirty.to(device),clean.to(device)
    
    #Forward Pass
    output=model(dirty)
    loss=criterion(output,clean)
    #Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    running_loss+=loss.item()
    epochloss+=loss.item()
  #Log
  losslist.append(running_loss/l)
  running_loss=0
                
print('Finished Training')


# In[12]:


plt.plot(range(len(losslist)),losslist)


# In[13]:



# Visualize some of the results
# Randomly generate 6 numbers in between 1 and 10k, run them through the model, and show the results with comparisons


f,axes= plt.subplots(6,3,figsize=(20,20))
axes[0,0].set_title("Original Image")
axes[0,1].set_title("Dirty Image")
axes[0,2].set_title("Cleaned Image")

test_imgs=np.random.randint(0,10000,size=6)
for idx in range((6)):
  dirty=testset[test_imgs[idx]][0]
  clean=testset[test_imgs[idx]][1]
  label=testset[test_imgs[idx]][2]
  dirty=dirty.view(dirty.size(0),-1).type(torch.FloatTensor)
  dirty=dirty.to(device)
  output=model(dirty)
  
  output=output.view(1,28,28)
  output=output.permute(1,2,0).squeeze(2)
  output=output.detach().cpu().numpy()
  
  dirty=dirty.view(1,28,28)
  dirty=dirty.permute(1,2,0).squeeze(2)
  dirty=dirty.detach().cpu().numpy()
  
  clean=clean.permute(1,2,0).squeeze(2)
  clean=clean.detach().cpu().numpy()
  
  axes[idx,0].imshow(clean,cmap="gray")
  axes[idx,1].imshow(dirty,cmap="gray")
  axes[idx,2].imshow(output,cmap="gray")
  


# In[ ]:





# # Variational autoencoders

# In[14]:


import torch
torch.rand(5, 3)


# In[15]:


# Load dataset

from torchvision import datasets, transforms

root = '../VAE-lib/data_mnist'

transformations = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root = root, train = True, download = True, transform = transformations)
mnist_test = datasets.MNIST(root = root, train = False, download = True, transform = transformations)


# In[16]:


from torch.utils.data import DataLoader

batch_size = 32
train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=True)


# In[17]:


from torch import nn
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
    
        self.fc1 = nn.Linear(784, 400)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim) 
        
    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        h3 = self.relu(self.fc3(x))
        return self.sigmoid(self.fc4(h3))


# In[18]:


def sample(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    
    return mu + eps*std


# In[19]:


def vae_loss(x, x_hat, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(x_hat, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


# In[20]:


from torch import optim
latent_dim = 32
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)
params = list(encoder.parameters())+list(decoder.parameters())
optimizer = optim.Adam(params, lr=1e-3)


# In[21]:


def train(encoder, decoder, train_loader, optimizer, num_epochs = 10):
    encoder.train()
    decoder.train()
    
    for epoch in range(num_epochs):
        train_loss = 0
        for i, (batch_data, batch_target) in enumerate(train_loader):

            optimizer.zero_grad()
            
            mu, logvar = encoder.forward(batch_data.view(-1, 784))
            
            latent_vector = sample(mu, logvar)
            
            recon_batch = decoder.forward(latent_vector)
            batch_loss = vae_loss(batch_data, recon_batch, mu, logvar)
            
            batch_loss.backward()
            train_loss += batch_loss.item()
            optimizer.step()
        print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    
    return


# In[22]:


train(encoder, decoder, train_loader, optimizer, num_epochs = 10)


# In[23]:


import matplotlib.pyplot as plt
from torchvision import utils
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[24]:


def create_interpolates(A, B, encoder, decoder):
    A_mu, A_logvar = encoder.forward(A.view(-1, 784))
    A_latent_vector = sample(A_mu, A_logvar)
    B_mu, B_logvar = encoder.forward(B.view(-1, 784))
    B_latent_vector = sample(B_mu, B_logvar)
    
    A_array = A_latent_vector.detach().numpy()
    B_array = B_latent_vector.detach().numpy()
    
    total_array = np.zeros((9, latent_dim))
    total_array[0] = A_array[0]
    total_array[8] = B_array[0]

    for i in range(1, 8):
        for j in range(latent_dim):
            
            space = (A_array[0][j] - B_array[0][j]) / 8
            total_array[i, j] = A_array[0][j] - space * i
    
    interpolation = []
    
    interpolation.append(decoder.forward(A_latent_vector))

    for i in range(1, 8):
        curr_tensor = torch.from_numpy(total_array[i]).float()
        interpolation.append(decoder.forward(curr_tensor))
    interpolation.append(decoder.forward(B_latent_vector))
    
    return interpolation


# In[25]:


similar_pairs = {}

for _, (x, y) in enumerate(test_loader):
    for i in range(len(y)):
        if y[i].item() not in similar_pairs:
            similar_pairs[y[i].item()] = []
        if len(similar_pairs[y[i].item()])<2:
            similar_pairs[y[i].item()].append(x[i])
  
    done = True
    for i in range(10):
        if i not in similar_pairs or len(similar_pairs[i])<2:
            done = False
  
    if done:
        break
        


# In[26]:


fig = plt.figure(figsize=(15, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(0, 10):
    curr_interp = create_interpolates(similar_pairs[i][0], similar_pairs[i][1], encoder, decoder)
    for j in range(0, 9):
        curr_plot = curr_interp[j].detach().numpy()
        curr_plot = curr_plot.reshape((28, 28))
        
        ax = fig.add_subplot(10, 9, (i * 9) + j + 1)
        ax.set_axis_off()
        plt.imshow(curr_plot, cmap='gray')

plt.show()


# In[27]:


random_pairs = {}

for _, (x, y) in enumerate(test_loader):
    for i in range(10):
        random_pairs[i] = []
        random_pairs[i].append(x[2*i])
        random_pairs[i].append(x[2*i+1])
    break


# In[28]:


fig = plt.figure(figsize=(15, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(0, 10):
    curr_interp = create_interpolates(random_pairs[i][0], random_pairs[i][1], encoder, decoder)
    for j in range(0, 9):
        curr_plot = curr_interp[j].detach().numpy()
        curr_plot = curr_plot.reshape((28, 28))
        # curr_plot = np.expand_dims(curr_plot, axis=0)
        
        ax = fig.add_subplot(10, 9, (i * 9) + j + 1)
        ax.set_axis_off()
        plt.imshow(curr_plot, cmap='gray')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




