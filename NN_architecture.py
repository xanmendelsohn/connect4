import numpy as np
import tensorflow.keras
import tensorflow.keras.layers as Kl
import tensorflow.keras.models as Km
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_A1:

    def __init__(self, model_name):
        self.model_name = model_name

    def create_model(self):
        print('new model')

        model = Km.Sequential()
        model.add(Kl.Conv2D(20, (5, 5), padding='same', input_shape=(6, 7, 1)))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(20, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(20, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))

        model.add(Kl.Flatten(input_shape=(7, 7, 1)))
        model.add(Kl.Dense(49))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(7))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(7))
        opt = tensorflow.keras.optimizers.Adam() #optimizers.adam() 
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

        model.summary()

        return model
    
class CNN_A2:

    def __init__(self, model_name):
        self.model_name = model_name

    def create_model(self):
        print('new model')

        model = Km.Sequential()
        model.add(Kl.Conv2D(20, (5, 5), padding='same', input_shape=(7, 7, 1)))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(20, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Flatten(input_shape=(7, 7, 1)))
        model.add(Kl.Dense(49))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(7))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(1, activation='linear'))
        opt = tensorflow.keras.optimizers.Adam() #optimizers.adam() 
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

        model.summary()

        return model
    
class CNN_A3:

    def __init__(self, model_name):
        self.model_name = model_name

    def create_model(self):
        print('new model')

        model = Km.Sequential()
        model.add(Kl.Conv2D(64, (4,4), input_shape=(6, 7, 1)))
        model.add(Kl.Activation('relu'))
        model.add(Kl.Conv2D(64, (2, 2)))
        model.add(Kl.Activation('relu'))
        model.add(Kl.Conv2D(64, (2, 2)))
        model.add(Kl.Activation('relu'))
        model.add(Kl.Flatten())
        model.add(Kl.Dense(64))
        model.add(Kl.Activation('relu'))
        model.add(Kl.Dense(7))

        opt = tensorflow.keras.optimizers.Adam() #optimizers.adam() 
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

        model.summary()

        return model
    
class CNN_A4:

    def __init__(self, model_name):
        self.model_name = model_name

    def create_model(self):
        print('new model')

        model = Km.Sequential()
        model.add(Kl.Conv2D(64, (4,4), 
                            input_shape=(6, 7, 1)))
        model.add(Kl.LeakyReLU(alpha=0.1))
        model.add(Kl.Conv2D(64, (2, 2)))
        model.add(Kl.LeakyReLU(alpha=0.1))
        model.add(Kl.Conv2D(64, (2, 2)))
        model.add(Kl.LeakyReLU(alpha=0.1))
        model.add(Kl.Flatten())
        model.add(Kl.Dense(64))
        model.add(Kl.LeakyReLU(alpha=0.1))
        model.add(Kl.Dense(7))

        opt = tensorflow.keras.optimizers.Adam()  
        model.compile(optimizer=opt, 
                      loss='mean_squared_error', 
                      metrics=['accuracy'])

        model.summary()

        return model
    
class PyCNNik_A4(nn.Module):

    def __init__(self, model_name):
        self.model_name = model_name
        super(PyCNNik_A4, self).__init__()

        #input (1, 7, 6)
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 4, padding = "valid") #3,4
        self.conv1_activation = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 2, padding = "valid") #2, 3
        self.conv2_activation = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 2, padding = "valid") #1, 2
        self.conv3_activation = nn.LeakyReLU(0.1)
        #flatten x = x.view(x.size(0),-1)
        self.linear1 = nn.Linear(128, 64) #64*1*2
        self.activationL1 = nn.LeakyReLU(0.1)
        self.out = torch.nn.Linear(64, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_activation(x)
        x = self.conv2(x)
        x = self.conv2_activation(x)
        x = self.conv3(x)
        x = self.conv3_activation(x)
        #x = x.view(x.size(0),-1)
        x = x.view(-1, 2*64)
        x = self.linear1(x)
        x = self.activationL1(x)
        x = self.out(x)
        
        return x
    
class PyCNNik_test(nn.Module):

    def __init__(self, model_name):
        self.model_name = model_name
        super(PyCNNik_test, self).__init__()
        self.linear1 = nn.Linear(7, 64)
        self.activationL1 = nn.LeakyReLU(0.1)
        self.out = torch.nn.Linear(384, 7)
        self.float()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activationL1(x)
        x = x.view(x.size(0),-1)
        x = self.out(x)
        
        return x

    
def weight_translate(w):
    #translate tf weight tensor shape to torch weight tensor shape
    if w.dim() == 2:
        w = w.t()
    elif w.dim() == 1:
        pass
    else:
        assert w.dim() == 4
        w = w.permute(3, 2, 0, 1)
    return w
    
def translate_kerasA4_to_pytorch(agent, pytorch_model):
    
    weights = agent.model.get_weights()
    
    pytorch_model.conv1.weight.data = weight_translate(torch.FloatTensor(weights[0])) 
    pytorch_model.conv1.bias.data = weight_translate(torch.FloatTensor(weights[1])) 
    pytorch_model.conv2.weight.data = weight_translate(torch.FloatTensor(weights[2]))
    pytorch_model.conv2.bias.data = weight_translate(torch.FloatTensor(weights[3]))
    pytorch_model.conv3.weight.data = weight_translate(torch.FloatTensor(weights[4]))
    pytorch_model.conv3.bias.data = weight_translate(torch.FloatTensor(weights[5]))
    pytorch_model.linear1.weight.data = weight_translate(torch.FloatTensor(weights[6]))
    pytorch_model.linear1.bias.data = weight_translate(torch.FloatTensor(weights[7]))
    pytorch_model.out.weight.data = weight_translate(torch.FloatTensor(weights[8]))
    #pytorch_model.out.bias.data = weight_translate(torch.FloatTensor(weights[9]))
    
    s = 'DQNs/' + agent.model_name + '_torch' + '.h5'
    torch.save(pytorch_model.state_dict(), s)
    #torch.save(pytorch_model, s)