import os
import numpy as np
import torch
import torch.nn as nn

# Converted Tensorflow model


# Build base model in pytorch
class CTModel(nn.Module): #  Determine what layers and their order in CNN object #  We are building model 88 from the PGDL dataset

    def __init__(self, input_model,verbose=False,data_type='float64'):
        super(CTModel, self).__init__()
        self.verbose=verbose
        #self.layers = [] # We are chaning to use a Module List. Ughn I should be using version control lol
        self.layers = nn.ModuleList()
        if data_type =='float64':
            self.dtype=torch.float64
        elif data_type =='float32':
            self.dtype=torch.float32
        # Flag to flip the first dense layer. 
        # When the first dense layer flips, this will
        # be set to False

        self.first_dense_flip = True

        # Go through layers of input model, and convert. 
        for l in input_model.layers:
            # convert layer name to string and only get the last part
            layer_name = str(l.__class__.__base__).split('.')[-1]
            #print (layer_name)

            #Could use match/case but that was only implemented in python 3.10
            if layer_name=="Conv2D'>":
                self.convert_conv2d_layer(tf_layer = l)
            
            elif layer_name=="MaxPooling2D'>":
                self.convert_max_pooling_layer(tf_layer = l)

            elif layer_name=="Flatten'>":
                self.convert_flatten_layer(tf_layer = l)

            elif layer_name=="Dense'>":
                self.convert_dense_layer(tf_layer = l)
            else:
                #import ipdb; ipdb.set_trace()
                raise('Uknown layer error')

    def forward(self, x):
        # Loop through the layers
        for l in self.layers:
            # On the flatten Layer, do a data transform to handle channels last
            if 'flatten.Flatten' in str(type(l)):
                x=x.permute((0,2,3,1))

            x = l(x)

        return x

    def convert_conv2d_layer(self,tf_layer) -> None:
        '''
        For a given input tensorflow layer, construct a pytorch layer, and add it to the layer queue. 
        This method adds both the converted pytorch layer, and its ReLu activation function. 
        '''

        in_channels = tf_layer.input_shape[-1] # channel last in tensorflow
        out_channels = tf_layer.output_shape[-1]
        kernel_size = tf_layer.kernel_size[0] # kernels should be of shape (n,n)
        padding = tf_layer.padding
        

        pt_layer = nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            dtype=self.dtype
                        )
        # Copy weights and also perform transpose to move from channels to channels last
        weights_np=np.transpose(tf_layer.weights[0],(3,2,0,1)).copy().astype('float64')
        
        # Copy biases
        bias_np=np.asarray(tf_layer.weights[1]).copy().astype('float64')

        with torch.no_grad():
            pt_layer.weight.copy_(torch.tensor(weights_np, requires_grad=True,dtype=torch.float64))
            pt_layer.bias.copy_(torch.tensor(bias_np, requires_grad=True,dtype=torch.float64))

        self.layers.append(pt_layer)
        self.layers.append(nn.ReLU())
        if self.verbose:
            print(' This made a conv 2d layer')
        
    def convert_dense_layer(self,tf_layer)->None:
        in_features=tf_layer.input_shape[1]
        out_features=tf_layer.output_shape[1]
        pt_layer=nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    dtype=self.dtype
        )

        #import ipdb; ipdb.set_trace()

        # Copy weights and also perform transpose to move from channels to channels last
        weights_np=np.transpose(tf_layer.weights[0]).copy().astype('float64')

        # Copy biases
        bias_np=np.asarray(tf_layer.weights[1]).copy().astype('float64')
        with torch.no_grad():
            pt_layer.weight.copy_(torch.tensor(weights_np, requires_grad=True,dtype=torch.float64))
            pt_layer.bias.copy_(torch.tensor(bias_np, requires_grad=True,dtype=torch.float64))

        self.layers.append(pt_layer)
        
        if 'relu' in str(tf_layer.activation):
            self.layers.append(nn.ReLU())

        if self.verbose:
            print(' This made a dense layer')

    def convert_flatten_layer(self,tf_layer)->None:
        self.layers.append(nn.Flatten())
        if self.verbose:
            print(' This made a flatten layer')

    def convert_max_pooling_layer(self,tf_layer)->None:
        '''
        For a given input tensorflow layer, construct a pytorch layer, and add it to the layer queue. 
        This method adds both the converted pytorch layer, and its ReLu activation function. 
        '''

        padding=tf_layer.padding
        kernel_size=tf_layer.pool_size[0]
        stride=tf_layer.strides[0]
        pt_layer=nn.MaxPool2d(
                    kernel_size = kernel_size,
                    stride = stride,
        #            padding=padding
        )
        self.layers.append(pt_layer)

        if self.verbose:
            print(' This made a max pooling layer')

'''
cnet = ConvNeuralNet()

training_data = data_manager.load_training_data()
data_batches = training_data.batch(16, drop_remainder=True)



#Load state dict

cnet_state_dict = torch.load('cnet.pth')

cnet.load_state_dict(cnet_state_dict)
'''
