import json
import os
import numpy as np
import pickle as pkl
from sys import path

import torch

import torch.nn as nn

def get_margin_distribution(model,X,y,margin_length=4):
    # For a given model, input X, and output y
    # Get margin distribution

    ######################################################################
    # Step 0: Book keeping
    ######################################################################
    batch_size = X.shape[0]

    ######################################################################
    # Step 1 Run input, while using hooks to store the intermediate inputs       
    ######################################################################
    # Define a hook function to store intermediate outputs

    def hook_fn(module, input, output):
        setattr(module, "_input", input)
        setattr(module, "_output", output)

    # Register hooks to the desired layers
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Flatten,nn.MaxPool2d,nn.Conv2d,nn.ReLU)):  # Modify this condition based on the types of layers you want
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

    output = model(X)

    ######################################################################
    # Step 2: Get the numerator and the grad ys
    ######################################################################

    num_class = output.shape[1]
    # get list relating to each batch
    rr = torch.arange(64).to('cuda:0') # Hardcoded to be on cuda device #Will need to be updated asap

    # get indexes max values for batch
    max_val = torch.argmax(output,dim=1)

    # Slice outputs to get max values for batch

    ind = torch.stack([rr,max_val],dim=1)

    # Eq 3 left value

    values_true = output[ind[:,0],ind[:,1]]


    values, indices = torch.topk(output,k=2)

    pred_y_p = indices[:,0]

    yp = y

    true_match_float = (indices[:,0].int() == yp).float()

    values_c = values[:,1]*true_match_float + values[:,0]*(1-true_match_float)

    indices_c = indices[:,1]*true_match_float + indices[:,0]*(1-true_match_float)

    numerator = values_true - values_c

    grad_ys= torch.nn.functional.one_hot(yp.long(),num_class)

    grad_ys-=torch.nn.functional.one_hot(indices_c.long(),num_class)


    ######################################################################
    # Step 3: Get the margin distribution
    ######################################################################

    # Access intermediate outputs
    intermediate_inputs = []
    intermediate_outputs = []
    for name, module in model.named_modules():
       if isinstance(module, (nn.Linear,  nn.Flatten,nn.MaxPool2d,nn.Conv2d)):
            intermediate_inputs.append(getattr(module, "_input"))

    for name, module in model.named_modules():

       if isinstance(module, ( nn.Flatten,nn.MaxPool2d,nn.ReLU,nn.Linear)):  # Modify this condition based on the types of layers you want
            intermediate_outputs.append(getattr(module, "_output").view(batch_size,-1).cpu().detach().numpy())

    dct = {}
    for i in range(margin_length):
        layer_dims = intermediate_inputs[i][0].ndimension()
        g = torch.autograd.grad(outputs=output, inputs=intermediate_inputs[i][0].requires_grad_(), grad_outputs=grad_ys, retain_graph=True)[0]
        
        norm = torch.sqrt(torch.sum(g*g,axis = np.arange(1,layer_dims).tolist()))
        dct[i] = numerator/norm

    # convert margin dict to numpy
    margin_dist = {k:v.cpu().detach().numpy() for k,v in dct.items()}

    ######################################################################
    # Step 4: Detach hooks
    ######################################################################

    for hook in hooks:
        hook.remove()


    ######################################################################
    # Step 5: return
    ######################################################################

    return margin_dist, intermediate_outputs


def get_normalized_margin(md,int_out, margin_length=4):
    
    if margin_length is not None:
        # Margin Distribution, Intermediate output
        all_activations = np.concatenate(
                        [np.squeeze(activation) for activation in int_out[:margin_length]],
                        axis=1,
                     )
    else:
        # Margin Distribution, Intermediate output
        all_activations = np.concatenate(
                        [np.squeeze(activation) for activation in int_out],
                        axis=1,
                     )


    response_flat = all_activations.reshape([all_activations.shape[0], -1])
    response_std = np.std(response_flat, axis=0)
   
    total_variation = (np.sum(response_std ** 2)) ** 0.5 # total variation
    # make the margin dist scale invariant by dividing on to total variation
    layers_norm = [v / total_variation for v in md.values()]

    return layers_norm

