'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
 
class Method_MLP(method, nn.Module):
    data = None
    history = None
    # it defines the max rounds to train the model
    max_epoch = 300
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # default optimizer: 'Adam', 'SGD', 'RMSprop'
    optimizer_name = 'Adam'

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 256)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(256, 64)
        self.activation_func_2 = nn.ReLU()
        self.fc_layer_3 = nn.Linear(64, 10)
        # because we use CrossEntropyLoss, we should not use the softmax layer

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        # outout layer result
        # self.fc_layer_3(h2) will be a nx10 tensor
        # n (denotes the input instance number): 0th dimension; 10 (denotes the class number): 1st dimension
        y_pred = self.fc_layer_3(h2)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):

        opt_name = self.optimizer_name.lower()
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html

        if opt_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif opt_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif opt_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('Optimizer not found')
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # convert X and y to tensors once before the training loop
        X_tensor = torch.FloatTensor(X)
        y_true = torch.LongTensor(y)

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output
            y_pred = self.forward(X_tensor)
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch % 10 == 0 or epoch == self.max_epoch - 1:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                acc = accuracy_evaluator.evaluate()
                loss_val = train_loss.item()

                if self.history:
                    self.history.record(Epoch=epoch, Loss=loss_val, Accuracy=acc)

                if epoch % 50 == 0 or epoch == self.max_epoch - 1:
                    print('Epoch:', epoch, 'Accuracy:', acc, 'Loss:', loss_val)

        if self.history:
            self.history.save()
    
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(X))
        # logits, which are the raw output of the neural network before the softmax layer
        # return the predicted class index (argmax of raw logits)
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}


class Method_MLP_Single_Layer(Method_MLP):
    """
    Ablation model: MLP with a single fully connected layer (784 -> 10).
    Inherits training, evaluation, and optimizer configuration from Method_MLP.
    """
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.fc_layer = nn.Linear(784, 10)

    def forward(self, x):
        '''Forward propagation through the single linear layer'''
        return self.fc_layer(x)


class Method_MLP_Dual_Layer(Method_MLP):
    """
    Ablation model: MLP with two fully connected layers (784 -> hidden_dim -> 10) and ReLU activation.
    Inherits training, evaluation, and optimizer configuration from Method_MLP.
    """
    def __init__(self, mName, mDescription, hidden_dim=256):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.fc_layer_1 = nn.Linear(784, hidden_dim)
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        '''Forward propagation through the dual layers'''
        h1 = self.activation_func_1(self.fc_layer_1(x))
        y_pred = self.fc_layer_2(h1)
        return y_pred


class Method_MLP_No_Activation(Method_MLP):
    """
    Ablation model: 3-layer MLP without activation functions (pure linear transformations).
    Demonstrates that without non-linearities, deep networks collapse into a linear model.
    """
    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)

    def forward(self, x):
        '''Forward propagation without activation functions'''
        h1 = self.fc_layer_1(x)
        h2 = self.fc_layer_2(h1)
        y_pred = self.fc_layer_3(h2)
        return y_pred