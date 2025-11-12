#Import libraries ( numpy , matplotlib , scipy   )

import numpy as np
import matplotlib.pyplot as plt
import scipy

#   ***************************************************************************************************



class Linear:
    def __init__(self,input_dim,output_dim,weight_scale=0.01,rng_seed=42):
        rng= np.random.RandomState(rng_seed)
        self.w = rng.randn(output_dim,input_dim)*weight_scale
        self.b = np.zeros((output_dim,))
        self.Input =None
        self.output=None
        self.parentd=[]
        self.childrens=[]




    def forword(self,Input):
        self.Input = Input
        self.output = Input @ self.w.T + self.b
        return self.output


    def Backpopagation(self,output_gradient):


        batch_size =self.Input.shape[0]


        d_input =  output_gradient @ self.w
        d_w =     output_gradient.T @ self.Input / batch_size
        d_b=      np.mean(output_gradient,axis=0)

        return d_input,d_w,d_b

#   *******************************************************************************************************



from keras.datasets import mnist
from matplotlib import pyplot


(train_X, train_y), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()







X= train_X.reshape(train_X.shape[0],-1)
y= train_y

indices= np.arange(X.shape[0])
np.random.shuffle(indices)
X=X[indices]
y=y[indices]


layer_dim = [784,100,10]


x_train = X/255
x_test = test_X.reshape(test_X.shape[0],-1)/255
y_train = y
y_test = test_y




def one_hot(y,num_classes=10):
    return np.eye(num_classes)[y]
y_train__oh =one_hot(y_train)
y_test__oh = one_hot(y_test)


# sigmoid function ***
def sigmoid(z):
    return 1/(1+np.exp(-z))


def tanh(z):
    return np.tanh(z)

def ReLU(z):
    return np.maximum(0,z)




def loss(y_pred,y_true):
    epsilon = 1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))


# Initialize w and b ***

input_size = x_train.shape[1]
output_size = 1
w =np.random.randn(output_size,input_size)*0.02
b= np.zeros(output_size)

def softmax(z):

    exp_zz =   np.exp(z -np.max(z,axis=1, keepdims=True) )

    return   exp_zz /np.sum(exp_zz,axis=1, keepdims=True)








# MLP Class

class MLP:
    def __init__(self,layer_dim,activation="sigmoid",weight_scale=0.01,rng_seed=2):
        self.activation_name= activation
        self.layers=[]
        for i in range(len(layer_dim)-1):
            layer = Linear(layer_dim[i],layer_dim[i+1],weight_scale=weight_scale,rng_seed=rng_seed+i)
            self.layers.append(layer)
        for i in range(len(self.layers)-1):
            self.layers[i].childrens.append(self.layers[i+1])
            self.layers[i+1].parentd.append(self.layers[i])
        self.graph = self.topological()

        self.trainble=[layer for layer in self.graph if hasattr(layer,"w") and hasattr(layer,"b")]

    def topological(self):
        visited = set()
        order = []
        def dfs(node):
            if node in visited :
                return
            visited.add(node)
            for child in node.childrens :
                dfs(child)
            order.append(node)
        dfs(self.layers[0])
        return order

    def activation(self,z):
        if self.activation_name== "sigmoid":
            return sigmoid(z)
        elif self.activation_name== "tanh":
            return tanh(z)
        elif self.activation_name== "ReLU":
            return ReLU(z)

    def activation_d(self,a):
        if self.activation_name == "sigmoid":
            return a*(1-a)
        elif self.activation_name == "tanh":
            return 1-np.square(a)
        elif self.activation_name == "ReLU":
            return (a>0).astype(float)






    def forward(self,x):

        activations= [x]

        for i,layer in enumerate(self.layers):

            z= layer.forword((activations[-1]))
            if i<len(self.layers)-1:
                a= self.activation(z)
            else:

               a= softmax(z)

            activations.append(a)

        self.activations=activations
        return activations[-1]





    def backward(self,y_pred,y_correct):

       grads= []

       dz=(y_pred-y_correct) / y_correct.shape[0]



       for i in reversed(range(len(self.layers))):

           layer = self.layers[i]



           d_input,d_w,d_b =layer.Backpopagation(dz)

           grads.insert(0,(d_w,d_b))

           if i>0:
               prev= self.activations[i]
               dz= d_input*self.activation_d(prev)
       return grads



    def update(self,grad,lr=0.01,l2_reg=0):


        for layer ,(dw,db) in zip(self.layers,grad):
            if l2_reg:
                dw =dw +l2_reg*layer.w
            layer.w -= lr*dw
            layer.b -= lr*db




# ******* Training *******

def training_model(X,y,y_onehot,layer_dim=layer_dim,activation="sigmoid",epoch=10,lr=0.5,batch_size=1,l2_reg=0.0):


    model =MLP(layer_dim,activation=activation,weight_scale=1)
    losses= []


    for i in range(epoch):
         perm= np.random.permutation(X.shape[0])
         X_shuffle= X[perm]
         y_shuffle= y_onehot[perm]
         for j in range(0,X.shape[0],batch_size):
             x_batch =X_shuffle[j:j+batch_size]
             y_batch = y_shuffle[j:j + batch_size]

             y_pred = model.forward(x_batch)
             grads = model.backward(y_pred, y_batch)
             model.update(grads, lr=lr, l2_reg=l2_reg)

         y_pred_full = model.forward(X)
         loss_value =loss(y_pred_full,y_onehot)
         losses.append(loss_value)

         if i%10 ==0:
             print(f"Epoch {i} | Loss = {loss_value:.4f}")
    return model,losses

# ********* compare different Batch size *********
activations= ["ReLU"]
batch_size= [1,32,128]
layer_dim = [784,784,10]



plt.figure(figsize=(10,6))


for act in activations:
    model,losses = training_model(x_train,y_train,y_train__oh,layer_dim=layer_dim,activation=act,epoch=5,lr=0.1,batch_size=1,l2_reg=0.0)
    plt.plot(losses,label=f"{act}")

plt.title("Training Loss vs Epochs for Different Activations")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)
plt.show()






y_pred_test= model.forward(x_test)
y_pred_classes =np.argmax(y_pred_test,axis=1)
accuracy = np.mean(y_pred_classes==y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")






