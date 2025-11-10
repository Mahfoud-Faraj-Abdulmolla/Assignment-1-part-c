import numpy as np
from scipy.signal import correlate2d
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class Conv:
 def __init__(self,Cin,Cout,kernel=3,lr=0.02,padding=1):
  self.Cin,self.Cout,self.kernel,self.lr,self.pad=Cin,Cout,kernel,lr,padding

  self.bias=np.zeros(Cout)
  self.kernels=np.random.randn(Cout,Cin,kernel,kernel)*0.2

 def forward(self,x):
  self.x=np.pad(x,((0,0),(self.pad,self.pad),(self.pad,self.pad)))
  h,w=x.shape[1:3]

  out=np.zeros((self.Cout,h-self.kernel+1,w-self.kernel+1))
  for i in range(self.Cout):

   for j in range(self.Cin):

    out[i]+=correlate2d(self.x[j],self.kernels[i,j],mode='valid')
   out[i]+=self.bias[i]
  self.mask=(out>0).astype(float)
  return out*self.mask

 def backward(self,grad):
  grad*=self.mask
  grad_k,grad_b=np.zeros_like(self.kernels),np.zeros_like(self.bias)
  grad_x=np.zeros_like(self.x)
  for i in range(self.Cout):

   grad_b[i]=grad[i].sum()

   for j in range(self.Cin):
    grad_k[i,j]=correlate2d(self.x[j],grad[i],mode='valid')

    grad_x[j]+=correlate2d(grad[i],self.kernels[i,j],mode='full')
  self.kernels-=self.lr*grad_k
  self.bias-=self.lr*grad_b
  return grad_x[:,self.pad:-self.pad,self.pad:-self.pad]

class MaxPooling:
 def __init__(self,size=2,stride=2):

  self.size,self.stride=size,stride

 def forward(self,x):
  self.x,C,h,w=x,*x.shape

  out_h,out_w=(h-self.size)//self.stride+1,(w-self.size)//self.stride+1

  out=np.zeros((C,out_h,out_w))
  self.mask=np.zeros_like(x)
  for c in range(C):

   for i in range(0,h-self.size+1,self.stride):

        for j in range(0,w-self.size+1,self.stride):
         region=x[c,i:i+self.size,j:j+self.size]
         max_val=np.max(region)
         out[c,i//self.stride,j//self.stride]=max_val
         self.mask[c,i:i+self.size,j:j+self.size]=(region==max_val)
  return out

 def backward(self,grad):
          grad_in=np.zeros_like(self.x)
          C,out_h,out_w=grad.shape
          for c in range(C):
           for i in range(out_h):
            for j in range(out_w):
             h0,w0=i*self.stride,j*self.stride
             grad_in[c,h0:h0+self.size,w0:w0+self.size]+=grad[c,i,j]*self.mask[c,h0:h0+self.size,w0:w0+self.size]
          return grad_in






class Linear:
 def __init__(self,input_dim,output_dim,lr=0.01):
  self.w=np.random.randn(output_dim,input_dim)*0.01
  self.b=np.zeros(output_dim)
  self.lr=lr




 def forward(self,x):
  self.Input=x
  return x@self.w.T+self.b





 def backward(self,grad_out):
  d_w=grad_out.T@self.Input
  d_b=grad_out.mean(axis=0)
  d_input=grad_out@self.w
  self.w-=self.lr*d_w
  self.b-=self.lr*d_b
  return d_input



def softmax(z):
 z=np.clip(z,-100,100)
 e=np.exp(z-np.max(z))
 return e/e.sum(axis=-1,keepdims=True)

def cross_entropy_loss(y_pred,y_true):
 eps=1e-8
 return -np.mean(np.sum(y_true*np.log(y_pred+eps),axis=1))



class CNN:
 def __init__(self,lr=0.002):
  self.conv1,self.pool1=Conv(3,16),MaxPooling()
  self.conv2,self.pool2=Conv(16,32),MaxPooling()
  self.conv3,self.pool3=Conv(32,64),MaxPooling()
  self.conv4=Conv(64,128)
  self.linear=Linear(128,10,lr)




 def forward(self,x):
  x=self.pool1.forward(self.conv1.forward(x))
  x=self.pool2.forward(self.conv2.forward(x))
  x=self.pool3.forward(self.conv3.forward(x))
  x=self.conv4.forward(x)
  self.feature=x
  return softmax(self.linear.forward(x.flatten().reshape(1,-1)))


#***********************************************************************
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
x_train=np.repeat(x_train[...,np.newaxis],3,axis=-1)
x_test=np.repeat(x_test[...,np.newaxis],3,axis=-1)
y_train_oh=to_categorical(y_train,10)
y_test_oh=to_categorical(y_test,10)

cnn=CNN(lr=0.002)
epochs=2
for epoch in range(epochs):
 losses=[]
 correct=0
 for i in range(60000):
  img=x_train[i].transpose(2,0,1)
  label=y_train_oh[i].reshape(1,-1)
  out=cnn.forward(img)
  losses.append(cross_entropy_loss(out,label))
  if np.argmax(out)==np.argmax(label):
   correct+=1
 acc=correct/60000
 print(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f}, Accurcy={acc:.3f}")

correct=0
for i in range(10000):
 img=x_test[i].transpose(2,0,1)
 label=y_test_oh[i].reshape(1,-1)
 out=cnn.forward(img)
 if np.argmax(out)==np.argmax(label):
  correct+=1
print(f"Test Accurcy={correct/10000:.3f}")


