import numpy as np
from scipy.signal import convolve2d

# ***********************************************************************************************************
class Conv:
    def __init__(self,Cin,Cout,kernal=3):

        self.Cin = Cin

        self.Cout= Cout

        self.kernal = kernal

        self.bais = np.zeros(Cout)

        self.kernals= np.random.randn(Cout,Cin,kernal,kernal)*0.2

    def forward(self,x):
        Cin,h,w =x.shape
        out_map = []

        for i in range(self.Cout):
            conv_sum = np.zeros((h-self.kernal+1,w-self.kernal+1))


            for j in range(Cin):
                conv_sum+= convolve2d(x[j],self.kernals[i,j],mode="valid")

            conv_sum += self.bais[i]
            conv_sum = np.maximum(conv_sum,0)
            out_map.append((conv_sum))
        return np.array(out_map)




#*******************************************************************************************************



class MaxPooling:
    def __init__(self,size=2,stride=2):

        self.size=size

        self.stride= stride


    def forward(self,x):

        C,h,w=x.shape

        out_h = (h-self.size)//self.stride+1

        out_w= (w-self.size)//self.stride+1

        pooled = np.zeros((C,out_h,out_w))

        for c in range(C):

            for i in range(0,h-self.size+1,self.stride):

                for j in range(0,w-self.size+1,self.stride):

                    region=x[c,i:i+self.size,j:j+self.size]

                    pooled[c,i//self.stride,j // self.stride]=np.max(region)

        return pooled


