import numpy as np
import scipy.special as scspec
#neural network class definition
class neuralNetwork():
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        #self.wih = (np.random.rand(self.hnodes,self.inodes) - 0.5)
        #self.who = (np.random.rand(self.onodes,selfhnodes) - 0.5)
        #Below more sophisticated approach
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #activation function, expit - is sigmoid function  
        self.activation_function = lambda x:scspec.expit(x)
        
    def train():
        pass
    def query(self,inputs_list):
        #convert inputs to 2-d array
        inputs = np.array(inputs_list,ndmin = 2).T
        #calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

# test 
net = neuralNetwork(3,3,3,0.3)
print(net.query([1.0,0.5,-1.5]))