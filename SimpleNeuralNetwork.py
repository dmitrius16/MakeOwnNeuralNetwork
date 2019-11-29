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
        
    def train(self, inputs_list,targets_list):
        #convert inputs list to 2d array
        inputs = np.array(inputs_list,ndmin = 2).T
        targets = np.array(targets_list,ndmin = 2).T
        #calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih,inputs)
        #calculate the signals emerding from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
       
         #calculate signals into final output layer
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        #error is the 
        output_errors = targets - final_outputs
        #hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T,output_errors)
        #update the weights for the links between the hidden and output layers
        self.who += self.lr*np.dot((output_errors*final_outputs*(1.0 - final_outputs)),np.transpose(inputs))
        self.wih += self.lr * np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),np.transpose(inputs))
        
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
net.train([1.0,0.5,-1.5],[0.2,0.2,0.2])
print(net.query([1.0,0.5,-1.5]))