from matplotlib.pyplot import figure
from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=200,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=10
    )

    
    rbm.cd1(visible_trainset=train_imgs, n_iterations=30)
    
    #''' Two layers RBM '''
#
 #   print ("\nStarting a Two RBM net..")
  #  
    


   # vishid = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1], ndim_hidden=500,
    #                                                is_bottom=True, image_size=image_size, batch_size=10)
            
    #hidpen = RestrictedBoltzmannMachine(ndim_visible=500, ndim_hidden=500, batch_size=10)

    #print ("training vis--hid")
    #""" 
    #CD-1 training for vis--hid 
    #"""     
    #vishid.cd1(train_imgs, n_iterations=50)       

    #print ("training hid--pen")
    #""" 
    #CD-1 training for hid--pen 
    #"""            
    #vishid.untwine_weights() 
    #hOut = vishid.get_h_given_v_dir(train_imgs)[1]
    #hidpen.cd1(hOut, n_iterations=10)     


    
   