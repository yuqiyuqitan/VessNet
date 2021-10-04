#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


def getConfusionMatrix(model, show_image=False):
    model.eval() #set the model to evaluation mode
    confusion_matrix=np.zeros((2,2),dtype=int) #initialize a confusion matrix
    num_images=testset_sizes['test'] #size of the testset
    
    with torch.no_grad(): #disable back prop to test the model
        for i, (inputs, labels) in enumerate(testloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #get predictions of the model
            outputs = model(inputs) 
            _, preds = torch.max(outputs, 1) 
            
            #get confusion matrix
            for j in range(inputs.size()[0]): 
                if preds[j]==1 and labels[j]==1:
                    term='TP'
                    confusion_matrix[0][0]+=1
                elif preds[j]==1 and labels[j]==0:
                    term='FP'
                    confusion_matrix[1][0]+=1
                elif preds[j]==0 and labels[j]==1:
                    term='FN'
                    confusion_matrix[0][1]+=1
                elif preds[j]==0 and labels[j]==0:
                    term='TN'
                    confusion_matrix[1][1]+=1
                #show image and its class in confusion matrix    
                if show_image:
                    print('predicted: {}'.format(class_names[preds[j]]))
                    print(term)
                    imshow(inputs.cpu().data[j])
                    print()
        # calculate performance matrix
        sensitivity = 100*confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1])
        specificity = 100*confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0])
        accuracy = 100*(confusion_matrix[0][0]+confusion_matrix[1][1])/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1])
        jaccard = 100*confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]+confusion_matrix[0][1])
        dice = 100*2*confusion_matrix[0][0]/(2*confusion_matrix[0][0]+confusion_matrix[1][0]+confusion_matrix[0][1])
        
        return sensitivity, specificity, accuracy, jaccard, dice 


# In[ ]:




