#Deep Learning for Alzheimer's Disease Detection: A CNN Approach

Using CNN model, Alzheimer's disease can be detected in four classes (non, very mild, mild, moderate). Trained model is integrated into a python application which enable user to detect Alzheimer's disease by importing 3D MRI image (NIfTI format).

#Process

*Alzheimer preprocessed MRI dataset is downloaded from Kaggle website. Dataset is imbalanced. Therefore, augmentation technique is used.
*2D-CNN architecture is build using Tensorflow and keras modules.
*Model is trained and evaluated. COnfusion matrix is drawn.
*Python GUI is created using Kivy package. It enable user to easily interface with the trained model. 

#Conclusion

In this work, we created a four-class predictive model for Alzheimer's disease: non-demented, very mildly demented, moderately demented, and mildly demented. After using augmentation techniques, the model's accuracy was an astounding 99.53%. Developed a user-friendly Python application utilising the Kivy module to give users easy access to our model. Users of this application can upload an NIfTI file and specify the medial temporal lobe's range. The application can forecast the disease stage with accuracy of 75.86% using our model. These findings open the door for additional study in this field and demonstrate the potential of applying deep learning techniques to increase the precision of Alzheimer's disease diagnosis.
