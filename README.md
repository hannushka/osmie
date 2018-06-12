Master thesis work by Hampus Londögård and Hannah Lindblad. The title of the thesis is "Improving the OpenStreetMap Data Set using Deep Learning" and the report is available at https://drive.google.com/open?id=18XKsvQLKRcZ06NvteBZdwLuXdm1VQbgv.

# Abstract

OpenStreetMap is an open source of geographical data where contributors can change, add, or remove data. Since anyone can contribute, the data set is prone to contain data of varying quality. In this work, we focus on three approaches for correcting Way component name tags in the data set: Correcting misspellings, flagging anomalies, and generating suggestions for missing names. 

Today, spell correction systems have achieved a high correction accuracy. However, the use of a language context is an important factor to the success of these systems. 

We present a way for performing spell correction without context through the use of a deep neural network. The structure of the network also makes it possible to adapt it to a different language by changing the training resources. The implementation achieves an F1 score of 0.86 (ACR 0.69) for Way names in Denmark. 
