# ROB311_TP4
ROB311_TP4 LBP, KNN, facial expression classification
add the test and train files in same folder of .ipynb

The implementation steps : 
image 48*48 --> lbp features 48*48 --> partition into 4*4 regions , calculate for each region a historgram (base on lbp values) bin = 10 , obtain 16 vectors of 1*10 --> flatten the historgrammes : 1*160 descriptor --> use 1*160 descriptor for the KNN --> predict categorie of a image with KNN base on its 1*160 descriptor.
