### A simple tutorial Using vgg16 to clssifier 8 categories based on tensorflow  

the dataset includes 8 categories,which are truck,tiger,flower,kittycat,guitar,houses,plane,person  
firstly, split data
```
python3 create_labels_files.py
```
secondly,create tf recond files
```
python3 create_tf_record.py
```
thirdly,start training
```
pyhton3 vgg16.py
```


