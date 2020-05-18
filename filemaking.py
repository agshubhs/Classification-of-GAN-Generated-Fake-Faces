import os,sys
  
# Function to rename multiple files  
path0= 'dataset/ffhq_/' 
path1= 'dataset/pgan_/' 
path2= 'dataset/sgan1/' 
path3= 'dataset/sgan2/' 

filepath= 'filelists/gan_images_train_list.txt' 
# sys.path.append(path) 
file1 = open(filepath,"w")

for count, filename in enumerate(os.listdir(path0)): 
    file1.write('dataset/ffhq_/'+filename+'|'+'dataset/ffhq__filter/'+filename + '|'+'0'+"\n")


for count, filename in enumerate(os.listdir(path1)): 
    file1.write('dataset/pgan_/'+filename+'|'+'dataset/pgan__filter/'+filename + '|'+'1'+"\n")

for count, filename in enumerate(os.listdir(path2)): 
    file1.write('dataset/sgan1/'+filename+'|'+'dataset/sgan1_filter/'+filename + '|'+'2'+"\n")

for count, filename in enumerate(os.listdir(path3)): 
    file1.write('dataset/sgan2/'+filename+'|'+'dataset/sgan2_filter/'+filename + '|'+'3'+"\n")


file1.close()
