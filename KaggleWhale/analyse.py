import pandas as pd

x = pd.read_csv('train.csv')

print( "%d training images" %x.shape[0])

print( "Nbr of samples/class\tNbr of classes")

for index, val in x["Id"].value_counts().value_counts().sort_index().iteritems():
    print("%d\t\t\t%d" %(index,val))
