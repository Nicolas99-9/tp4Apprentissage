import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
from pprint import pprint

with gzip.open('mnist.pkl.gz','rb','latin-1') as ifile:
    train_set,valid_set,test_set = pickle.load(ifile)



def show_image(i,j):
    im = train_set[j][i].reshape(28,28)
    plt.imshow(im,plt.cm.gray)
    plt.show()


#show_image(0,0)
pprint(train_set[0][0])
# train set => matrice contenant les images et deuxieme elemeent est l etiquette associee
#print(valid_set)

# image de 28 * 28 => valeur de gris



#prend une matrice et la binarise
def binarize(tableau):
    for e in range(0,len(tableau)):
        for i in range(0,len(tableau[e])):
            if(tableau[e][i]>0.5):
                tableau[e][i] = 1
            else:
                tableau[e][i] =0

def binarize_short(tableau):
    for e in range(0,len(tableau)):
            if(tableau[e]>0.5):
                tableau[e] = 1
            else:
                tableau[e] =0


binarize_short(train_set[0][0])


def save_tagger(filename,tag):
    output = pickle.open(filename,'wb')
    dump(tag,output,-1)
    output.close()

def loads(file_name):
    input = open(file_name,'rb')
    tagger = pickle.load(input)
    input.close()
    return tagger


#save("save1.p",train_set)

#show_image(0,0)
#new_train_set = loads("save1.p")
#print(len(new_train_set))

def histo_horizontal(image):
    nb = 0
    tab = []
    for i in range(0,len(image)):
        if(image[i]==1):
          nb+=1
        if(i%28==0):
            tab.append(nb)
            nb = 0
    return tab


print("histogramme horizontale: ",histo_horizontal(train_set[0][0]))

def histo_vertical(image):
    tab = [0 for i in range(0,28)]
    for i in range(0,len(image)):
        if(image[i]==1):
          tab[i%28] +=1
    return tab
print("histogramme verticale: ",histo_vertical(train_set[0][0]))

def display_histo_hor(tableau):
    plt.plot([i for i in range(0,len(tableau))],tableau)
    plt.show()

def display_histo_ver(tableau):
    plt.plot(tableau,[i for i in range(0,len(tableau))])
    plt.show()


#display_histo_hor(histo_horizontal(train_set[0][0]))

#display_histo_ver(histo_vertical(train_set[0][0]))


#print((train_set[1][:10]))
#1 : train_set[1][3]
# 3 :train_set[1][7]
# 5 : train_set[1][0]




