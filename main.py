import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
from pprint import pprint
import operator
from itertools import islice

with gzip.open('mnist.pkl.gz','rb','latin-1') as ifile:
    train_set,valid_set,test_set = pickle.load(ifile)



def show_image(i,j):
    im = train_set[j][i].reshape(28,28)
    plt.imshow(im,plt.cm.gray)
    plt.show()


def show_image2(tab):
    im = np.array(tab).reshape(28,28)
    plt.imshow(im,plt.cm.gray)
    plt.show()


#show_image(0,0)
#pprint(train_set[0][0])
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


#print("histogramme horizontale: ",histo_horizontal(train_set[0][0]))

def histo_vertical(image):
    tab = [0 for i in range(0,28)]
    for i in range(0,len(image)):
        if(image[i]==1):
          tab[i%28] +=1
    return tab
#print("histogramme verticale: ",histo_vertical(train_set[0][0]))

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


def moyennes(liste_image):
    #mon_tab = [[0 for i in range(0,28)] for  j in range(0,28)]
    mon_tab = [0 for  j in range(0,784)]
    for image in liste_image:
        for i in range(len(image)):
            mon_tab[i] += image[i]
    taille = float(len(liste_image))
    for i in range(len(mon_tab)):
        mon_tab[i] = mon_tab[i]/taille
    return mon_tab


#moyenne sur des chiffres differents
#ma_moyenne  = moyennes(train_set[0][:5])
#show_image2(ma_moyenne)


def moyenne_egal(val):
    liste_appel = []
    for vale in range(len(train_set[1])):
        if(train_set[1][vale]==val):
            liste_appel.append(train_set[0][vale])
    ma_moyenne  = moyennes(liste_appel)
    show_image2(ma_moyenne)

#moyenne_egal(4)

# espace euclident => distance entre points
# plus des points sont proches plus ils sont similaires
# attribuer l etiquette de l exemple le plus proche 
# pour etre plus robuste faire moyenne des etiquettes des k voisins les plus proches
# complexites : N * d (nombre exemples * nombre de dimensions) pour classifier un point


#DIstance minimale
# determiner un representant de chaque classe en utilisant une moyenne
#attribuer une etiquette de la classe dont representant est plus proche

def sortes(ma_list):
    result = sorted(ma_list.items(), key=operator.itemgetter(1),reverse= True)
    return result

def distance(a,b):
    return np.linalg.norm(a-b)


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))



def get_k_nearest_voisins(k,liste,x):
    lis = {}
    for i in range(len(liste)):
        lis[i] = distance(x,liste[i])
    return (sortes(lis)[:k])


def prediction_k(image_to_predict,k,train_datas,labels):
    voisons_proche = get_k_nearest_voisins(k,train_datas,image_to_predict)
    compte= [0 for i in range(0,10)]
    for e,v in voisons_proche:
        compte[labels[e]]+=1
    print(compte)
    return (sorted(compte,reverse=True)[0])
    

get_k_nearest_voisins(3,[0,2,3,5,8,2],2)










