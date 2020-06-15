from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import re
import numpy as np

# Setting which embedding and what dimension of the embedding we want
dataset = "multi"
class_list = "lists/%s-class-list.txt"%dataset
embed = 'glove'
embed_model = 'glove.6B.50d.txt'
binary = False
dim = 50

# Making the list of classes
f = open(class_list)
my_file = f.read().split("\n")
f.close()
class_list = []

for item in my_file:
    item = str(item)
    m = re.search('/(.+?)/',item)
    if m:
        found = m.group(1)
    class_list.append(found)

del[class_list[len(class_list)-1]]


# Loading the specific word embedding model
if embed == 'w2v':
    filename = '../../../model/%s-pret/%s'%(embed,embed_model)
elif embed == 'glove':
    glove_input_file = '../../../model/%s-pret/%s'%(embed,embed_model)
    word2vec_output_file = embed_model + '.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)
    filename = word2vec_output_file 
elif embed == 'fasttext':
    filename = '../../../model/%s-pret/%s'%(embed,embed_model)    

# Loading the model in w2v format for harmonization
model = KeyedVectors.load_word2vec_format(filename, binary=binary)
#words = list(model.wv.vocab)
num_class =  len(class_list)
a = np.zeros((num_class,dim),dtype = float)
i = 0
for item in class_list:
    item = str(item)
    item = item.lower()
    vec  = model[item]
    a[i,:] = vec
    print(vec[0:10])
    i = i + 1 
    print(i)
    
np.save('%s_%s.npy'%(dataset,embed_model),a)


