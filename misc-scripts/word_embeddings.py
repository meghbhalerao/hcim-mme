from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import re
import numpy as np
from scipy import spatial 

# Setting which embedding and what dimension of the embedding we want
dataset = "office"
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
print(class_list)
# Loading the specific word embedding model
if embed == 'w2v' or embed == "fasttext":
    filename = '../word_embedding_models/%s-pret/%s'%(embed,embed_model)
elif embed == 'glove':
    glove_input_file = '../word_embedding_models/%s-pret/%s'%(embed,embed_model)
    word2vec_output_file = embed_model + '.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)
    filename = word2vec_output_file 

# Function for finding closest embedding vector if the label is not in the vocab
def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

# Loading the model in w2v format for harmonization across different models
model = KeyedVectors.load_word2vec_format(filename, binary=binary)
word_list = list(model.wv.vocab)
# Creating a dict of words and the corresponding word vectors
embeddings_dict = {}
for word in word_list:
    embeddings_dict[word] = model[word]
# creating a class list of original and fake labels
class_dict = {}
for class_ in class_list:
    class_dict[class_] = class_.lower()
# Handline some special cases here for words which are not present in the dict
if dataset == "multi":
    class_dict['bottlecap'] = 'bottle_cap'
    class_dict['teddy-bear'] = 'teddy_bear'


num_class =  len(class_list)
a = np.zeros((num_class,dim),dtype = float)
i = 0

total=[]
print(len(embeddings_dict))
for og_labels in class_dict.keys():
    if og_labels in embeddings_dict:
        total.append(embeddings_dict[og_labels]/np.linalg.norm(embeddings_dict[og_labels]))
        print(og_labels)
        print("computed direct label")
    else:
        mini_labels=class_dict[og_labels].split('_')
        print(mini_labels)
        final_embedding=np.sum([embeddings_dict[xlabel] for xlabel in mini_labels],axis=0)
        nearest_label=find_closest_embeddings(final_embedding)[0]
        total.append(embeddings_dict[nearest_label]/np.linalg.norm(embeddings_dict[nearest_label]))
        print(nearest_label)
        print("computed nearest neighbour")
    	
np.save('%s_%s.npy'%(dataset,embed_model),total)


