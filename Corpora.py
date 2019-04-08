import numpy as np

def dict_corpora_func(inverted = False):
    matras_and_sounds = list(["અ","આ","ઇ","ઈ","ઉ","ઊ","એ","ઐ","ઓ","ઔ","ઋ","ઍ","ઑ","ં","ઁ","ઃ","઼","ા","િ","ી","ુ","ૂ","ૃ","ૄ","ૅ","ે","ૈ","ૉ","ો","ૌ","્"])
    Consonants = ["ક","ખ","ગ","ઘ","ઙ","ચ","છ","જ","ઝ","ઞ","ટ","ઠ","ડ","ઢ","ણ","ત","થ","દ","હ","ધ","ન","પ","ફ","બ","ભ","મ","ય","ર","લ","ળ","વ","શ","ષ","સ","ૠ","ૡ"]
    placeholders = [" "]
    corpora = matras_and_sounds + Consonants + placeholders
    if(inverted):
        return {k:v for k,v in enumerate(corpora)}
    else:
        return {v:k for k,v in enumerate(corpora)}

def vectorise_string(str_sent, dict_corpora):
    vector = list()
    for i in str_sent:
        vector.append(dict_corpora[i])
    return np.array(vector)

def de_vectorize_string(str_recv, dict_corpora):
    vector = list()
    dict_corpora[len(dict_corpora)] = ""
    for i in str_recv:
        vector.append(dict_corpora[i])
    return ''.join(vector)
