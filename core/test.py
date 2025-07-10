import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from photon import Photon

#arquivo para escrever os testes de cada função
#escrever os testes com tempo de execução de cada função
#java_compare() deve testar os métodos python e os do JavaCV: resultado x tempo de proc.

#usar o tangela para testar o algoritmo de canny

tangela = Photon.pokefetch('tangela')
tentacruel= Photon.pokefetch('tentacruel')
terastodonte= Photon.magic_gather('terastodon')
cutilada= Photon.magic_gather('cutilada banidora')
brilliant= Photon.magic_gather('restauração brilhante')
prison= Photon.magic_gather('prisao do intercessor')
vil= Photon.magic_gather('repelir o vil')
dog= Photon.magic_gather('companheira espirituosa')

test = dog

Photon.show(test)
