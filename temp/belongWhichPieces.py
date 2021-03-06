#-*- coding: UTF-8 -*-
from scipy.linalg.misc import norm
import numpy as np
def belongWhichPieces(data, Pieces, numberOfPieces, *otherArg):

    if numberOfPieces == 0:
        return -1
    rightSubSpace = Pieces['rightSubspace']

    v = data / norm(data, 2)
    tmp_w = np.zeros(numberOfPieces)

    for j in range(numberOfPieces):
        U = rightSubSpace[j].transpose()
        V = v * U
        weight = V * V.transpose()
        tmp_w[j] = weight
    w_id = np.argsort(tmp_w)
    w_v = tmp_w[w_id]
    if len(otherArg) > 0 and w_v[-1] < otherArg[0]:
        belongsId = -1
    else:
        belongsId = w_id[-1]

    return belongsId