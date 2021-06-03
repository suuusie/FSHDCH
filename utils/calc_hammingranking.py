import numpy as np
# from matplotlib import pyplot as plt

def calc_hammingDist(B1, B2):
    #used
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_map(qB, rB, query_L, retrieval_L, num_class1):
    #used
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    query_L = np.array(query_L)
    retrieval_L = np.array(retrieval_L)
    query_L = query_L[:, num_class1:len(query_L[0])]  # 第二层的label
    retrieval_L = retrieval_L[:, num_class1:len(retrieval_L[0])]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, int(tsum), int(tsum))

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map

def calc_prc(qh, rh, ql, rl, num_class1):
    #used
    # prc
    ql = ql[:, num_class1: len(ql[0])]
    rl = rl[:, num_class1: len(rl[0])]
    r_n = rh.shape[0]
    bit_n = qh.shape[1]

    hamdist = ((bit_n * 1.0 - np.dot(qh, rh.T)) / 2).astype(np.float16)
    rank = np.argsort(hamdist, 1)

    sim = (np.dot(ql, rl.T) > 0).astype(np.int)
    sim_num_of_query = np.sum(sim, 1)
    sim = np.array([s[rk] for s, rk in zip(sim, rank)], np.int)

    # pointnum = 1000
    # step = r_n // pointnum
    pointnum = r_n
    step = 1
    prc = np.zeros((2, pointnum))
    for t in range(pointnum):
        topn = (t + 1) * step
        # print('topn:%f'%topn)
        # print('sim_sum_of_query:')
        # print(sim_num_of_query)
        s = np.sum(sim[:, 0: topn], 1)
        prc[1][t] = np.mean(s / topn) #precision
        prc[0][t] = np.mean(s / sim_num_of_query) #recall

    mR = prc[0][:]
    mR = np.insert(mR,0,0) #insert 0 to the start
    mP = prc[1][:]
    mP = np.insert(mP,0,mP[0])

    recall = np.linspace(0,1, num=1001) # 0-1 step:0.001
    precision = interpolate_pr(mR, mP, recall)

    precision_recall = np.zeros((2, 1001))
    precision_recall[0][:] = precision
    precision_recall[1][:] = recall
    return precision_recall

def interpolate_pr(r, p, recs):
    #used
    n = p.shape[0]
    precision = []
    for j in range(recs.shape[0]):
        rec = recs[j]
        done = 0
        for i in range(n-1):
            if((r[i] <= rec) & (rec <= r[i+1])):
                done = 1
                if(r[i] == r[i+1]):
                    # precision[j] = (p[i] + p[i+1]) / 2
                     precision.append((p[i] + p[i+1]) / 2)
                else:
                    # precision[j] = p[i] + ((rec - r[i]) * (p[i+1] - p[i])) / (r[i+1] - r[i])
                     precision.append(p[i] + ((rec - r[i]) * (p[i+1] - p[i])) / (r[i+1] - r[i]))
                break
        if done == 0:
            print('not done! for %f'%rec)

    return precision











def calc_pre_rec(qh, rh, ql, rl, num_class1):
    # prc
    ql = ql[:, num_class1: len(ql[0])]
    rl = rl[:, num_class1: len(rl[0])]
    r_n = rh.shape[0]
    bit_n = qh.shape[1]

    hamdist = ((bit_n * 1.0 - np.dot(qh, rh.T)) / 2).astype(np.float16)
    rank = np.argsort(hamdist, 1)

    sim = (np.dot(ql, rl.T) > 0).astype(np.int)
    sim_num_of_query = np.sum(sim, 1)
    sim = np.array([s[rk] for s, rk in zip(sim, rank)], np.int)

    pointnum = 1000
    step = r_n // pointnum
    prc = np.zeros((2, pointnum))
    for t in range(pointnum):
        topn = (t + 1) * step
        print('topn:%f'%topn)
        print('sim_sum_of_query:')
        print(sim_num_of_query)
        s = np.sum(sim[:, 0: topn], 1)
        prc[1][t] = np.mean(s / topn) #precision
        prc[0][t] = np.mean(s / sim_num_of_query) #recall
        # Curve( [prc]).draw()





    return prc

def calc_map_3layer(qB, rB, query_L, retrieval_L,num_class1, num_class2, num_class3):
    # different layer with different weight
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    alpha1 = 0.1
    alpha2 = 0.3
    alpha3 = 0.6

    num_query = query_L.shape[0]
    query_L = np.array(query_L)
    retrieval_L = np.array(retrieval_L)
    # query_L = query_L[:, 14:len(query_L[0])]  # 第二层的label
    # retrieval_L = retrieval_L[:, 14:len(retrieval_L[0])]
    map = 0
    map1 = 0
    map2 = 0
    map3 = 0
    for iter in range(num_query):
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd3 = (np.dot(query_L[iter, (num_class1+num_class2):(num_class1+num_class2+num_class3)], retrieval_L[:,(num_class1+num_class2):(num_class1+num_class2+num_class3)].transpose()) > 0).astype(np.float32)
        gnd2 = (np.dot(query_L[iter, num_class1:(num_class1+num_class2)], retrieval_L[:,num_class1: (num_class1+num_class2)].transpose()) > 0).astype(np.float32)
        gnd1 = (np.dot(query_L[iter, :num_class1], retrieval_L[:,:num_class1].transpose()) > 0).astype(np.float32)

        temp = gnd2 - gnd3
        temp = (temp == 1).astype(np.float32)
        gnd2 = temp

        temp1 = gnd1 - gnd2 - gnd3
        temp1 = (temp1 == 1).astype(np.float32)
        gnd1 = temp1

        gnd1 = gnd1[ind]
        gnd2 = gnd2[ind]
        gnd3 = gnd3[ind]

        # gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum3 = np.sum(gnd3)
        tsum2 = np.sum(gnd2)
        tsum1 = np.sum(gnd1)
        if tsum1 == 0:
            continue

        if tsum3 != 0:
            count3 = np.linspace(1, int(tsum3), int(tsum3))
            tindex3 = np.asarray(np.where(gnd3 == 1)) + 1.0
            map3 = np.mean(count3 / tindex3)

        if tsum2 != 0:
            count2 = np.linspace(1, int(tsum2), int(tsum2))
            tindex2 = np.asarray(np.where(gnd2 == 1)) + 1.0
            map2 = np.mean(count2 / tindex2)

        count1 = np.linspace(1, int(tsum1), int(tsum1))
        tindex1 = np.asarray(np.where(gnd1 == 1)) + 1.0
        map1 = np.mean(count1 / tindex1)

        map = map + alpha1 * map1 + alpha2 * map2 + alpha3 * map3

    map = map / num_query
    return map

def calc_map_2layer(qB, rB, query_L, retrieval_L,num_class1, num_class2):
    # different layer with different weight
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    alpha1 = 0.3
    alpha2 = 0.7

    num_query = query_L.shape[0]
    query_L = np.array(query_L)
    retrieval_L = np.array(retrieval_L)
    # query_L = query_L[:, 14:len(query_L[0])]  # 第二层的label
    # retrieval_L = retrieval_L[:, 14:len(retrieval_L[0])]
    map = 0
    map1 = 0
    map2 = 0

    for iter in range(num_query):
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd2 = (np.dot(query_L[iter, num_class1:(num_class1+num_class2)], retrieval_L[:,num_class1: (num_class1+num_class2)].transpose()) > 0).astype(np.float32)
        gnd1 = (np.dot(query_L[iter, :num_class1], retrieval_L[:,:num_class1].transpose()) > 0).astype(np.float32)

        temp = gnd1 - gnd2
        temp = (temp == 1).astype(np.float32)
        gnd1 = temp

        gnd1 = gnd1[ind]
        gnd2 = gnd2[ind]


        # gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum2 = np.sum(gnd2)
        tsum1 = np.sum(gnd1)
        if tsum1 == 0:
            continue


        if tsum2 != 0:
            count2 = np.linspace(1, int(tsum2), int(tsum2))
            tindex2 = np.asarray(np.where(gnd2 == 1)) + 1.0
            map2 = np.mean(count2 / tindex2)

        count1 = np.linspace(1, int(tsum1), int(tsum1))
        tindex1 = np.asarray(np.where(gnd1 == 1)) + 1.0
        map1 = np.mean(count1 / tindex1)

        map = map + alpha1 * map1 + alpha2 * map2

    map = map / num_query
    return map


def calc_map_3layer_new(qB, rB, query_L, retrieval_L,num_class1, num_class2, num_class3):
    # +1 0 -1
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    alpha1 = 0.1
    alpha2 = 0.3
    alpha3 = 0.6

    num_query = query_L.shape[0]
    query_L = np.array(query_L)
    retrieval_L = np.array(retrieval_L)
    # query_L = query_L[:, 14:len(query_L[0])]  # 第二层的label
    # retrieval_L = retrieval_L[:, 14:len(retrieval_L[0])]
    map = 0
    map1 = 0
    map3 = 0
    for iter in range(num_query):
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd3 = (np.dot(query_L[iter, (num_class1+num_class2):(num_class1+num_class2+num_class3)], retrieval_L[:,(num_class1+num_class2):(num_class1+num_class2+num_class3)].transpose()) > 0).astype(np.float32)
        gnd2 = (np.dot(query_L[iter, num_class1:(num_class1+num_class2)], retrieval_L[:,num_class1: (num_class1+num_class2)].transpose()) > 0).astype(np.float32)
        gnd1 = (np.dot(query_L[iter, :num_class1], retrieval_L[:,:num_class1].transpose()) > 0).astype(np.float32)

        temp = gnd1 + gnd2 + gnd3
        temp = (temp == 0).astype(np.float32)
        gnd1 = temp

        gnd1 = gnd1[ind]
        gnd3 = gnd3[ind]

        # gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum3 = np.sum(gnd3)
        tsum1 = np.sum(gnd1)

        if tsum1 !=0:
            count1 = np.linspace(1, int(tsum1), int(tsum1))
            tindex1 = np.asarray(np.where(gnd1 == 1)) + 1.0
            map1 = np.mean(count1 / tindex1)


        if tsum3 != 0:
            count3 = np.linspace(1, int(tsum3), int(tsum3))
            tindex3 = np.asarray(np.where(gnd3 == 1)) + 1.0
            map3 = np.mean(count3 / tindex3)


        map = map + (map3 - map1)

    map = map / num_query
    return map

def calc_top_n(qB, rB, query_L, retrieval_L, n):

    num_query = query_L.shape[0]
    query_L = np.array(query_L)
    retrieval_L = np.array(retrieval_L)
    query_L = query_L[:, 14:len(query_L[0])]  # 第二层的label
    retrieval_L = retrieval_L[:, 14:len(retrieval_L[0])]
    res = 0
    topn = []
    for iter in range(num_query):
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if (tsum == 0):
            continue;
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        topn.append([iter,ind[0:n] + 3000])
        print('iter: ', iter, '    ind:   ',ind[0:n])
        gnd = gnd[ind]
        gnd = gnd[0:n]
        nsum = np.sum(gnd)
        # print(nsum)
        res += nsum / n
    res = res / num_query
    return topn

def calc_first_n(qB, rB, query_L, retrieval_L, n):
    num_query = query_L.shape[0]
    res = 0
    for iter in range(num_query):
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        print('iter: ', iter, '   ind :   ', ind[iter])
        if ind[iter] < n:
            res = res + 1
    res = res / num_query
    return res


if __name__ == '__main__':
    qB = np.array([[1, -1, 1, 1],
                   [-1, -1, -1, 1],
                   [1, 1, -1, 1]])
    rB = np.array([[1, -1, 1, -1],
                   [-1, -1, 1, -1],
                   [-1, -1, 1, -1],
                   [1, 1, -1, -1],
                   [-1, 1, -1, -1],
                   [1, 1, -1, 1]])
    query_L = np.array([[1, 1, 0, 1, 0, 0],
                        [1, 1, 0, 0, 1, 0],
                        [1, 0, 1, 0, 0, 1]])
    retrieval_L = np.array([[1, 1, 0, 1, 0, 0],
                            [1, 1, 0, 0, 1, 0],
                            [1, 1, 0, 0, 1, 0],
                            [1, 1, 0, 0, 1, 0],
                            [1, 0, 1, 0, 0, 1],
                            [1, 0, 1, 0, 0, 1]])

    # map = calc_map(qB, rB, query_L, retrieval_L)
    # map1 = calc_map_2layer(qB, rB, query_L, retrieval_L,1,5)
    # map2 = calc_map_2layer_new(qB, rB, query_L, retrieval_L,1,5)
    # map3 = calc_map_3layer_new(qB, rB, query_L, retrieval_L,1,2,3)
    prc = calc_prc(qB, rB, query_L, retrieval_L,2)
    #
    # # print(map)
    # print(map1)
    # print(map2)
    # print(map3)

    # tsum = 1000
    #
    # recall = np.linspace(0,1, num=1001)
    # print(recall)
