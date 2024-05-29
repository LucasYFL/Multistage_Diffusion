import numpy as np
from math import floor, ceil
# num = 12
# k = 5
# def time_cost(arr):
#     center = arr[(len(arr))//2]
#     # center = np.mean(arr)
#     # center = np.median(arr)
#     # assert np.array_equal(np.abs(center-arr)**2,(center-arr)**2)
#     return np.sum(np.abs(center-arr))
# mi = floor(num/(2*k))
# MI = ceil(3*num/(2*k))
# # timesteps =( np.arange(0,num)+1)

# def time_cost(arr):
#     center = np.mean(arr)
#     return np.sum((center-arr)**2)
# timesteps =np.array([1,2,3,10,12,13,25,26,100,101,1001,1002])
# D = np.zeros((num,k))
# B = np.zeros_like(D,dtype=int)
# D[:,:] = np.inf
# D[:,0] = [time_cost(x) for x in [timesteps[:i] for i in range(1,num+1)]]
# for m in range(1,k):
#     for i in range(m,num):
#         lcs = [time_cost(x) for x in [timesteps[t:i+1] for t in range(m,i+1)]]
#         temp = D[m-1:i,m-1] + lcs
#         # minj = np.where(temp == temp.min())[0][0]# np.argmin(temp)
#         minj = np.argmin(temp)
#         D[i,m] = temp[minj]
#         B[i,m] = minj+m
# for m in range(1,k):
#     lower = 1+(m-1)*mi
#     for i in range(m,num):
#         lb = max(lower,i+1-MI)
#         ub = i+1-mi
#         if ub < lb:
#             continue
#         lcs = [time_cost(x) for x in [timesteps[t:i+1] for t in range(lb,ub+1)]]
#         # print(lb,ub,mi)
#         temp = D[lb-1:ub,m-1] + lcs
#         minj = np.argmin(temp)
#         D[i,m] = temp[minj]
#         B[i,m] = minj+lb
# lms = [B[num-1,k-1]]
# temp = list(range(k-1))
# temp.reverse()
# for m in temp:
#     lms.append(B[lms[-1]-1,m])
# print(lms)
# print(timesteps[lms])

class interval_cluster:
    def __init__(self,timesteps,k):
        self.timesteps = timesteps
        self.k = k
        self.num = len(timesteps)
        self.mi = floor(self.num/(2*self.k))
        self.MI = ceil(3*self.num/(2*self.k))
    def calculate(self,time_cost):
        mi = self.mi
        MI = self.MI
        num = self.num
        k = self.k
        D = np.zeros((self.num,self.k))
        B = np.zeros_like(D,dtype=int)
        D[:,:] = np.inf
        D[:,0] = [time_cost(x) for x in [self.timesteps[:i] for i in range(1,self.num+1)]]
        for m in range(1,self.k):
            lower = 1+(m)*mi
            start = num-1 if m == self.k-1 else m
            for i in range(start,num):
                lb = max(lower,i+1-MI)
                ub = i+1-mi
                if ub < lb:
                    continue
                lcs = [time_cost(x) for x in [self.timesteps[t:i+1] for t in range(lb,ub+1)]]
                # print(lb,ub,mi)
                temp = D[lb-1:ub,m-1] + lcs
                minj = np.argmin(temp)
                D[i,m] = temp[minj]
                B[i,m] = minj+lb 
        lms = [B[num-1,k-1]]
        temp = list(range(k-1))
        temp.reverse()
        for m in temp:
            lms.append(B[lms[-1]-1,m])
    
        return lms,self.timesteps[lms]

# def time_cost(arr):
#     center = arr[(len(arr))//2]
#     return np.sum(np.abs(center-arr))
# timesteps =( np.arange(0,100)+1)
# cl = interval_cluster(timesteps,5)
# print(cl.calculate(time_cost))
