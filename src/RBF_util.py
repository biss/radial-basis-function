__author__ = 'biswajeet'

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans
import sklearn.metrics as cma
import math, random

iris = datasets.load_iris()
X = iris.data
y = list(iris.target + 1)

hidden_nodes = 10

"""fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()"""

est = KMeans(n_clusters=hidden_nodes)
est.fit(X)

"""labels = est.labels_

ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')"""
cluster_ctr = est.cluster_centers_
print "cluster centroids",cluster_ctr
#print X
#plt.show()

#new_data = np.array(shape = ())
f_res = []
for i in X:
    res = []
    #print "original data point",i
    for j in cluster_ctr:
        temp = []
        for x, z in zip(i, j):
            temp.append(x - z)
        #print "data subtracted from mu", temp
        temp = np.array(temp)
        res.append(math.exp(-np.dot(temp, temp.T)/2))
    #print "higher dimensional representation of the original data point", res
    f_res.append(res)

#print "final data form",f_res

final_data = np.full((150,hidden_nodes+2), 0.0)
eta = .1

for k in range(150):
    temp = [1]
    temp = np.concatenate((temp, f_res[k], [y[k]]))
    #print "prepared temp", temp
    final_data[k] = temp
#final_res = np.array(final_res)
print "final data with label",final_data
print "shape of final data", final_data.shape

train_data = np.zeros(shape=(120,hidden_nodes+2))
test_data = np.zeros(shape=(30,hidden_nodes+2))
l = k = 0
for i in range(150):
    if i % 5 == 0:
        test_data[l] = final_data[i]
        l += 1
    else:
        train_data[k] = final_data[i]
        k+= 1

print "train data", train_data.shape
print "test data", test_data.shape

#initial weights
u_weights = np.array([[random.uniform(-10, 10) for i in range(hidden_nodes+1)], [random.uniform(-10, 10) for i in range(hidden_nodes+1)],
                      [random.uniform(-10, 10) for i in range(hidden_nodes+1)]])
print u_weights

#matrix of weights in the last iteration, required for the termination condition
l_weights = np.full((3, (hidden_nodes + 1)), 0.0)

def train_model(label):
    print 'for class : ', label
    pdt = 0
    l_weights = np.full((3, (hidden_nodes + 1)), 0.0)
    print 'l_weights are', l_weights[label - 1]
    print 'u_weights are', u_weights[label - 1]
    # class label >= 0 vs class non-label < 0 - online learning
    #while (abs(l_weights[label - 1] - u_weights[label - 1]).any() > .0001):
    count = 20
    while count > 0:
        print "count: ", 20 - count
        #l_weights[label - 1] = u_weights[label - 1]

        # show_plot(formula, x_range)
        print 'training data and weight vector and pdt is :', final_data[0, :-1], u_weights[label - 1]\
            , np.dot(final_data[0, :-1], np.transpose(u_weights[label - 1]))

        for i in train_data:
            print 'initial weights and data vector',(u_weights, i[:-1])
            pdt = np.dot(i[:-1], np.transpose(u_weights[label - 1]))
            print 'product is ',pdt
            if (i[-1] != label and pdt >= 0):
                u_weights[label - 1] = u_weights[label - 1] - eta * i[:-1]
                print "first case : label is ",label, "wt and data", (u_weights[label-1], eta*i[:-1])

            if (i[-1] == label and pdt < 0):
                u_weights[label - 1] = u_weights[label - 1] + eta * i[:-1]
                print "second case label is ",label, "wt and data", (u_weights[label-1], eta*i[:-1])
        #print 'abs diff', abs(l_weights[label - 1] - u_weights[label - 1])
        count -= 1

def predict_model(test_data):
    pred = [None]*30
    count = 0
    #print "sayandeep",self.test_data
    for i in test_data:
        t_pdt_1 = np.dot(i[:hidden_nodes + 1], np.transpose(u_weights[0]))
        t_pdt_2 = np.dot(i[:hidden_nodes + 1], np.transpose(u_weights[1]))
        t_pdt_3 = np.dot(i[:hidden_nodes + 1], np.transpose(u_weights[2]))
        print count,"printings",t_pdt_1, t_pdt_2, t_pdt_3
        if t_pdt_1 >= 0:
            pred[count] = 1
        elif t_pdt_2 >= 0:
            pred[count] = 2
        elif t_pdt_3 >= 0:
            pred[count] = 3
        else:
            pred[count] = 0
        count += 1
    #print self.prediction
    cm = cma.confusion_matrix(test_data[:, -1], pred)
    print 'confusion matrix for the model is', cm
    print pred

train_model(1)
train_model(2)
train_model(3)

print "final weights learned", u_weights
predict_model(test_data)