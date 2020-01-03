import random
from matplotlib import style
from sklearn.cluster import  KMeans
import numpy as np
import glob
import matplotlib.pyplot as plt
style.use('ggplot')

#generates plots
def genPlot(kvals,sum_of_distances,name,col):
    plt.plot(kvals, sum_of_distances, color=col, linestyle='solid', linewidth=3, markerfacecolor='black',
             markersize=12,label = name)
    plt.xlabel('Cluster Count')
    plt.ylabel('SSE')
    plt.title('Elbow Plot')
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    #plt.show()

#runs Scikit's Kmeans with ++ initialization
def KMeansplus(program, n):
    x = np.array(program)
    kmeansplus = KMeans(n_clusters=n, random_state=0,init='k-means++').fit(x)
    labels = kmeansplus.labels_
    my_labels = list(dict.fromkeys(labels))
    my_labels = list(my_labels)
    cluster_totals = []
    SSEkmeansplus.append(kmeansplus.inertia_)
    i = 0
    while i < len(my_labels):
        count = 0
        for item in labels:
            if item == my_labels[i]:
                count += 1
        cluster_totals.append(count)
        f.write('\n')
        f.write("Cluster " + str(i + 1) + ": " + str(count) + ' programs')
        i += 1
    f.write('\n')
    f.write('KMeans++ SSE: ' + str(kmeansplus.inertia_))
#vectorizes program files
def read_in_files(file):  # Help from Carlos Samaniego
    import_count = 0
    classcount = 0
    methodcount = 0
    try_count = 0
    int_count = 0
    double_count = 0
    equal_count = 0
    for_count = 0
    while_count = 0
    if_count = 0
    comment_count = 0
    equalitycount = 0

    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if '//' in line:
                comment_count += 1
            if 'import' in line and '//' not in line:
                import_count += 1
            if '==' in line and '//' not in line:
                equalitycount += 1
            if 'def' in line and '//' not in line:
                methodcount += 1
            if 'class' in line and '//' not in line:
                classcount += 1
            if 'try' in line and '//' not in line:
                try_count += 1
            if 'int' in line and '//' not in line:
                int_count += 1
            if 'double' in line and '//' not in line:
                double_count += 1
            if '=' in line and '//' not in line:
                equal_count += 1
            if 'for' in line and '//' not in line:
                for_count += 1
            if 'while' in line and '//' not in line:
                while_count += 1
            if 'if' in line and '//' not in line:
                if_count += 1

    return [comment_count, int_count, double_count, equal_count, for_count, while_count, if_count,import_count,classcount,methodcount,try_count,equalitycount]

#https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42
#Kmeans algorithim
class K_Means:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, vectors, centroids):
        self.centroids = {}

        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []
        sum_distances = 0
        z=1
        for features in vectors:
            distances = [np.linalg.norm(list(set(features) - set(centroids[centroid]))) for centroid in centroids]
            for i in range(len(distances)):
                distances[i] = int(distances[i])

            classification = distances.index(min(distances))
            sum_distances += (min(distances)**2)
            self.classes[classification].append(features)
            if(k==2):
                f.write('\n')
                f.write('Program ' +str(z) + ' is in cluster ' + str(classification+1))
                #f.write('\n')
                z+=1
        return sum_distances, self.classes


f = open('Output.txt','w')
f.write('MINIMAL')
f.write('\n')
java_files = glob.glob('Assignment4/*.java')
vector_dict = {}
for file in java_files:
    vector_dict[file] = read_in_files(file)
vector_list = []
for file in java_files:
    vector_list.append(read_in_files(file))

k = 2
sum_of_distances = []
SSEkmeansplus = []
#displays cluster membership until 15
while (k < 15):
    f.write('\n')
    f.write(str(k) + " Clusters:")
    centroids = {}
    for i in range(k):
        centroids[i] = random.choice(vector_list)

    km = K_Means(k)
    clusters = km.fit(vector_list, centroids)
    sum_of_distances.append(clusters[0])
    for key, value in (clusters[1]).items():
        f.write('\n')
        f.write("Cluster {}: ".format(key + 1) + str(len(value)) + " programs")

    f.write('\n')
    f.write("KMeans SSE: " + str(sum_of_distances[k - 2]))
    # SCIKIT RESULTS
    f.write('\n')
    f.write('\n')
    f.write('Results from scikit Kmeans++ clustering:')
    KMeansplus(vector_list, k)
    f.write('\n')

    k += 1

genPlot(range(2,15),sum_of_distances,'Kmeans.png','black')
genPlot(range(2,15),SSEkmeansplus,'KmeansPlus.png','blue')
SSEmin = min(sum_of_distances)
thresh = float(SSEmin + (SSEmin*.05))
count = 2

#outputs optimal cluster configuration for KMeans and KMeans++
for i in sum_of_distances:
    if(i<thresh):
        print("Best number of clusters for KMeans is " + str(count)+ " for the .1 threshold level")
        break
    count+=1
SSEmin = min(SSEkmeansplus)
thresh = float(SSEmin + (SSEmin*.1))
count = 2
for i in SSEkmeansplus:
    if(i<thresh):
        print("Best number of clusters for KMeans++ is " + str(count) + " for the .1 threshold level")
        break
    count+=1











