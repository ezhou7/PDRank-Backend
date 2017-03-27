from scipy.cluster.vq import vq, kmeans, whiten

# features = array([[1.1,2.2,3.3,4.4],[100.1,200.2,300.3,400.4],[1000.1,2000.2,3000.3,4000.4]])


class Clustering:
    @staticmethod
    def k_means(vecs):
        n = 2
        whitened = whiten(vecs)
        code_book, _ = kmeans(whitened, n)
        cluster_num, _ = vq(whitened, code_book)
        print(cluster_num)