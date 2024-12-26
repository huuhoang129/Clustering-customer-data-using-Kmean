from sklearn.cluster import KMeans
import pandas as pd

# Hàm tìm k tối ưu bằng cách đánh giá distortion
def find_optimal_k(data, max_k=10):
    distortions = []
    K = range(1, max_k + 1)
    threshold = 0.1  # Ngưỡng độ giảm distortion không đáng kể
    optimal_k = 1
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
        if len(distortions) > 1:
            decrease_ratio = (distortions[-2] - distortions[-1]) / distortions[-2]
            if decrease_ratio < threshold:
                optimal_k = k - 1
                break
    print("K bằng: ",optimal_k) 
 

data = pd.read_csv('file/rrr.csv')
optimal_k = find_optimal_k(data)

