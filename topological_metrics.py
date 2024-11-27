from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape
from gtda.plotting import plot_diagram
import matplotlib.pyplot as plt
import numpy as np
import torch 

model = "Encoder"

ref_pcs = torch.load(f"./results/{model}/ref_chair.pt").cpu().numpy()
smp_pcs = torch.load(f"./results/{model}/samples_chair.pt").cpu().numpy()

# Sample 100 pts 
idxs = np.random.choice(list(range(2048)),500)


PL = PersistenceLandscape()
VR = VietorisRipsPersistence(homology_dimensions=[0, 1])  # Parameter explained in the text

features = []
labels = []
for batch_idx in range(0,662,50):
    print(batch_idx)
    ref_pcs_sub = ref_pcs[batch_idx:batch_idx+50,idxs,:]
    smp_pcs_sub = smp_pcs[batch_idx:batch_idx+50,idxs,:]

    point_clouds = np.vstack([ref_pcs_sub,smp_pcs_sub])
    labels.append(np.array(10 * [0] + 10 * [1]))
    diagrams = VR.fit_transform(point_clouds)

    features.append(PL.fit_transform(diagrams))


np.save(f"./results/TopologicalMetrics/features_{model}.npy",np.vstack(features))
np.save(f"./results/TopologicalMetrics/labels_{model}.npy",np.vstack(labels))
#
# # for pcl in features: 
# #     print(pcl)
# #     plt.plot(pcl[0])
# #
# # plt.show()
# #
# #
