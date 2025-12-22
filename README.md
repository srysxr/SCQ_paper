# SCQ_paper
This repository contains the **code and resources** for Structure-Adaptive Conformal Inference for Large-Scale Out-of-Distribution Testing. It includes simulation experiments, real-data analyses, and supplementary simulations.
You can download the datasets used in this project from the following links:
- **CICIDS dataset**: available [here](https://www.unb.ca/cic/datasets/ids-2017.html).
- **PageBlocks dataset**: available [here](https://archive.ics.uci.edu/dataset/78/page+blocks+classification).
## Folders

- `/simu`
  - `/simu/feature_dim`: Impacts of feature dimensionality (Section 5.2)
  - `/simu/dist_shift`: Performance under distribution shifts (Section 5.3)
  - `/simu/data_imb`: Performance under data imbalance (Section 5.4)

- `/real_data`
  - `/real_data/CICIDS2017`: Analysis of the cybersecurity data (Section 6.1)
  - `/real_data/PageBlocks`: Analysis of the PageBlocks data (Section 6.2)

- `/simu_sup`
  - `/simu_sup/comp_w_naive_ams`: Comparison of P-TAMS with naive AMS method (Appendix D.2)
  - `/simu_sup/comp_w_icp_ams`: Comparison of P-TAMS with ICP-AMS method (Appendix D.3)
  - `/simu_sup/fdr_adaptive_attainment`: The adaptivity in achieving the nominal FDR level (Appendix D.4)


