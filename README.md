# PowerFlowNet
Leveraging Message Passing GNNs for High-Quality Power Flow Approximation.

![image](https://github.com/StavrosOrf/PoweFlowNet/assets/17108978/1a6398c5-cac6-40cf-a3a1-0bc8fb66a0dc)


PowerFlowNet's distinctiveness, compared to existing PF GNN approaches, lies in its adept utilization of the capabilities from message-passing GNNs and high-order GCNs in a unique arrangement called PowerFlowConv, for handling a trainable masked embedding of the network graph. This innovative approach renders PoweFlowNet remarkably scalable, presenting an effective solution for the PF problem.

The **PowerFlowNet Paper** can be found at: [link](https://www.sciencedirect.com/science/article/pii/S0142061524003338) 

### Description

PowerFlowNet transforms the PF into a GNN node-regression problem by representing each bus as a node and each transmission line as an edge while maintaining the network's connectivity.

![image](https://github.com/StavrosOrf/PoweFlowNet/assets/17108978/3c3314c8-c111-41a7-8eb6-2116533f7f72)


### Instructions


To train a model run train.py with the desired arguments. For example:
```
python3 train.py --cfg_json ./configs/standard.json\
                --num-epochs 2000\
                --data-dir ./data/
                --batch-size 128\
                --train_loss_fn mse_loss\
                --lr 0.001\
                --case 118v2\
                --model MaskEmbdMultiMPN\
                --save
```


### Datasets

Follow the links below to download the datasets and the trained models used in the paper.

[Dataset link](https://surfdrive.surf.nl/files/index.php/s/Qw4RHLvI2RPBIBL)

[Trained models link](https://surfdrive.surf.nl/files/index.php/s/iunfVTGsABT5NaD)



### File Structure
runnable files:
- `train.py` trains the model
- `results.py` plots the results
- and more scripts to generate results and plots ...

# Useful Information
First two dimensions out of seven in `edge_features` are `from_node` and `to_node`, and they are indexed from $1$. This is processed in the `PowerFlowData` dataset class. It is reindexed from $0$ and the `from_node` and `to_node` are removed from the `edge_features` tensor.

Raw data format: 
| Number | Description |
| --- | --- |
| N | number of nodes |
| E | number of edges |
| Fn = 9 | number of features per node |
| Fe = 5 | orginally 7, first two dims are `from_node` and `to_node` number of features per edge |
| Fn_out = 8 | number of output features per node |

| Tensor | Dimension |
| --- | --- |
| `Data.x` | (batch_size*N, Fe) |
| `Data.edge_index` | (2, E) |
| `Data.edge_attr` | (E, Fe) |
| `Data.y` | (batch_size*N, Fn) |


### Citation

If you use parts of this framework, datasets, or trained models, please cite as:
```
@article{LIN2024110112,
  title = {PowerFlowNet: Power flow approximation using message passing Graph Neural Networks},
  journal = {International Journal of Electrical Power & Energy Systems},
  volume = {160},
  pages = {110112},
  year = {2024},
  issn = {0142-0615},
  doi = {https://doi.org/10.1016/j.ijepes.2024.110112},
  author = {Nan Lin and Stavros Orfanoudakis and Nathan Ordonez Cardenas and Juan S. Giraldo and Pedro P. Vergara},
}
```

### We found serveral possible errors in your codebase.

#### 1. In _/dataset_generator.py_  

  **function get_trafo_z_pu()**
![下载](https://github.com/user-attachments/assets/3e83b1e6-12b6-4b15-b8c2-176493b29a79)

I noticed, in lines 56, 57(yellow marks), they multiplied 1000, which is weird.
We drew the scatter plot of the values of r and x of each edge, and found the r and x of the transformer lines are extremely large.
```
data_v2_edge = np.load('   YOUR PATH    /PoweFlowNet/data/raw/case118v2_edge_features.npy')

data_v2_edge_r = pd.DataFrame(data_v2_edge[:,:,2])

means = data_v2_edge_r.mean()
std_devs = data_v2_edge_r.std()

plt.figure(figsize=(10, 6))

for col in data_v2_edge_r.columns:
    plt.scatter([col]*len(data_v2_edge_r[col]), data_v2_edge_r[col], color='lightblue', alpha=0.6, edgecolor='k')

plt.plot(data_v2_edge_r.columns, means, marker='o', linestyle='-', color='blue', label='Mean')

plt.errorbar(data_v2_edge_r.columns, means, yerr=std_devs, fmt='o', color='blue', capsize=5, label='Mean ± 1 Std Dev')

plt.xlabel('Index (Columns)')
plt.ylabel('Values')

plt.title('data_v2_edge_r  Individual Values, Mean and Standard Deviation Range per Column')

plt.legend()
plt.grid(True)
plt.show()
```
![下载 (1)](https://github.com/user-attachments/assets/2abee28b-6eb5-4753-a63c-0fb375a5925b)

```
data_v2_edge = np.load('   YOUR PATH    /PoweFlowNet/data/raw/case118v2_edge_features.npy')

data_v2_edge_x = pd.DataFrame(data_v2_edge[:,:,3])

means = data_v2_edge_x.mean()
std_devs = data_v2_edge_x.std()

plt.figure(figsize=(10, 6))

for col in data_v2_edge_x.columns:
    plt.scatter([col]*len(data_v2_edge_x[col]), data_v2_edge_x[col], color='lightblue', alpha=0.6, edgecolor='k')

plt.plot(data_v2_edge_x.columns, means, marker='o', linestyle='-', color='blue', label='Mean')

plt.errorbar(data_v2_edge_x.columns, means, yerr=std_devs, fmt='o', color='blue', capsize=5, label='Mean ± 1 Std Dev')

plt.xlabel('Index (Columns)')
plt.ylabel('Values')

plt.title('data_v2_edge_x  Individual Values, Mean and Standard Deviation Range per Column')

plt.legend()
plt.grid(True)
plt.show()
```
![5baac7c6-ec9c-4349-9d48-95e33ec20d2c](https://github.com/user-attachments/assets/de40a670-4088-4d00-ae8e-b81c49fa38c4)

Once you remove the "*1000", the r and x of the transformers will change back to their original level. Same with the values in PYPOWER.
**However, we don't know whether you are mean to do this!!!**

#### 2. And, in line 60(bule marks), it should be changed to the same as line 73: "`return r_pu, x_pu`" 

#### 3. In _/utils/evaluation.py_

**Function evaluate_epoch_v2()**
![下载 (2)](https://github.com/user-attachments/assets/f9da6828-0a4e-4663-81fd-67bb1bb04e9b)

In line 162: the first batch of the loss value did not time len(data). This will make the total loss smaller than the actual value.
This should be modified as:
`   total_loss_terms = {key: value.item()*len(data) for key, value in loss_terms.items()}`

### EXTRA
We are a dedicated AI team under the China Southern Power Grid, and we are keen to connect with the best teams worldwide who share our passion for powerflow calculations. We believe that collaboration and the exchange of ideas can lead to groundbreaking advancements in this field. If you are interested in exploring potential cooperation, we would be thrilled to hear from you. Please feel free to reach out to me at 572120779@qq.com
Looking forward to connecting with you!
