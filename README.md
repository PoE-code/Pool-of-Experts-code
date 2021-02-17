# Pool-of-Experts-code

This is the authors' implementation of the following paper:

"Pool of Experts: Realtime Querying Specialized Knowledge in Massive Neural Networks", SIGMOD, 2021

# Additional experimental result 
<table> 
    <thead> 
     <tr> 
      <th rowspan=2>subset</th>
      <th colspan=2>subNetwork</th>
      <th colspan=2>OriginalNetwork</th>
      <th rowspan=2>A/B (%)</th>
     </tr>
     <tr> 
      <th># Params (A)</th>
      <th>Test - acc</th>
      <th># Params (B)</th>
      <th>Test - acc</th>
     </tr>
    </thead> 
    <tbody align='center'> 
     <tr> 
      <td colspan=6>Using MNISTdataset Network</td> 
     </tr>
     <tr> 
      <td>[1, 4]</td>
      <td>21,251</td>
      <td>0.999</td>
      <td rowspan=4>124,825</td>
      <td>0.983</td>
      <td>17.02</td>
     </tr>
     <tr> 
      <td>[0, 2, 6]</td>
      <td>29,947</td>
      <td>0.992</td>
      <td>0.980</td>
      <td>23.99</td>
     </tr>
     <tr> 
      <td>[0, 4, 6, 7]</td>
      <td>40,243</td>
      <td>0.992</td>
      <td>0.981</td>
      <td>32.24</td>
     </tr>
     <tr> 
      <td>ALL</td>
      <td>124,825</td>
      <td>0.979</td>
      <td>0.979</td>
      <td>100.</td>
     </tr>
     <tr> 
      <td colspan=6>Using FashionMNISTdataset Network</td> 
     </tr>
     <tr> 
      <td>subset for shoes categories [5, 7, 9]</td>
      <td>87,147</td>
      <td>0.963</td>
      <td rowspan=2>330,670</td>
      <td>0.962</td>
      <td>26.35</td>
     </tr>
     <tr> 
      <td>ALL</td>
      <td>330,670</td>
      <td>0.911</td>
      <td>0.911</td>
      <td>100.</td>
     </tr>
    </tbody> 
</table>
### 5.3 Experiments on Model Consolidation - Comparison between soft loss and scale loss

# Quick Start: CIFAR-100
We provide an CIFAR-100 example for Pool of Experts

### Preprocessing phase
    python Run_preprocessing.py

We provide Oracle in DB_pretrained, library and some experts in DB_PoE

You can check the accuracy of Oracle and each model for primitive tasks by executing the above command

### Service phase
    python Run_Service.py --queriedTask <primitive tasks>
*Example*: `python Run_Service.py --queriedTask people vehicles_1 vehicles_2`

*Available primitive tasks*: 

    'aquatic_mammals', 'flowers', 'large_carnivores', 'large_omnivores_and_herbivores', 'medium-sized_mammals', 'people', 'small_mammals', 'vehicles_1', 'vehicles_2'

You can check the accuracy of model for queried composite task by executing the above command

# Implementation and training details
All algorithms were implemented using PyTorch and evaluated on a machine with an NVIDIA Quadro RTX 6000 and Intel Core Xeon Gold 5122.

When training all the models, we use a stochastic gradient descent (SGD) with 0.9 momentum and the weight decay of L-2 regularization was fixed to 5e-4. 

The batch size of all networks was set to 512.

In all networks, we set the temperature T for distillation to 4 and the weight parameter alpha for scale loss to 0.3.

In order to extract each expert by training primitive models, we train only the expert part (i.e., conv 4) for 100 epochs, where the initial learning rate is set to 0.1 and reduced by 0.1 times at 40 and 80 epochs. The setting of transfer learning is the same.

When training a target architecture without using the library part, the learning process was continued for 200 epochs, where the initial learning rate was 0.1 and reduced by 0.1 times at 80 and 160 epochs. 

In the WRN architecture for Tiny-ImageNet experiments, the stride of the first convolutional layer is set to 2 because the input size of Tiny-ImageNet is (64, 64, 3).
