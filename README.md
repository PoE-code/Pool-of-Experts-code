# Pool-of-Experts-code

This is the authors' implementation of the following paper:

"Pool of Experts: Realtime Querying Specialized Knowledge in Massive Neural Networks", SIGMOD, 2021

# Additional experimental result

### 5.3 Experiments on Model Consolidation

<table>
<td> Comparison between soft loss and scale loss </td>
<tr>
<td><img src = 'addImg/table6_cross_expert_errors.PNG' height = '300px'></td>
</tr>
</table>
*L*
We conduct an experiment to examine whether our $\L_{scale}$ loss can help to address the logit scale problem, as described in Section \ref{sec:main:service}. To this end, we focus on a specific type of errors caused by wrong experts that take a highest probability away from the correct expert, which we call \textit{cross-expert errors}. This type of errors are distinguished from those locally made within the classes of an expert, and cross-expert errors are more likely to happen when logit scales of experts being merged are quite different.

In order to see the effectiveness of $\L_{scale}$, we build a different group of experts intentionally trained by only $\L_{soft}$, and build task-specific models upon them by the same knowledge consolidation method. We then look into how the proportion of cross-expert errors to all the errors has been changed after we add $\L_{scale}$ when extracting experts. In Table \ref{tab:additional misclassification}, we can observe that the rates of cross-expert errors are reduced when we use $\L_{scale}$ along with $\L_{soft}$. Thus, the experts trained by both $\L_{soft}$ and $\L_{scale}$ turn out to be more robust to cross-expert errors than those trained by only $\L_{soft}$. As the problem of overconfident experts has already been resolved by $\L_{soft}$, we can surely claim that minimizing the $\mathcal{L}_{scale}$ loss is effective to mitigate the logit scale problem.

We compared the merge results using PoE for $E_i$ trained using only $\mathcal{L}_{soft}$ and $\mathcal{L}_{scale}$ added $\mathcal{L}_{CKD}$, respectively, to see the effect of $\mathcal{L}_{scale}$ in Section \ref{sec:main:preprocess}. The results are shown in Table \ref{tab:additional misclassification}. The results showed the average of the results for each combination. In Table \ref{tab:additional misclassification} \textit{out mis/all mis} means the rate at which the misclassification occurred due to having the largest logit value in the wrong $M(H_j)$ other than the model $M(H_i)$ of the $H_i$ to which each image belongs. $\mathcal{L}_{soft}$ can confirm that the \textit{out mis/all mis} are higher than the $\mathcal{L}_{CKD}$ in both CIFAR-100 and Tiny-ImageNet, even though the high confidence issue has been resolved.


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
