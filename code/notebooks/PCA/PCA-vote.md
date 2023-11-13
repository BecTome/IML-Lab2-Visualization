# Visualization on Vote Dataset

# 1. Load Data




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>handicapped-infants_n</th>
      <th>handicapped-infants_y</th>
      <th>water-project-cost-sharing_n</th>
      <th>water-project-cost-sharing_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Analyze your PCA algorithm in three data sets

In this case, the analysis is done in the Vote dataset. The dataset is loaded and the PCA algorithm is applied to the dataset. The results are shown in the following figures.

All the features are categorical so no previous standardization is needed. Inside the algorithm, data centralization is done.

To check and analyze the results, all components are preserved even if only the first two are shown in the figures.

## 2.1. Handmade PCA




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.538132</td>
      <td>0.208806</td>
      <td>-0.664641</td>
      <td>-0.174588</td>
      <td>0.450081</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.566310</td>
      <td>0.385575</td>
      <td>0.730238</td>
      <td>0.070440</td>
      <td>0.545899</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.181888</td>
      <td>1.863804</td>
      <td>-0.193548</td>
      <td>-0.056694</td>
      <td>-0.517372</td>
    </tr>
  </tbody>
</table>
</div>



Given that the data is implicitly centered but not standardized, the mean of the data is zero but its variance is not one. It doesn't affect the results but it is important to take into account when analyzing the results.




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
      <th>PC6</th>
      <th>PC7</th>
      <th>PC8</th>
      <th>PC9</th>
      <th>PC10</th>
      <th>PC11</th>
      <th>PC12</th>
      <th>PC13</th>
      <th>PC14</th>
      <th>PC15</th>
      <th>PC16</th>
      <th>PC17</th>
      <th>PC18</th>
      <th>PC19</th>
      <th>PC20</th>
      <th>PC21</th>
      <th>PC22</th>
      <th>PC23</th>
      <th>PC24</th>
      <th>PC25</th>
      <th>PC26</th>
      <th>PC27</th>
      <th>PC28</th>
      <th>PC29</th>
      <th>PC30</th>
      <th>PC31</th>
      <th>PC32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
      <td>4.350000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.429253e-16</td>
      <td>-1.250596e-16</td>
      <td>6.061562e-17</td>
      <td>2.261760e-16</td>
      <td>-2.779386e-16</td>
      <td>-1.100014e-16</td>
      <td>-1.033656e-16</td>
      <td>-2.071778e-16</td>
      <td>2.982927e-17</td>
      <td>-2.590520e-17</td>
      <td>-2.399103e-17</td>
      <td>-8.616990e-17</td>
      <td>9.245478e-17</td>
      <td>5.252822e-17</td>
      <td>-1.351409e-16</td>
      <td>-1.595148e-17</td>
      <td>-8.495758e-17</td>
      <td>3.730094e-16</td>
      <td>5.206563e-17</td>
      <td>3.440734e-17</td>
      <td>-5.385220e-17</td>
      <td>-3.002707e-16</td>
      <td>1.658954e-18</td>
      <td>4.787039e-17</td>
      <td>8.026785e-17</td>
      <td>1.653530e-16</td>
      <td>-8.039546e-18</td>
      <td>-1.805070e-16</td>
      <td>-1.919282e-16</td>
      <td>1.183281e-16</td>
      <td>-4.220762e-17</td>
      <td>-2.193648e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.899175e+00</td>
      <td>8.106390e-01</td>
      <td>7.187142e-01</td>
      <td>6.343247e-01</td>
      <td>5.914352e-01</td>
      <td>5.287056e-01</td>
      <td>4.995937e-01</td>
      <td>4.727407e-01</td>
      <td>4.600772e-01</td>
      <td>4.272869e-01</td>
      <td>4.051163e-01</td>
      <td>3.777704e-01</td>
      <td>3.584728e-01</td>
      <td>3.218303e-01</td>
      <td>3.020834e-01</td>
      <td>2.787719e-01</td>
      <td>2.528397e-01</td>
      <td>2.077362e-01</td>
      <td>1.654829e-01</td>
      <td>1.568745e-01</td>
      <td>1.523889e-01</td>
      <td>1.451530e-01</td>
      <td>1.400680e-01</td>
      <td>1.200577e-01</td>
      <td>1.141221e-01</td>
      <td>1.070284e-01</td>
      <td>1.063907e-01</td>
      <td>9.479078e-02</td>
      <td>8.929495e-02</td>
      <td>8.198097e-02</td>
      <td>7.294299e-02</td>
      <td>6.328710e-02</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.674723e+00</td>
      <td>-1.830662e+00</td>
      <td>-1.672052e+00</td>
      <td>-1.657765e+00</td>
      <td>-2.113420e+00</td>
      <td>-1.822915e+00</td>
      <td>-1.848776e+00</td>
      <td>-1.564779e+00</td>
      <td>-1.397251e+00</td>
      <td>-1.713605e+00</td>
      <td>-1.458133e+00</td>
      <td>-1.334529e+00</td>
      <td>-1.489735e+00</td>
      <td>-1.066851e+00</td>
      <td>-8.394470e-01</td>
      <td>-3.607605e-01</td>
      <td>-1.016883e+00</td>
      <td>-7.349914e-01</td>
      <td>-7.305911e-01</td>
      <td>-6.300715e-01</td>
      <td>-5.636544e-01</td>
      <td>-7.710158e-01</td>
      <td>-7.061491e-01</td>
      <td>-4.506641e-01</td>
      <td>-5.044013e-01</td>
      <td>-6.756387e-01</td>
      <td>-6.611841e-01</td>
      <td>-5.652254e-01</td>
      <td>-4.446878e-01</td>
      <td>-3.755918e-01</td>
      <td>-4.719187e-01</td>
      <td>-6.265121e-01</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-2.093480e+00</td>
      <td>-5.123059e-01</td>
      <td>-5.248992e-01</td>
      <td>-4.036364e-01</td>
      <td>-3.751128e-01</td>
      <td>-3.651273e-01</td>
      <td>-2.571362e-01</td>
      <td>-2.565243e-01</td>
      <td>-2.690824e-01</td>
      <td>-1.846318e-01</td>
      <td>-2.163167e-01</td>
      <td>-1.479769e-01</td>
      <td>-1.551346e-01</td>
      <td>-1.407614e-01</td>
      <td>-1.421797e-01</td>
      <td>-1.483616e-01</td>
      <td>-8.364694e-02</td>
      <td>-1.944183e-02</td>
      <td>-3.924491e-02</td>
      <td>-6.663376e-02</td>
      <td>-3.643403e-02</td>
      <td>-2.587115e-02</td>
      <td>-3.834290e-02</td>
      <td>-5.180294e-02</td>
      <td>-2.096528e-02</td>
      <td>-1.378192e-02</td>
      <td>-1.944748e-02</td>
      <td>-1.908876e-02</td>
      <td>-1.286666e-02</td>
      <td>-1.928171e-02</td>
      <td>-1.135138e-02</td>
      <td>-1.408814e-02</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.830608e-01</td>
      <td>-5.065332e-02</td>
      <td>-4.372962e-02</td>
      <td>1.767239e-02</td>
      <td>-5.834657e-03</td>
      <td>-4.343836e-02</td>
      <td>-3.548744e-02</td>
      <td>1.317923e-02</td>
      <td>-3.656334e-02</td>
      <td>-5.761824e-02</td>
      <td>6.881272e-04</td>
      <td>1.939246e-03</td>
      <td>2.326268e-03</td>
      <td>1.976942e-02</td>
      <td>-1.887146e-02</td>
      <td>-6.746586e-02</td>
      <td>5.523667e-03</td>
      <td>3.651510e-02</td>
      <td>-3.337897e-03</td>
      <td>-1.394194e-02</td>
      <td>1.637635e-03</td>
      <td>9.596522e-03</td>
      <td>2.307559e-02</td>
      <td>1.185474e-02</td>
      <td>6.795144e-03</td>
      <td>2.198073e-03</td>
      <td>5.195697e-03</td>
      <td>6.635339e-04</td>
      <td>2.446121e-03</td>
      <td>-5.063793e-03</td>
      <td>-3.251899e-03</td>
      <td>-1.181548e-03</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.911763e+00</td>
      <td>5.641920e-01</td>
      <td>6.801756e-01</td>
      <td>4.334864e-01</td>
      <td>4.411104e-01</td>
      <td>3.104001e-01</td>
      <td>3.292198e-01</td>
      <td>2.958541e-01</td>
      <td>2.440116e-01</td>
      <td>1.543131e-01</td>
      <td>2.147572e-01</td>
      <td>1.461472e-01</td>
      <td>1.696045e-01</td>
      <td>1.602904e-01</td>
      <td>2.351203e-01</td>
      <td>1.264062e-01</td>
      <td>1.255376e-01</td>
      <td>8.368895e-02</td>
      <td>3.329362e-02</td>
      <td>4.831229e-02</td>
      <td>4.322917e-02</td>
      <td>5.218879e-02</td>
      <td>4.844932e-02</td>
      <td>3.512182e-02</td>
      <td>2.231585e-02</td>
      <td>2.017805e-02</td>
      <td>3.088968e-02</td>
      <td>2.171679e-02</td>
      <td>1.520893e-02</td>
      <td>1.247767e-02</td>
      <td>9.648521e-03</td>
      <td>1.203538e-02</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.520503e+00</td>
      <td>2.093992e+00</td>
      <td>1.432702e+00</td>
      <td>1.671510e+00</td>
      <td>1.636564e+00</td>
      <td>1.513569e+00</td>
      <td>1.847978e+00</td>
      <td>1.314840e+00</td>
      <td>1.484858e+00</td>
      <td>1.494331e+00</td>
      <td>1.473337e+00</td>
      <td>1.263322e+00</td>
      <td>1.254793e+00</td>
      <td>1.325076e+00</td>
      <td>1.295259e+00</td>
      <td>2.054530e+00</td>
      <td>6.756137e-01</td>
      <td>6.905355e-01</td>
      <td>7.766002e-01</td>
      <td>5.728380e-01</td>
      <td>7.332668e-01</td>
      <td>7.153266e-01</td>
      <td>5.857734e-01</td>
      <td>6.154366e-01</td>
      <td>8.313240e-01</td>
      <td>5.785994e-01</td>
      <td>4.419634e-01</td>
      <td>4.617137e-01</td>
      <td>5.600971e-01</td>
      <td>4.760591e-01</td>
      <td>4.765389e-01</td>
      <td>2.465702e-01</td>
    </tr>
  </tbody>
</table>
</div>



The classes are quite well separated in the first two components. The first component is the one that separates the classes the most. The second component separates the classes but not as much as the first one. It's clear that the X axis is the one with higher variance and hence, the most helpful to separate the classes.


    
![png](PCA-vote_files/PCA-vote_12_0.png)
    


As was previously mentioned, the first component is the one with higher variance. The second component has a lower variance than the first one but it is still higher than the third one. The third component has the lowest variance of all of them.

We can see that around the last half of the PCA's there isn't much variance explained. This is due to the fact that the data is categorical (binary) and in our One Hot Encoding, there is a column for Yes and another one for No. This causes that each pair of features are linearly dependent and hence, they don't provide any information, which is represented as a zero variance in the PC space.


    
![png](PCA-vote_files/PCA-vote_14_0.png)
    


If our own PCA implementation is ran, the results are exactly the same. This is because the algorithm is the same and the results are the same. The only difference is that the algorithm is implemented in a different way.

However, in the following scatterplot can be seen that the plot is "rotated". This isn't something to worry about because the results are the same. When Principal Components are calculated, the only constraint is that they are orthogonal and unitary so there's ambiguity in the sign of the components. This is why the plot is rotated.

## 2.2. Sklearn PCA




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.538132</td>
      <td>0.208806</td>
      <td>0.664641</td>
      <td>-0.174588</td>
      <td>-0.450081</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.566310</td>
      <td>0.385575</td>
      <td>-0.730238</td>
      <td>0.070440</td>
      <td>-0.545899</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.181888</td>
      <td>1.863804</td>
      <td>0.193548</td>
      <td>-0.056694</td>
      <td>0.517372</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
      <th>PC6</th>
      <th>PC7</th>
      <th>PC8</th>
      <th>PC9</th>
      <th>PC10</th>
      <th>PC11</th>
      <th>PC12</th>
      <th>PC13</th>
      <th>PC14</th>
      <th>PC15</th>
      <th>PC16</th>
      <th>PC17</th>
      <th>PC18</th>
      <th>PC19</th>
      <th>PC20</th>
      <th>PC21</th>
      <th>PC22</th>
      <th>PC23</th>
      <th>PC24</th>
      <th>PC25</th>
      <th>PC26</th>
      <th>PC27</th>
      <th>PC28</th>
      <th>PC29</th>
      <th>PC30</th>
      <th>PC31</th>
      <th>PC32</th>
    </tr>
    <tr>
      <th>Feature</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>handicapped-infants_n</th>
      <td>0.256762</td>
      <td>-0.093263</td>
      <td>0.088130</td>
      <td>0.325191</td>
      <td>-0.187823</td>
      <td>-0.054035</td>
      <td>-0.010755</td>
      <td>-0.099566</td>
      <td>0.032208</td>
      <td>-0.040712</td>
      <td>-0.049296</td>
      <td>-0.013444</td>
      <td>0.020559</td>
      <td>-0.000532</td>
      <td>-0.016617</td>
      <td>-0.033761</td>
      <td>0.023484</td>
      <td>-0.009327</td>
      <td>-0.001945</td>
      <td>0.009095</td>
      <td>0.011213</td>
      <td>-0.010045</td>
      <td>0.008297</td>
      <td>-0.019264</td>
      <td>-0.034341</td>
      <td>0.028547</td>
      <td>0.007734</td>
      <td>-0.014879</td>
      <td>-0.001525</td>
      <td>-0.039435</td>
      <td>-0.001553</td>
      <td>0.004958</td>
    </tr>
    <tr>
      <th>handicapped-infants_y</th>
      <td>-0.247978</td>
      <td>0.092984</td>
      <td>-0.092867</td>
      <td>-0.325534</td>
      <td>0.188177</td>
      <td>0.059874</td>
      <td>0.033589</td>
      <td>0.101374</td>
      <td>-0.026258</td>
      <td>0.035704</td>
      <td>0.037351</td>
      <td>0.014655</td>
      <td>-0.017975</td>
      <td>0.008312</td>
      <td>-0.019891</td>
      <td>-0.036120</td>
      <td>0.004268</td>
      <td>-0.012493</td>
      <td>0.000053</td>
      <td>0.006020</td>
      <td>0.006528</td>
      <td>-0.015385</td>
      <td>0.008100</td>
      <td>-0.017561</td>
      <td>-0.036774</td>
      <td>0.027704</td>
      <td>0.007737</td>
      <td>-0.015908</td>
      <td>-0.001519</td>
      <td>-0.038979</td>
      <td>-0.001177</td>
      <td>0.005114</td>
    </tr>
    <tr>
      <th>water-project-cost-sharing_n</th>
      <td>-0.060097</td>
      <td>-0.332781</td>
      <td>-0.013825</td>
      <td>0.126901</td>
      <td>0.282068</td>
      <td>-0.025295</td>
      <td>0.036589</td>
      <td>-0.024194</td>
      <td>-0.086435</td>
      <td>-0.005958</td>
      <td>-0.052785</td>
      <td>0.026765</td>
      <td>-0.024300</td>
      <td>-0.006227</td>
      <td>-0.038646</td>
      <td>-0.061662</td>
      <td>0.033674</td>
      <td>-0.124000</td>
      <td>0.001099</td>
      <td>-0.013478</td>
      <td>-0.008015</td>
      <td>0.008825</td>
      <td>-0.017555</td>
      <td>0.013022</td>
      <td>0.009428</td>
      <td>0.001968</td>
      <td>0.001022</td>
      <td>0.002682</td>
      <td>0.000697</td>
      <td>0.002428</td>
      <td>0.001057</td>
      <td>-0.000072</td>
    </tr>
    <tr>
      <th>water-project-cost-sharing_y</th>
      <td>0.067773</td>
      <td>0.349206</td>
      <td>0.040549</td>
      <td>-0.132102</td>
      <td>-0.251269</td>
      <td>0.023667</td>
      <td>-0.047180</td>
      <td>0.008828</td>
      <td>0.078873</td>
      <td>-0.023695</td>
      <td>0.031225</td>
      <td>-0.032808</td>
      <td>0.044424</td>
      <td>0.036774</td>
      <td>-0.055636</td>
      <td>-0.065791</td>
      <td>0.033492</td>
      <td>-0.120655</td>
      <td>-0.001657</td>
      <td>-0.008777</td>
      <td>-0.013344</td>
      <td>0.009017</td>
      <td>-0.018191</td>
      <td>0.014066</td>
      <td>0.010838</td>
      <td>0.001187</td>
      <td>-0.001501</td>
      <td>0.002315</td>
      <td>0.000420</td>
      <td>0.001406</td>
      <td>0.001262</td>
      <td>0.000537</td>
    </tr>
    <tr>
      <th>adoption-of-the-budget-resolution_n</th>
      <td>0.388262</td>
      <td>-0.056683</td>
      <td>-0.093090</td>
      <td>0.014175</td>
      <td>-0.013003</td>
      <td>0.112083</td>
      <td>-0.104561</td>
      <td>-0.017891</td>
      <td>-0.009556</td>
      <td>-0.022129</td>
      <td>0.123548</td>
      <td>0.154440</td>
      <td>0.053762</td>
      <td>-0.026558</td>
      <td>0.023243</td>
      <td>-0.047957</td>
      <td>0.043163</td>
      <td>-0.007515</td>
      <td>0.002510</td>
      <td>-0.002124</td>
      <td>0.015871</td>
      <td>-0.017043</td>
      <td>0.014757</td>
      <td>-0.009129</td>
      <td>-0.036488</td>
      <td>-0.003285</td>
      <td>0.008572</td>
      <td>0.014939</td>
      <td>0.000450</td>
      <td>0.023403</td>
      <td>-0.034884</td>
      <td>0.000907</td>
    </tr>
    <tr>
      <th>adoption-of-the-budget-resolution_y</th>
      <td>-0.391197</td>
      <td>0.050639</td>
      <td>0.092826</td>
      <td>-0.018025</td>
      <td>0.015376</td>
      <td>-0.101254</td>
      <td>0.109810</td>
      <td>0.016419</td>
      <td>0.029665</td>
      <td>0.016330</td>
      <td>-0.132460</td>
      <td>-0.162299</td>
      <td>-0.037400</td>
      <td>0.026046</td>
      <td>-0.066785</td>
      <td>-0.017325</td>
      <td>-0.017591</td>
      <td>-0.004656</td>
      <td>-0.002918</td>
      <td>0.015584</td>
      <td>0.005263</td>
      <td>-0.023484</td>
      <td>0.012141</td>
      <td>-0.009982</td>
      <td>-0.035800</td>
      <td>-0.004538</td>
      <td>0.006567</td>
      <td>0.015016</td>
      <td>0.000727</td>
      <td>0.023915</td>
      <td>-0.033461</td>
      <td>0.000192</td>
    </tr>
    <tr>
      <th>physician-fee-freeze_n</th>
      <td>-0.418859</td>
      <td>0.072861</td>
      <td>0.060585</td>
      <td>0.034733</td>
      <td>0.051962</td>
      <td>-0.059536</td>
      <td>0.096207</td>
      <td>-0.046892</td>
      <td>-0.027004</td>
      <td>0.026827</td>
      <td>0.047287</td>
      <td>0.021786</td>
      <td>0.165800</td>
      <td>-0.005133</td>
      <td>0.012468</td>
      <td>-0.023528</td>
      <td>0.049384</td>
      <td>-0.001479</td>
      <td>-0.027682</td>
      <td>0.051657</td>
      <td>0.005845</td>
      <td>-0.046368</td>
      <td>0.002744</td>
      <td>-0.005681</td>
      <td>-0.020688</td>
      <td>-0.001277</td>
      <td>-0.001812</td>
      <td>0.008827</td>
      <td>-0.001960</td>
      <td>0.018799</td>
      <td>0.035651</td>
      <td>0.003930</td>
    </tr>
    <tr>
      <th>physician-fee-freeze_y</th>
      <td>0.420358</td>
      <td>-0.093019</td>
      <td>-0.067991</td>
      <td>-0.035083</td>
      <td>-0.046999</td>
      <td>0.067987</td>
      <td>-0.089122</td>
      <td>0.042825</td>
      <td>0.038540</td>
      <td>-0.029205</td>
      <td>-0.048473</td>
      <td>-0.021170</td>
      <td>-0.133382</td>
      <td>0.021243</td>
      <td>-0.048275</td>
      <td>-0.044381</td>
      <td>-0.018943</td>
      <td>0.000065</td>
      <td>0.014419</td>
      <td>-0.024733</td>
      <td>0.016648</td>
      <td>0.009137</td>
      <td>0.022953</td>
      <td>-0.020495</td>
      <td>-0.039466</td>
      <td>-0.000161</td>
      <td>0.004796</td>
      <td>0.009210</td>
      <td>0.001003</td>
      <td>0.024531</td>
      <td>0.040310</td>
      <td>0.004026</td>
    </tr>
    <tr>
      <th>el-salvador-aid_n</th>
      <td>-0.438722</td>
      <td>-0.046481</td>
      <td>0.001325</td>
      <td>0.029435</td>
      <td>-0.049771</td>
      <td>-0.035008</td>
      <td>-0.081525</td>
      <td>-0.038750</td>
      <td>-0.034227</td>
      <td>0.043613</td>
      <td>0.073877</td>
      <td>0.019482</td>
      <td>0.005140</td>
      <td>-0.069134</td>
      <td>-0.056070</td>
      <td>-0.050979</td>
      <td>-0.129082</td>
      <td>-0.022231</td>
      <td>0.004475</td>
      <td>-0.003297</td>
      <td>0.001676</td>
      <td>-0.027246</td>
      <td>0.049812</td>
      <td>0.012795</td>
      <td>0.000503</td>
      <td>-0.019542</td>
      <td>-0.033730</td>
      <td>-0.006700</td>
      <td>0.032675</td>
      <td>-0.008298</td>
      <td>0.000409</td>
      <td>-0.001764</td>
    </tr>
    <tr>
      <th>el-salvador-aid_y</th>
      <td>0.445002</td>
      <td>0.031042</td>
      <td>-0.012254</td>
      <td>-0.023394</td>
      <td>0.049251</td>
      <td>0.036673</td>
      <td>0.073160</td>
      <td>0.040531</td>
      <td>0.028021</td>
      <td>-0.056796</td>
      <td>-0.078383</td>
      <td>-0.018857</td>
      <td>-0.007108</td>
      <td>0.062092</td>
      <td>0.013364</td>
      <td>-0.025318</td>
      <td>0.133084</td>
      <td>0.023029</td>
      <td>0.008879</td>
      <td>-0.000606</td>
      <td>0.007695</td>
      <td>-0.018352</td>
      <td>0.049073</td>
      <td>0.016387</td>
      <td>0.011008</td>
      <td>-0.022527</td>
      <td>-0.038724</td>
      <td>-0.005186</td>
      <td>0.032759</td>
      <td>-0.009732</td>
      <td>-0.001485</td>
      <td>-0.002827</td>
    </tr>
    <tr>
      <th>religious-groups-in-schools_n</th>
      <td>-0.326629</td>
      <td>-0.083771</td>
      <td>-0.154849</td>
      <td>-0.059606</td>
      <td>-0.029664</td>
      <td>0.154418</td>
      <td>-0.105107</td>
      <td>-0.097530</td>
      <td>-0.084733</td>
      <td>-0.084417</td>
      <td>-0.071026</td>
      <td>-0.104618</td>
      <td>0.067532</td>
      <td>-0.027301</td>
      <td>0.001095</td>
      <td>-0.030182</td>
      <td>0.012873</td>
      <td>0.005534</td>
      <td>0.007716</td>
      <td>-0.001091</td>
      <td>0.004900</td>
      <td>-0.002819</td>
      <td>0.020179</td>
      <td>-0.006349</td>
      <td>0.006914</td>
      <td>-0.034704</td>
      <td>-0.014338</td>
      <td>0.010804</td>
      <td>-0.047588</td>
      <td>-0.014050</td>
      <td>-0.000731</td>
      <td>0.006669</td>
    </tr>
    <tr>
      <th>religious-groups-in-schools_y</th>
      <td>0.338410</td>
      <td>0.071962</td>
      <td>0.149910</td>
      <td>0.055550</td>
      <td>0.034071</td>
      <td>-0.163114</td>
      <td>0.105312</td>
      <td>0.087148</td>
      <td>0.080148</td>
      <td>0.086541</td>
      <td>0.072594</td>
      <td>0.104703</td>
      <td>-0.071273</td>
      <td>0.021545</td>
      <td>-0.039114</td>
      <td>-0.037306</td>
      <td>-0.008565</td>
      <td>0.000613</td>
      <td>0.002938</td>
      <td>-0.001173</td>
      <td>0.000084</td>
      <td>-0.005743</td>
      <td>0.022566</td>
      <td>-0.002804</td>
      <td>0.005935</td>
      <td>-0.034717</td>
      <td>-0.012554</td>
      <td>0.011378</td>
      <td>-0.046319</td>
      <td>-0.012120</td>
      <td>-0.000509</td>
      <td>0.008121</td>
    </tr>
    <tr>
      <th>anti-satellite-test-ban_n</th>
      <td>0.379034</td>
      <td>0.115427</td>
      <td>0.003968</td>
      <td>0.018448</td>
      <td>0.012183</td>
      <td>0.059542</td>
      <td>0.121799</td>
      <td>-0.127773</td>
      <td>-0.110907</td>
      <td>0.023628</td>
      <td>0.106316</td>
      <td>-0.102036</td>
      <td>-0.078324</td>
      <td>-0.035388</td>
      <td>0.014069</td>
      <td>-0.062037</td>
      <td>0.003478</td>
      <td>0.014575</td>
      <td>-0.010529</td>
      <td>-0.017397</td>
      <td>0.014836</td>
      <td>-0.027636</td>
      <td>0.016107</td>
      <td>0.029102</td>
      <td>0.015619</td>
      <td>0.007492</td>
      <td>0.025583</td>
      <td>-0.039122</td>
      <td>-0.008894</td>
      <td>0.014755</td>
      <td>-0.001188</td>
      <td>0.013479</td>
    </tr>
    <tr>
      <th>anti-satellite-test-ban_y</th>
      <td>-0.383204</td>
      <td>-0.120736</td>
      <td>-0.002134</td>
      <td>-0.024748</td>
      <td>-0.020122</td>
      <td>-0.053566</td>
      <td>-0.117781</td>
      <td>0.116119</td>
      <td>0.102622</td>
      <td>-0.032751</td>
      <td>-0.126630</td>
      <td>0.098583</td>
      <td>0.083048</td>
      <td>0.045331</td>
      <td>-0.056325</td>
      <td>-0.012671</td>
      <td>0.000636</td>
      <td>0.023049</td>
      <td>-0.003179</td>
      <td>-0.021362</td>
      <td>0.017718</td>
      <td>-0.022263</td>
      <td>0.016454</td>
      <td>0.027954</td>
      <td>0.017021</td>
      <td>0.006774</td>
      <td>0.023689</td>
      <td>-0.039903</td>
      <td>-0.010065</td>
      <td>0.012782</td>
      <td>-0.001962</td>
      <td>0.014067</td>
    </tr>
    <tr>
      <th>aid-to-nicaraguan-contras_n</th>
      <td>0.415938</td>
      <td>0.035219</td>
      <td>-0.034406</td>
      <td>0.007198</td>
      <td>0.054026</td>
      <td>0.073077</td>
      <td>0.085984</td>
      <td>-0.057661</td>
      <td>-0.035758</td>
      <td>-0.050972</td>
      <td>-0.024534</td>
      <td>0.024561</td>
      <td>0.083432</td>
      <td>0.145027</td>
      <td>-0.035350</td>
      <td>-0.015994</td>
      <td>-0.088508</td>
      <td>0.017774</td>
      <td>-0.005053</td>
      <td>-0.018436</td>
      <td>-0.015450</td>
      <td>-0.015742</td>
      <td>0.022399</td>
      <td>0.019029</td>
      <td>0.015437</td>
      <td>0.014422</td>
      <td>0.029410</td>
      <td>0.043260</td>
      <td>0.007304</td>
      <td>-0.011973</td>
      <td>0.001600</td>
      <td>0.005063</td>
    </tr>
    <tr>
      <th>aid-to-nicaraguan-contras_y</th>
      <td>-0.432645</td>
      <td>-0.045070</td>
      <td>0.039255</td>
      <td>-0.018180</td>
      <td>-0.046593</td>
      <td>-0.069572</td>
      <td>-0.073391</td>
      <td>0.039556</td>
      <td>0.025169</td>
      <td>0.045495</td>
      <td>0.023388</td>
      <td>-0.028099</td>
      <td>-0.052978</td>
      <td>-0.134658</td>
      <td>-0.018051</td>
      <td>-0.051413</td>
      <td>0.087170</td>
      <td>0.038568</td>
      <td>-0.009260</td>
      <td>-0.023628</td>
      <td>-0.007026</td>
      <td>-0.004042</td>
      <td>0.016708</td>
      <td>0.017637</td>
      <td>0.021359</td>
      <td>0.016101</td>
      <td>0.031927</td>
      <td>0.044732</td>
      <td>0.010152</td>
      <td>-0.011519</td>
      <td>0.003072</td>
      <td>0.005905</td>
    </tr>
    <tr>
      <th>mx-missile_n</th>
      <td>0.406467</td>
      <td>0.090756</td>
      <td>-0.011579</td>
      <td>0.006791</td>
      <td>0.046729</td>
      <td>0.105583</td>
      <td>0.084247</td>
      <td>-0.006033</td>
      <td>0.094181</td>
      <td>0.035692</td>
      <td>-0.115158</td>
      <td>0.026832</td>
      <td>0.047571</td>
      <td>-0.124969</td>
      <td>-0.029990</td>
      <td>-0.031622</td>
      <td>-0.028517</td>
      <td>0.022922</td>
      <td>-0.011113</td>
      <td>-0.057892</td>
      <td>-0.000234</td>
      <td>-0.051677</td>
      <td>-0.060529</td>
      <td>-0.013893</td>
      <td>0.002242</td>
      <td>-0.002146</td>
      <td>-0.022307</td>
      <td>0.003400</td>
      <td>0.002574</td>
      <td>-0.001725</td>
      <td>0.000751</td>
      <td>0.001819</td>
    </tr>
    <tr>
      <th>mx-missile_y</th>
      <td>-0.387329</td>
      <td>-0.107458</td>
      <td>0.004448</td>
      <td>-0.008759</td>
      <td>-0.064276</td>
      <td>-0.109568</td>
      <td>-0.094382</td>
      <td>-0.004678</td>
      <td>-0.104391</td>
      <td>-0.060978</td>
      <td>0.107214</td>
      <td>-0.017107</td>
      <td>-0.074112</td>
      <td>0.143231</td>
      <td>-0.004613</td>
      <td>-0.021892</td>
      <td>0.028091</td>
      <td>0.030503</td>
      <td>-0.015310</td>
      <td>-0.044027</td>
      <td>-0.001305</td>
      <td>-0.054806</td>
      <td>-0.053888</td>
      <td>-0.010331</td>
      <td>0.000621</td>
      <td>-0.001828</td>
      <td>-0.020262</td>
      <td>0.003925</td>
      <td>0.001441</td>
      <td>-0.002143</td>
      <td>-0.001227</td>
      <td>0.000594</td>
    </tr>
    <tr>
      <th>immigration_n</th>
      <td>-0.017249</td>
      <td>0.224573</td>
      <td>-0.395418</td>
      <td>0.138979</td>
      <td>0.056454</td>
      <td>-0.122753</td>
      <td>-0.017805</td>
      <td>0.026215</td>
      <td>0.011740</td>
      <td>-0.012669</td>
      <td>-0.008694</td>
      <td>-0.003714</td>
      <td>-0.005920</td>
      <td>-0.003208</td>
      <td>-0.019028</td>
      <td>-0.035317</td>
      <td>0.001798</td>
      <td>0.005944</td>
      <td>-0.005633</td>
      <td>-0.007631</td>
      <td>0.011308</td>
      <td>-0.015694</td>
      <td>0.009032</td>
      <td>0.000781</td>
      <td>0.004445</td>
      <td>-0.001177</td>
      <td>0.018026</td>
      <td>-0.005651</td>
      <td>-0.010304</td>
      <td>-0.001615</td>
      <td>0.002826</td>
      <td>-0.040412</td>
    </tr>
    <tr>
      <th>immigration_y</th>
      <td>0.022922</td>
      <td>-0.219610</td>
      <td>0.398331</td>
      <td>-0.142262</td>
      <td>-0.058217</td>
      <td>0.118346</td>
      <td>0.025176</td>
      <td>-0.026021</td>
      <td>-0.007223</td>
      <td>0.011677</td>
      <td>0.006537</td>
      <td>0.007679</td>
      <td>0.017867</td>
      <td>0.003866</td>
      <td>-0.018074</td>
      <td>-0.021732</td>
      <td>0.005223</td>
      <td>0.005904</td>
      <td>-0.001256</td>
      <td>-0.012279</td>
      <td>0.009032</td>
      <td>-0.015464</td>
      <td>0.011157</td>
      <td>-0.001322</td>
      <td>0.003356</td>
      <td>-0.002409</td>
      <td>0.016277</td>
      <td>-0.007101</td>
      <td>-0.010288</td>
      <td>-0.002223</td>
      <td>0.002869</td>
      <td>-0.040769</td>
    </tr>
    <tr>
      <th>synfuels-corporation-cutback_n</th>
      <td>0.093685</td>
      <td>-0.300887</td>
      <td>-0.208771</td>
      <td>-0.125813</td>
      <td>-0.171341</td>
      <td>-0.071169</td>
      <td>0.178048</td>
      <td>0.012174</td>
      <td>0.016343</td>
      <td>0.015430</td>
      <td>0.035175</td>
      <td>-0.003647</td>
      <td>0.042839</td>
      <td>-0.001972</td>
      <td>-0.028944</td>
      <td>-0.057901</td>
      <td>-0.000398</td>
      <td>0.003578</td>
      <td>0.027237</td>
      <td>0.015869</td>
      <td>0.005055</td>
      <td>0.006314</td>
      <td>0.004525</td>
      <td>-0.057823</td>
      <td>0.040877</td>
      <td>0.010586</td>
      <td>0.001032</td>
      <td>-0.004192</td>
      <td>0.006012</td>
      <td>0.007120</td>
      <td>-0.003233</td>
      <td>0.002629</td>
    </tr>
    <tr>
      <th>synfuels-corporation-cutback_y</th>
      <td>-0.095445</td>
      <td>0.288090</td>
      <td>0.203185</td>
      <td>0.121956</td>
      <td>0.178397</td>
      <td>0.076070</td>
      <td>-0.160656</td>
      <td>-0.007704</td>
      <td>-0.030520</td>
      <td>-0.014440</td>
      <td>-0.023508</td>
      <td>0.010182</td>
      <td>-0.023616</td>
      <td>0.023033</td>
      <td>-0.024649</td>
      <td>-0.063690</td>
      <td>-0.000899</td>
      <td>0.016395</td>
      <td>0.025368</td>
      <td>0.028024</td>
      <td>0.003382</td>
      <td>-0.001568</td>
      <td>0.006035</td>
      <td>-0.059089</td>
      <td>0.040882</td>
      <td>0.011921</td>
      <td>0.001976</td>
      <td>-0.003147</td>
      <td>0.004957</td>
      <td>0.006539</td>
      <td>-0.002753</td>
      <td>0.002509</td>
    </tr>
    <tr>
      <th>education-spending_n</th>
      <td>-0.373715</td>
      <td>0.073263</td>
      <td>0.027069</td>
      <td>-0.036426</td>
      <td>-0.024719</td>
      <td>-0.010622</td>
      <td>0.118973</td>
      <td>-0.066559</td>
      <td>-0.002180</td>
      <td>-0.222494</td>
      <td>-0.022177</td>
      <td>0.109352</td>
      <td>-0.067842</td>
      <td>-0.039822</td>
      <td>-0.017852</td>
      <td>-0.065541</td>
      <td>-0.010664</td>
      <td>0.035831</td>
      <td>0.073627</td>
      <td>0.033822</td>
      <td>-0.021472</td>
      <td>0.007355</td>
      <td>-0.028174</td>
      <td>0.026557</td>
      <td>-0.011839</td>
      <td>-0.003797</td>
      <td>0.002251</td>
      <td>-0.000656</td>
      <td>0.000638</td>
      <td>0.000085</td>
      <td>0.001894</td>
      <td>-0.001837</td>
    </tr>
    <tr>
      <th>education-spending_y</th>
      <td>0.375356</td>
      <td>-0.089930</td>
      <td>-0.021131</td>
      <td>0.027077</td>
      <td>0.022592</td>
      <td>-0.010871</td>
      <td>-0.108923</td>
      <td>0.048085</td>
      <td>-0.014179</td>
      <td>0.200055</td>
      <td>0.018576</td>
      <td>-0.091113</td>
      <td>0.080769</td>
      <td>0.037020</td>
      <td>-0.021291</td>
      <td>-0.071253</td>
      <td>0.005617</td>
      <td>0.041412</td>
      <td>0.080404</td>
      <td>0.025477</td>
      <td>-0.029411</td>
      <td>0.005480</td>
      <td>-0.032394</td>
      <td>0.029158</td>
      <td>-0.013356</td>
      <td>-0.003342</td>
      <td>-0.001064</td>
      <td>-0.003475</td>
      <td>0.000587</td>
      <td>-0.001498</td>
      <td>0.000711</td>
      <td>-0.002038</td>
    </tr>
    <tr>
      <th>superfund-right-to-sue_n</th>
      <td>-0.362164</td>
      <td>-0.072057</td>
      <td>-0.056387</td>
      <td>0.045284</td>
      <td>0.091409</td>
      <td>0.101003</td>
      <td>0.018735</td>
      <td>-0.119189</td>
      <td>0.227823</td>
      <td>0.012334</td>
      <td>0.078716</td>
      <td>-0.021799</td>
      <td>-0.008534</td>
      <td>0.046611</td>
      <td>-0.034616</td>
      <td>-0.047515</td>
      <td>0.014461</td>
      <td>0.031034</td>
      <td>-0.049878</td>
      <td>-0.006223</td>
      <td>-0.061523</td>
      <td>0.030782</td>
      <td>-0.005318</td>
      <td>-0.012351</td>
      <td>-0.008822</td>
      <td>-0.028424</td>
      <td>0.014066</td>
      <td>-0.014293</td>
      <td>0.009959</td>
      <td>-0.002507</td>
      <td>-0.000674</td>
      <td>0.000642</td>
    </tr>
    <tr>
      <th>superfund-right-to-sue_y</th>
      <td>0.363283</td>
      <td>0.060661</td>
      <td>0.050331</td>
      <td>-0.052611</td>
      <td>-0.071525</td>
      <td>-0.104751</td>
      <td>-0.017121</td>
      <td>0.086367</td>
      <td>-0.245938</td>
      <td>-0.024860</td>
      <td>-0.087418</td>
      <td>0.042307</td>
      <td>0.037571</td>
      <td>-0.035366</td>
      <td>-0.017294</td>
      <td>-0.050476</td>
      <td>-0.001276</td>
      <td>0.024064</td>
      <td>-0.054498</td>
      <td>-0.004000</td>
      <td>-0.057820</td>
      <td>0.027827</td>
      <td>-0.002969</td>
      <td>-0.015495</td>
      <td>-0.012156</td>
      <td>-0.027872</td>
      <td>0.016006</td>
      <td>-0.011944</td>
      <td>0.009677</td>
      <td>-0.001081</td>
      <td>-0.000511</td>
      <td>0.000938</td>
    </tr>
    <tr>
      <th>crime_n</th>
      <td>-0.374738</td>
      <td>0.055446</td>
      <td>-0.064892</td>
      <td>0.005342</td>
      <td>-0.088904</td>
      <td>0.081263</td>
      <td>0.043933</td>
      <td>-0.104506</td>
      <td>-0.071709</td>
      <td>0.157854</td>
      <td>-0.092038</td>
      <td>0.106088</td>
      <td>-0.052366</td>
      <td>0.041880</td>
      <td>-0.022186</td>
      <td>-0.037518</td>
      <td>0.020759</td>
      <td>0.026234</td>
      <td>-0.026267</td>
      <td>-0.003103</td>
      <td>-0.023414</td>
      <td>0.025213</td>
      <td>0.015432</td>
      <td>0.013151</td>
      <td>-0.000710</td>
      <td>0.044422</td>
      <td>-0.039374</td>
      <td>0.002615</td>
      <td>-0.015492</td>
      <td>0.011273</td>
      <td>-0.002691</td>
      <td>-0.004784</td>
    </tr>
    <tr>
      <th>crime_y</th>
      <td>0.376650</td>
      <td>-0.068617</td>
      <td>0.066206</td>
      <td>-0.014497</td>
      <td>0.105187</td>
      <td>-0.084784</td>
      <td>-0.038083</td>
      <td>0.088254</td>
      <td>0.055828</td>
      <td>-0.171152</td>
      <td>0.094984</td>
      <td>-0.095963</td>
      <td>0.069588</td>
      <td>-0.042452</td>
      <td>-0.031377</td>
      <td>-0.038787</td>
      <td>-0.021192</td>
      <td>0.027606</td>
      <td>-0.032302</td>
      <td>-0.001171</td>
      <td>-0.020814</td>
      <td>0.023750</td>
      <td>0.011150</td>
      <td>0.012412</td>
      <td>-0.000321</td>
      <td>0.043541</td>
      <td>-0.037751</td>
      <td>0.000267</td>
      <td>-0.015795</td>
      <td>0.009613</td>
      <td>-0.002453</td>
      <td>-0.003848</td>
    </tr>
    <tr>
      <th>duty-free-exports_n</th>
      <td>0.321917</td>
      <td>0.004188</td>
      <td>0.019896</td>
      <td>-0.190412</td>
      <td>0.071452</td>
      <td>-0.175340</td>
      <td>-0.085835</td>
      <td>-0.221078</td>
      <td>0.030761</td>
      <td>0.018921</td>
      <td>-0.042192</td>
      <td>0.021662</td>
      <td>0.012514</td>
      <td>0.008349</td>
      <td>-0.028009</td>
      <td>-0.045136</td>
      <td>-0.005132</td>
      <td>0.024464</td>
      <td>-0.031006</td>
      <td>0.017676</td>
      <td>0.078340</td>
      <td>0.037305</td>
      <td>-0.024636</td>
      <td>0.008922</td>
      <td>0.001317</td>
      <td>-0.008914</td>
      <td>-0.002288</td>
      <td>0.004453</td>
      <td>0.007879</td>
      <td>-0.004890</td>
      <td>-0.001275</td>
      <td>0.000402</td>
    </tr>
    <tr>
      <th>duty-free-exports_y</th>
      <td>-0.318925</td>
      <td>-0.002283</td>
      <td>-0.004013</td>
      <td>0.169859</td>
      <td>-0.054308</td>
      <td>0.187120</td>
      <td>0.098269</td>
      <td>0.203656</td>
      <td>-0.050213</td>
      <td>-0.001611</td>
      <td>0.043843</td>
      <td>-0.018059</td>
      <td>0.007269</td>
      <td>0.020941</td>
      <td>-0.041366</td>
      <td>-0.060981</td>
      <td>-0.013995</td>
      <td>0.023923</td>
      <td>-0.034877</td>
      <td>0.017158</td>
      <td>0.071372</td>
      <td>0.039303</td>
      <td>-0.028966</td>
      <td>0.014087</td>
      <td>0.000623</td>
      <td>-0.011370</td>
      <td>-0.002248</td>
      <td>0.004849</td>
      <td>0.006253</td>
      <td>-0.004801</td>
      <td>-0.001155</td>
      <td>0.000410</td>
    </tr>
    <tr>
      <th>export-administration-act-south-africa_n</th>
      <td>0.165725</td>
      <td>0.016549</td>
      <td>-0.039019</td>
      <td>0.013057</td>
      <td>0.011381</td>
      <td>0.043860</td>
      <td>-0.002260</td>
      <td>-0.048375</td>
      <td>-0.071152</td>
      <td>-0.006550</td>
      <td>0.064817</td>
      <td>0.013979</td>
      <td>0.004131</td>
      <td>-0.026775</td>
      <td>-0.243744</td>
      <td>0.126768</td>
      <td>0.049899</td>
      <td>0.014181</td>
      <td>0.008726</td>
      <td>0.011867</td>
      <td>-0.002065</td>
      <td>-0.002982</td>
      <td>-0.003686</td>
      <td>-0.000462</td>
      <td>0.001266</td>
      <td>-0.001643</td>
      <td>0.000027</td>
      <td>-0.002943</td>
      <td>0.000358</td>
      <td>-0.000595</td>
      <td>0.000254</td>
      <td>0.000253</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](PCA-vote_files/PCA-vote_19_0.png)
    



    
![png](PCA-vote_files/PCA-vote_20_0.png)
    


# 3. Compare with IncrementalPCA

The PCA object proves to be beneficial but exhibits limitations when dealing with large datasets. Its primary drawback is its exclusive support for batch processing, necessitating that all data fit into main memory. In contrast, the IncrementalPCA object offers an alternative processing approach, enabling partial computations that closely align with PCA results while handling data in a minibatch manner. This facilitates the implementation of out-of-core Principal Component Analysis through two methods:

1. Utilizing the `partial_fit` method on sequentially fetched data chunks from the local hard drive or a network database.
2. Invoking the `fit` method on a sparse matrix or a memory-mapped file using `numpy.memmap`.

Notably, IncrementalPCA stores estimates of component and noise variances, updating `explained_variance_ratio_` incrementally. Consequently, memory usage is contingent on the number of samples per batch rather than the overall dataset size.




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.538132</td>
      <td>0.208806</td>
      <td>0.664641</td>
      <td>0.174588</td>
      <td>-0.450081</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.566310</td>
      <td>0.385575</td>
      <td>-0.730238</td>
      <td>-0.070440</td>
      <td>-0.545899</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.181888</td>
      <td>1.863804</td>
      <td>0.193548</td>
      <td>0.056694</td>
      <td>0.517372</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](PCA-vote_files/PCA-vote_24_0.png)
    



    
![png](PCA-vote_files/PCA-vote_25_0.png)
    


The main difference is performance in time and memory. The IncrementalPCA is much faster for big datasets than the PCA and it uses less memory. However, the results are not exactly the same. The explained variance ratio is the same but the components are not. This is because the IncrementalPCA is an approximation of the PCA. The approximation is good but it is not exact.

What can be observed from the following comparison is that, when batch size is increased, IPCA is faster and uses less memory. However, in this case, approximation error from IPCA is negligible. This is observed from the difference in two first explained variance ratio and the average difference in components. The approximation error is not noticeable.

In the other hand, it doesn't either speed up the process. This is because the dataset is not big enough to notice the difference. However, if the dataset was bigger, the difference would be noticeable.


    
![png](PCA-vote_files/PCA-vote_27_0.png)
    


For a `batch_size` equal to the number of samples, the results are the same as the PCA. This is because the algorithm is the same. The only difference is that the algorithm is implemented in a different way. The visible difference in the plots can be due to external factors such as the random initialization of the algorithm.

Our implementation of PCA is faster than the incremental one until `batch_size` of 260. However, when `batch_size` is increased, the IPCA is faster as it is expected given that sklearn PCA is faster than ours.


    
![png](PCA-vote_files/PCA-vote_29_0.png)
    


# 4. Use PCA with k-Means and BIRCH to compare performances

As it was mentioned before, after projecting the data into the 2 first Principal Components, the classes are quite well separated. A lot of variables weren't providing almost any information in terms of variance. This causes the well known dimensionality curse, in which when the number of features is increased, the performance of the algorithm decreases. This is why PCA is used, to reduce the number of features and hence, the dimensionality of the data.

# Cluster the transformed Data using BIRCH


    
![png](PCA-vote_files/PCA-vote_36_0.png)
    


As can be infered from the previous plot, in every case, incrementing the number of components even adding more information, leads to equal or worse results. This is because of the dimensionality curse. The more features, the worse the performance of the algorithm.

In the case of silhouette score, a model trained only with the first component has far better silhouette score than in the case of the model trained without applying PCA.

In the other hand, in terms of V-Measure, KMeans increases its performance when trained with just one component. However, Birch's performance slightly decreases. As it is an informed metric, it is more sensitive to the loss of information.

Another interesting fact is that both algorithms have exactly the same performance when trained with the first component.

    Evaluation results on BIRCH using the original dataset
    Silhouette 0.26 - V-Measure 0.64
    --------------------------------------------------
    Evaluation results on KMeans using the original dataset
    Silhouette 0.35 - V-Measure 0.54
    --------------------------------------------------
    Evaluation results on Birch using the PC1
    Silhouette 0.74 - V-Measure 0.54
    --------------------------------------------------
    Evaluation results on KMeans using the PC1
    Silhouette 0.74 - V-Measure 0.54
    




    <matplotlib.legend.Legend at 0x7f3c98144a10>




    
![png](PCA-vote_files/PCA-vote_39_1.png)
    


Performance in time improves as well. This difference would be more noticeable if the dataset was bigger. From here, another advantage of dimensionality reduction method can be inferred. It is not only that the performance of the algorithm increases but also the time needed to train the algorithm decreases.

# 5. Cluster the transformed Data (SVD) using K-Means and Birch

Use sklearn.decomposition.truncatedSVD to reduce the dimensionality of your data sets
and cluster it with your own k-Means, the one that you implemented in Work 1, and with the
BIRCH from sklearn library. Compare your new results with the ones obtained previously. 

## Non Centered Data




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SV1</th>
      <th>SV2</th>
      <th>SV3</th>
      <th>SV4</th>
      <th>SV5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706905</td>
      <td>-2.462044</td>
      <td>0.216076</td>
      <td>0.663765</td>
      <td>-0.177440</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.929801</td>
      <td>-2.476804</td>
      <td>0.409878</td>
      <td>-0.735057</td>
      <td>0.086129</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.464877</td>
      <td>-1.118496</td>
      <td>1.868968</td>
      <td>0.186572</td>
      <td>-0.044793</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](PCA-vote_files/PCA-vote_45_0.png)
    



    
![png](PCA-vote_files/PCA-vote_46_0.png)
    



    
![png](PCA-vote_files/PCA-vote_48_0.png)
    





    <matplotlib.legend.Legend at 0x7f3c6b566b50>




    
![png](PCA-vote_files/PCA-vote_49_1.png)
    


Behavior is strange. Singular Values are usually sorted from higher to lower magnitude. IN this case, the first one is lower than the second one and, after that, the behavior is the expected one. In Figure Scatter, there are a couple of points which are clearly separated from the rest of the distribution. These outliers are the ones that are causing this behavior.

If data gets scaled and centered, the behavior is the expected one. The first singular value is the highest one and the rest are sorted from higher to lower magnitude.

Indeed, the same behavior than PCA is observed.

## Centered Version




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SV1</th>
      <th>SV2</th>
      <th>SV3</th>
      <th>SV4</th>
      <th>SV5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.089195</td>
      <td>0.481182</td>
      <td>1.282707</td>
      <td>-0.481998</td>
      <td>-0.939985</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.150858</td>
      <td>0.597434</td>
      <td>-1.472942</td>
      <td>0.084576</td>
      <td>-1.127143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.506092</td>
      <td>3.859816</td>
      <td>0.030941</td>
      <td>-0.091015</td>
      <td>1.005174</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](PCA-vote_files/PCA-vote_53_0.png)
    



    
![png](PCA-vote_files/PCA-vote_54_0.png)
    



    
![png](PCA-vote_files/PCA-vote_55_0.png)
    





    <matplotlib.legend.Legend at 0x7f3ce43e01d0>




    
![png](PCA-vote_files/PCA-vote_56_1.png)
    


The results are exactly the same than for the case of PCA. 

    Evaluation results on BIRCH using the transformed dataset
    




    (0.7343127013809588, 0.5368847869421306)



    Evaluation results on KMeans using the transformed dataset
    




    (0.7342234950068158, 0.5256846352651141)




    
![png](PCA-vote_files/PCA-vote_60_0.png)
    




# 6. Visualize in low-dimensional space

Visualize in low-dimensional space. You need to visualize your original data sets, the result of
the k-Means and BIRCH algorithms without the dimensionality reduction, and the result of the
k-Means and BIRCH algorithms with the dimensionality reduction. To visualize in a lowdimensional space (2D or 3D) you will use: PCA and ISOMAP. You will find useful information
of how to deal with this algorithm at:

Given that the data is categorical, the visualization of its 2 first components is not very useful. Until now, 2 first linear projections have been used in order to visualize the data, such as SVD and PCA. The result is that they both provide similar information.

In this case, Self Organized Maps are used in order to have a nonlinear point of view, i.e. to use a different approach.


    
![png](PCA-vote_files/PCA-vote_65_0.png)
    



    
![png](PCA-vote_files/PCA-vote_66_0.png)
    


In this case, both algorithms provide similar information. This can be due to the fact that the data is not very complex and hence, the linear projection is enough to visualize the data.
