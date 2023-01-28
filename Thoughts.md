# Pseudocode

## Important notes 

- **Training data** should be stored in variables named `patterns` and `targets`.
- The code should work for arbitrary sizes of input/output patterns and number of training patterns.
- Be careful about dimensions and order (matrices)

### Questions

## Code needed for the problems
### 3.1.1 Generate data
Variables:
1. Number of samples (for each class)
2. Mean values
3. Standard deviation

    Generate data from normal distributions.

### 3.1.2 Implement and apply perceptron learning rules
#### 1 Apply and compare perceptron learning with delta learning in sequential mode
1. Perceptron, online
    - The perceptron rule in sequential mode can be written as $\Delta W=\eta(\overline{t}-f_{step}(W^T\overline{x}))\overline{x}$. 
2. Delta, online
    - The delta rule in sequential mode can be written as $\Delta W=\eta(\overline{t}-W^T\overline{x})\overline{x}$.
3. Comparison
    - Number of misclassified examples
    - MSE

#### 2 Compare sequential with batch learning, delta. 
1. Delta, online 
    - See 3.1.2 1
2. Delta, batch
    - The delta rule in batch mode can be written as $\Delta W=\eta(T-W^TX)X$. 
3. Comparison 
    - When do the algorithms converge
    - Number of misclassified examples
    - MSE
4. Graphics

#### 3 Remove the bias, train the network with delta batch.
1. Delta, batch
    - See 3.1.2 2
2. Test convergence

### 3.1.3 Classifaction of samples that are not linearly separable
Generate new data in the following way:
1. Number of samples (for each class), $n$
2. Mean values, $\mu_A$ and $\mu_B$
3. Standard deviations, $\sigma_A$ and $\sigma_B$
4. Perceptron online and delta online/batch
5. Comparison
    - Number of misclassified examples
    - MSE
6. Graphics
7. Remove samples
    - Random $25$% from each class
    - Random $50$% from class $A$
    - Random $50$% from class $B$
    - Random $20$% and $80$% from the subset in class $A$ for which $x_1<0$ and $x_1>0$, respectively. 

The samples in $A$ are generated in the following way:

  > $\frac{n}{2}$ pairs $(x_1,x_2)$ such that $x_1\in N(\mu_A, \sigma_A)$ and $\frac{n}{2}$ pairs $(x_1,x_2)$ such that $x_1\in N(-\mu_A, \sigma_A)$.
  > In either case $x_2\in N(\mu_A, \sigma_A)$.

$B$ is created in the same way as 3.1.1 .

## For code

### What is needed
- Initial variables
    1. Number of samples (for each class)
    2. Mean values (for both classes)
    3. Standard deviation (for both classes)
    4. Number of epochs. $20$ epochs
    5. Learning rate. A suitable learning rate is $0.001$
    6. Initial weights. Weights are chosen from $N(0,\sigma)$, the numbers should be small.
    7. Bias 
    8. Permutation(s)
- Functions
    - Create initial weight matrix. To include the bias, add an extra input with value one and a corresponding weight with value $(-?)\theta$
    - Learning rules. Binary labels (0/1) for perceptron rule. Bipolar labels (-1/1) for delta rule
    - Errors
    - Remove samples
- Graphics
    - Subfigures paired
    - Plots of samples
        * Descision boundary. The decision boundary can be derived from $Wx=0$.
    - Plots of error
        * Number of misclassified examples
        * MSE
    
### Pseudocode functions     
Initial weight matrix

    ```
    Input: classes matrices, n, bias
        (Dim: Sample=1xk, X=(k+1)xn, T=1xn, W=1x(k+1))
        weights + array k bias 
        matrix(weights)
        transpose(matrix)
    Output: matrix representing the classes with bias
    ```
 
Learning rules

    ```
    Input: permutation, matrix representing the classes X, W weight matrix, T activation matrix, learning rate eta, array with three elements 
           (options for learning rules)
    
        loop through permutation:
            Calculate by matrix formula for delta online
            Calculate by matrix formula for perceptron online
       
        Calculate by matrix formula for delta batch
    
    Output: weights for chosen learning rules after one epoch
    ```    
  
Convergence

    ```
    Input: W weight matrix, matrix representing the classes X, array with previous results, array with three elements 
           (options for learning rules)
        compute how many errors there are in learning rules by checking the corresponding equations.
    Output: updated array with added errors from W
    ```
    
Remove samples 1-3

    ```
    Input: int percent A, int percent B
        remove requested percent from each
    Output: New samples
    ```
    
Remove samples 4

    ```
    Input: weights
        Remove samples by rule
    Output: New samples
    ```
