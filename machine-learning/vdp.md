# Variance Decomposition Proportions

This post outlines the idea of collinearity in a dataset (including a formal-ish definition), possible issues arising with collinearity and ways of diagnosing the existence and degree of collinearity. Furthermore, this outlines a few issues with historically conventional ways of assessing collinearity and explains (in layman terms) the idea of the variance-decomposition proportions for singular values.

*Disclaimer: this post is more self-serving than anything, because I have the brain of an adolescent chimpanzee at best and a ice block in stone tablet clothing at worst, things are broken down in the simplest form so that 6-months in the future I can reference this easily without having to do a math degree*

## Collinearity

Collinearity has eluded definition, and has been colloquially referred to as as _multicollinearity and ill-conditioning_. For a given set of numerical data in a dataset (referred to as the matrix/data-matrix from here on), we can say k of the variates (so k of the **columns** in this dataset) are collinear if the vectors that represent them lie in a subspace of dimension less than $k$; i.e. if one of the columns is a linear combination of one or more of the others.

[Johnston et. al.](https://link.springer.com/article/10.1007/s11135-017-0584-6) give qualitative descriptions/conditions-to-be-met for collinearity:
- When the variables concerned are control variables in a regression model, whose coefficients are not to be interpreted, but the variables of interest do not display collinearity, either among themselves or with the control variables 
- One or more of the variables is a power of another variable included in the regression for example, some regressions include both age and age2 as variables, and these are almost certain to be collinear
- The variables concerned are dummy variables representing variables with three or more
categories.

*Note that while I agree with what they are attempting to describe, their focus on the model as opposed to the data is less than appealing*

In real-world scenarios, this perfect collinearity rarely occurs, so a broader notion is needed (see  [Belsley et. al](https://www.amazon.com/Regression-Diagnostics-Identifying-Influential-Collinearity/dp/0471691178)). A more expansive description of collinearity is when the angle between the vectors reperented by two (or more) of the variates (columns) is relatively small. The idea is that we have unobserved (latent) variables that may be directly related to two or more of our variates, and may be due to either the nature of the data (e.g. the wingspan and length of an aircraft are collinear with weight most of the times) OR the number of data is low. This second type of collinearity occurs because not all combinations of environmental conditions exist in the study area or when very many variables are involved (see [Dormann et. al.](https://www.biom.uni-freiburg.de/Dateien/PDF/dormann2012ecography_proofprints.pdf))


At this point it is of note that collinearity is inherently a **data** problem, NOT A MODEL PROBLEM. The most obvious issue with collinearity is that having one variate which is essentially a scaled version of another provides no new information, leading to a difficulty in separating the influence of collinear explanatory variables on a given target. Now that we've established collinearity as a data problem, we can further extend the issue to instability of the linear least-squares predictors. When we have collinear data, the net effect is of predictor instability, which can lead to spurious coefficients and little to no explainability.

See [Dormann et. al.](https://www.biom.uni-freiburg.de/Dateien/PDF/dormann2012ecography_proofprints.pdf) for more references.


### Historical Ways to Investigate Collinearity
Historically, there have been several ways of addressing collinearity:
1. Hypothesis testing
This uses typically a $t$ test for explanatory variables, and is often cited as evidence of collinearity when the test statistic is low. However, according to [Belsley et. al.], this hypothesis test is neither necessary nor sufficient for detecting collinearity, and further does not assess the degree of collinearity present
2. Correlation Matrix
The correlation matrix of a data matrix is typically taken as $\bold{X^TX}$, if the data matrix is centered and scaled to unit length. The absence of high correlations cannot be viewed as evidence of no problem (for example three or more variates collinear with no two variates alone highly correlated, which the correlation matrix is incapable of diagnosing). Additionally the correlation matrix cannot diagnose several coexisting near dependencies (e.g. if we have high correlations between a and b, and c and d, we cannot tell whether there's some underlying common dependency).
3. The determinant of the correlation matrix 
[Farrar and Glauber](https://www.scirp.org/(S(lz5mqp453edsnp55rrgjct55))/reference/referencespapers.aspx?referenceid=2468723) assumed that an $n\times p$ matrix with orthogonal columns, the data $\bold{X}$ is a sample from a $p$-variate Gaussian distribution. Their proposed measure is the partial correlation between $\bold{X}_i$ and $\bold{X}_j$, adjusted for all other variates. According to Belsley et. al, using the determinant of the correlation matrix (which is essentially what Farrar and Glauber use) frequently indicates collinearity when no problem exists. The final criticism comes from [Kumar](https://ideas.repec.org/a/tpr/restat/v57y1975i3p365-66.html), where the assumption that our data matrix can be modeled as a stochastic, i.i.d. matrix is hardly ever satisfied.
4. VIF 
The weakness of VIF (like using $\bold{R}$) comes down to its inability to distinguish among several coexisting near dependencies, and a lack of clear definition of what a high or unacceptable VIF is. Furthermore, if using VIF, the degree of collinearity indicated by the VIF is not agreed upon, and the "best-practices" thresholds have not converged over the years
*For reference, the diagonal elements of $\bold{R}^{-1}$ are called the variance inflation factors, where $\bold{R}^{-1}=(\bold{X}^T\bold{X})^{-1}$.*

5. Eigenvalues and Eigenvectors (or principal components) of the correlation matrix 
The idea to use "small" eigenvalues as an indicator of collinearity is not misfounded. The main issue with this approach is that of finding exactly what "small".




A general matrix may not have eigenvalues/vectors (since it may not be symmetric, orthogonal). We can discover analagous features, called _singular values_ and _singular vectors_.

## Singular Value Decomposition

Getting some jargon out of the way:

- Orthogonal matrix: one whose transpose is equal to its inverse
- Transpose of a matrix: changing the rows into columns


```python
import cupy as cp
from cupy import linalg

# orthogonal
A = cp.array(
    [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ]
)

linalg.inv(A) == A.T  # see!
```




    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]])




```python
# transpose
B = cp.array(
    [
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],
    ]
)
B.T  # rows are now columns!

```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




% from MML
> The singular value decomposition (SVD) of a matrix is a central matrix decomposition method in linear algebra. It has been referred to as the “fundamental theorem of linear algebra” (Strang, 1993) because it can be applied to all matrices, not only to square matrices, and it always exists.

Let $A^{m×n}$ be a rectangular matrix of rank
$r \in [0, min(m, n)]$. The SVD of A is a decomposition of the form:

$$


SVD(A) = U\Sigma V^T
$$

where:

$$
U \in R^{m\times m}  \\
V \in R^{n\times n} \\
\Sigma \in R^{m\times n}
$$

The diagonal entries $\sigma_i, i=1,...,r$ of $\Sigma$ are known as the **singular values** of $A$ and is unique.

In layman's terms: the singular values in $\Sigma$ allow the space of rows to be mapped to the space of columns. SVD essentially "factors" out the main axes of interactions by transforming the space of rows to the space of columns.



## Condition Indices
*This diagnostic was initially proposed by [Belsley, Kuh and Welsch (1980)](https://www.amazon.com/Regression-Diagnostics-Identifying-Influential-Collinearity/dp/0471691178)*


If we assume that our data $\bold{X}$ has exact linear dependencies, the $\text{rank}(\bold{X})<p$. The singular-value-decomposition can be thought of as a way to decompose an arbitrary matrix into left and right singular vectors and a set of _singular values_, which essentially is used to transform the 'row space' into 'column space'.
$$
X \equiv \bold{UDV}^T
$$

[Belsely et. al.] propose a diagnostic called the *condition number*, which is the ratio of the largest singular-value (diagonals of the vector $D$ above) to  every other singular value. The matrix is ill-conditioned when the index is larger (the singular value is small). The authors further go on to show that the condition indices provide an upper bound on the sensitivity of the diagonal elements of the data matrix with respect to every other matrix.
$$
\eta_k = \frac{\mu_{max}}{\mu_k}
$$

The takeaway from this is that there are as many near dependencies amongst the columns as there are high condition indices.

## Variance Decomposition
The estimated variance of regression coefficients are now used to indicate the degree to which a given singular value may contribute to more than one variate (and hence indiate collinearity). The variance-covariance of the LS estimator $\bold{b}=(\bold{X^TX})^{-1}\bold{X^Ty}$ is $\sigma^2(\bold{X^TX})^{-1}$. Using the SVD, the variance-covariance matrix of $\bold{b}$ can be written as:
$$
\bold{V(b)}=\sigma^2(\bold{X^TX})^{-1}=\sigma^2 \bold{VD^{-2}V^T}
$$
 for the $k^{th}$ component of $\bold{b}$:
$$
var(b_k) = \sigma^2\sum_{j}\frac{v^2_{kj}}{\mu^2_j}
$$
The idea is that the above decomposes the variance of $b$ into a sum of components, each associate with ONLY one of the $p$ singular vectors. Small $\mu_j$ will be large relative to other components if there's a near-dependency. Therefore, an unusually high *proportion* of the variance of more than one coefficient with the same singular value may indicate that the given dependency is causing issues. These can be calculated as:
$$
\phi_{kj}\equiv \frac{v^2_{kj}}{\mu_j^2}
$$
and the variance-decomposition proportions are:
$$
\pi_{jk}\equiv \frac{\phi_{kj}}{\phi_k}
$$



## How to Actually Use This

Okay, that's as far as I want to take the math, let's figure out how to actually use this. The approach suggested was to determine, for each singular value, the proportion of variation in a given column that is as a consequence of one of each of every singular values. Each column has the proportoinal contribution of each singular value to the given variable. To find potential collineariy, a single _single value_ may contribute highly to more than one variable (column).

The most useful tool is what *Belsley et. al.* refer to as the $\Pi$ table:
| Singular Value | $var(b_1)$ | $var(b_2)$ | ... |
| --- | --- | --- | --- | 
| $\mu_1$ | | |  
| $\mu_2$ | | |  
| $\mu_3$ | | |  
| $\mu_4$ | | |  

The idea above is to:
1. Singular Value with a high condition index: this indicates the number of near dependendencies amongst data columns, and magnitude of high condition indices provide a measure of "tightness"
2. High var-decomp proportions for two or more estimated regression coefficients: each high variance-decomposition proportion identifies those variates that are involved in the corresponding near dependency, and the magnitude of these proportions in conjunction with the high condition index provides a measure of the degree to which the corresponding regression estimate has been degraded by the presence of collinearity

Looking for which singular values (row) contributes to more than one column


## A Worked Example!

This section follows the example using the Consumption function in section 3.4 of [](), and is essentially reverse-engineered python code from [brian-lau/colldiag](https://github.com/brian-lau/colldiag)


```python
from cudf import read_csv, DataFrame, Series
import cupy as cp

df  = read_csv('consumption.txt', sep='\t')
X = df[['c', 'dpi', 'r', 'd_dpi']].values

# rearrange columns as in book
X = cp.vstack([X[:-1, 0], X[1:, 1], X[1:, 2], X[1:, 3]]).T
labels = ['C', 'DPI', 'R', 'dDPI']

# in order to replicate the book's results, we need to add a column of 1's as an intercept term
X = cp.hstack([cp.ones((X.shape[0], 1)), X])
labels.insert(0, 'int')
n, p = X.shape

if p != len(labels):
    raise ValueError("Labels don't match design matrix.")

# Normalize each column to unit length (pg 183 in Belsley et al)
_len = cp.sqrt(cp.sum(X ** 2, axis=0)) 
X = X/_len

U, S, V = cp.linalg.svd(X,full_matrices=False)



lambda_ = S  # already diagonal values alone
condind = S[0] / lambda_ # all SV's by the largest
phi_mat = (V.T * V.T) / (lambda_ ** 2) # square the V and scale by the largets lambda**2, essentially phi_kj from above
phi = cp.sum(phi_mat, axis=1).reshape(-1, 1) # expects COLUMN
vdp = cp.divide(phi_mat,phi).T # final division

# for printing prett-ily
vdp_df = DataFrame(data=vdp, columns=labels)
vdp_df = vdp_df.assign(condind=condind)
vdp_df.insert(0, 'sv', range(1, 1+len(vdp_df)))
vdp_df = vdp_df.set_index('sv')

# need to find rows where condition indices are high
# and multiple row values are high
collinear = []
for row in vdp_df.index.values_host:
    # filter for "high" condind
    s = vdp_df.loc[row][labels]
    if vdp_df.loc[row, 'condind'] > 30 and len(s[s > 0.5]) > 2:
        collinear_tuple = tuple(s[s > 0.5].index.values_host)
        collinear.append(collinear_tuple)

# which replicates the table in 3.4 of Belsley, Kuh and Welsch!
vdp_df
```




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
      <th>int</th>
      <th>C</th>
      <th>DPI</th>
      <th>R</th>
      <th>dDPI</th>
      <th>condind</th>
    </tr>
    <tr>
      <th>sv</th>
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
      <th>1</th>
      <td>0.001383</td>
      <td>0.000003</td>
      <td>0.000003</td>
      <td>0.000244</td>
      <td>0.001594</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.003785</td>
      <td>0.000010</td>
      <td>0.000007</td>
      <td>0.001425</td>
      <td>0.135836</td>
      <td>4.142638</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.310490</td>
      <td>0.000028</td>
      <td>0.000037</td>
      <td>0.012988</td>
      <td>0.000640</td>
      <td>7.798541</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.263488</td>
      <td>0.004662</td>
      <td>0.004818</td>
      <td>0.984368</td>
      <td>0.048055</td>
      <td>39.405786</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.420854</td>
      <td>0.995297</td>
      <td>0.995135</td>
      <td>0.000975</td>
      <td>0.813874</td>
      <td>375.614256</td>
    </tr>
  </tbody>
</table>
</div>



## Okay, so I have collinearity, now what? 
Once variates with near dependencies identified, use regression of one variate by the other. The authors recommend that individual regression analyses need be carried out using one of the collinears against the others. This fits with the idea of "conditioning" the matrix $X_1$ on $X_2$ and vice versa. This aspect is beyond the scope of this post (and I would prefer to go in-depth into it as opposed to cluttering this with more theory), since the VDP is what really piqued my attention. For further reading, see the references section.

### References

- Belsley, Kuh, &amp; Welsch. (1980). Regression diagnostics. identifying influential data and sources of collinearity. Wiley.
- [Deisenroth, M. P., Faisal, A. A., Ong, C. S., &amp; Kamiński, F. (2022). Matematyka W Uczeniu Maszynowym. Helion](https://mml-book.com)
- [Singular Value Decomposition as Simply as Possible](https://gregorygundersen.com/blog/2018/12/10/svd/)
