# PDE-FIND-Project

Given a set of observed measurements of a dynamical system on a discretized spatiotemporal domain, can we discover its underlying governing partial differential equation(s)?

In this project, we intend to recover the governing, time-dependent PDE(s) of **three unknown systems** by constructing an overcomplete candidate library from the solution (observed measurements) and its derivatives, and then selecting the terms that are most informative about the dynamics of the system.


## Contents
1. General Framework for PDE Discovery
2. Computing Derivatives
3. Sparse Regression
4. PDE-1 Discovery
5. PDE-2 Discovery
6. PDE-3 Discovery



## General Framework for PDE Discovery

Our goal is to identify the underlying partial differential equation (PDE) from measurement data. For a 1+1D system, where the solution $u(x,t)$ is measured on a discretized spatiotemporal domain $\tilde{\Omega}$ with $n$ spatial points and $m$ time points, we first arrange the data into a vector form. Specifically, the time derivative $u_t$ is discretized into a column vector

$$
\mathbf{U}_t \in \mathbb{R}^{(n\cdot m)\times 1} =
\begin{bmatrix}
u_t(x_0,t_0) \\
u_t(x_1,t_0) \\
u_t(x_2,t_0) \\
\vdots \\
u_t(x_{n-1},t_m) \\
u_t(x_{n},t_m)
\end{bmatrix}.
$$

At the same time, we construct a feature library (or candidate matrix) $\Theta$ that consists of the measured solution $u$ and its spatial (and possibly mixed) derivatives. For example, a typical candidate library might look like

$$
\Theta =
\begin{bmatrix}
1 & u(x_0,t_0) & u_x(x_0,t_0) & u(x_0,t_0)^2 & \cdots & u^3\,u_{xxx}(x_0,t_0) \\
1 & u(x_1,t_0) & u_x(x_1,t_0) & u(x_1,t_0)^2 & \cdots & u^3\,u_{xxx}(x_1,t_0) \\
1 & u(x_2,t_0) & u_x(x_2,t_0) & u(x_2,t_0)^2 & \cdots & u^3\,u_{xxx}(x_2,t_0) \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
1 & u(x_{n-1},t_m) & u_x(x_{n-1},t_m) & u(x_{n-1},t_m)^2 & \cdots & u^3\,u_{xxx}(x_{n-1},t_m) \\
1 & u(x_{n},t_m)   & u_x(x_{n},t_m)   & u(x_{n},t_m)^2   & \cdots & u^3\,u_{xxx}(x_{n},t_m)
\end{bmatrix} \in \mathbb{R}^{(n\cdot m)\times D},
$$

where $D$ is the number of candidate functions. We assume that this library is overcomplete, meaning that the true PDE can be represented as a sparse linear combination of these candidate functions.

The PDE is then written in the linear regression form

$$
\mathbf{U}_t = \Theta\,\xi,
$$

where $\xi \in \mathbb{R}^{D \times 1}$ is a coefficient vector. By applying a sparse regression method (such as Sequential Threshold Ridge Regression, STRidge), we solve for $\xi$. Each nonzero entry $\xi_j \neq 0$ indicates that the candidate term $\Theta_j$ plays a role in the governing PDE. The recovered equation is then

$$
u_t(x,t) = \sum_{j=1}^{D} \Theta_j(x,t)\,\xi_j.
$$

For a coupled 2+1D system (File 3), the procedure is analogous but extended to two fields. The measurements for $u(x,y,t)$ and $v(x,y,t)$ are arranged such that

$$
\mathbf{U}_t \in \mathbb{R}^{(n\cdot m)\times 2} =
\begin{bmatrix}
u_t(x_0,y_0,t_0) & v_t(x_0,y_0,t_0) \\
u_t(x_1,y_0,t_0) & v_t(x_1,y_0,t_0) \\
\vdots & \vdots \\
u_t(x_n,y_m,t_M) & v_t(x_n,y_m,t_M)
\end{bmatrix}.
$$

We use the same candidate library $\Theta \in \mathbb{R}^{(n\cdot m)\times D}$ constructed from $u$, $v$, and their spatial derivatives (e.g., $u_x, u_y, v_x, v_y$, etc.). The coupled system is then represented as

$$
\mathbf{U}_t = \Theta\,\Xi,
$$

where $\Xi \in \mathbb{R}^{D\times 2}$ is a coefficient matrix. The first column of $\Xi$ corresponds to the PDE for $u_t$ and the second column to $v_t$. Thus, we have

$$
\begin{aligned}
u_t(x,y,t) &= \sum_{j=1}^{D} \Theta_j(x,y,t)\,\xi_j^{(u)}, \\
v_t(x,y,t) &= \sum_{j=1}^{D} \Theta_j(x,y,t)\,\xi_j^{(v)}.
\end{aligned}
$$



## Computing Derivatives

In our approach, both low- and high-order derivatives from the solution $u(x,t)$ (and similarly for $v(x,t)$ in coupled systems) at each observational point are estimated to ensure robust estimation from potentially noisy data. The ultimate goal is to be able to construct the feature library $\Theta$ used in our sparse regression framework for PDE discovery.

The function ``TikhonovDiff`` is employed to compute derivatives via Tikhonov regularization. Specifically, it solves the minimization problem

$$
\min_{g}\,\|Ag - f\|_2^2 + \lambda \|Dg\|_2^2,
$$

where $A$ approximates trapezoidal integration and $D$ represents a finite-difference operator. For $d>1$, the output of ``TikhonovDiff`` is further differentiated using the ``FiniteDiff`` function. The latter implements a second-order finite difference scheme; for instance, the first derivative is approximated as

$$
u'(x_i) \approx \frac{u(x_{i+1}) - u(x_{i-1})}{2\Delta x},
$$

with appropriate one-sided formulas at the boundaries, and similarly for higher-order derivatives. We assume a regularly spaced grid.

Variant functions, such as `ConvSmoother`, `PolyDiff`, and `PolyDiffPoint`, are designed to be employed for different cases depending on specific smoothing and differentiation requirements, including mainly when the data is expected to be noisy or the grid be non-uniform.


## Sparse Regression

Once the candidate library $\Theta \in \mathbb{R}^{(n\cdot m)\times D}$ is assembled from the discretized solution $u(x,t)$ and its spatial derivatives, and the temporal derivative is computed and organized into $U_t \in \mathbb{R}^{(n\cdot m)\times 1}$, the identification problem is cast as


$$
U_t = \Theta\,\xi.
$$


Because $\Theta$ is overdetermined (with $n\cdot m > D$), a standard least-squares solution would yield a dense vector $\xi$. However, we know from the underlying physics that the true PDE is parsimonious, so only a few terms are active. To promote sparsity in $\xi$, we solve the regularized problem

$$
\min_{\xi\in\mathbb{R}^{D}} \|U_t - \Theta\,\xi\|_2^2 + \lambda\,\|\xi\|_2^2,
$$

and then apply a hard threshold, setting coefficients with $|\xi_j|$ below a certain tolerance to zero. This procedure—implemented via our STRidge and TrainSTRidge functions—ensures that the final coefficient vector is sparse, with nonzero entries corresponding to the relevant terms in the PDE. In the case of coupled systems, the formulation is extended to

$$
U_t = \Theta\,\Xi,
$$

with $\Xi \in \mathbb{R}^{D\times 2}$ and each column recovered separately by sparse regression. In both cases, the sparsity of the solution is crucial: it leads to an interpretable model in which each nonzero coefficient $\xi_j$ (or $\xi_j^{(u)}$ and $\xi_j^{(v)}$ in the coupled case) directly indicates an active term in the governing equation.


---
---
---


   


## PDE-1 Discovery

<div align="center">

<img src="PDE-1/pde_1_approximated_terms.png" alt="PDE-1 Approximated Terms" style="width:600px;">
<p><b>Fig. 1:</b> Approximated PDE terms for PDE-1.</p>

<img src="PDE-1/pde_1_candidate_terms.png" alt="PDE-1 Candidate Terms" style="width:600px;">
<p><b>Fig. 2:</b> Candidate terms for PDE-1.</p>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="PDE-1/pde_1_residual_over_time.png" alt="Residual over time for PDE-1" style="width:400px;">
        <p><b>Fig. 3:</b> Residual over time for PDE-1.</p>
      </td>
      <td align="center">
        <img src="PDE-1/pde_1_scatter_true_vs_predicted_ut.png" alt="Scatter plot for PDE-1" style="width:400px;">
        <p><b>Fig. 4:</b> Scatter plot of true vs. reconstructed $u_t$ for PDE-1.<br></p>
      </td>
    </tr>
  </table>
</div>

<img src="PDE-1/pde_1_spatiotemporal_animation_comparison.gif" alt="PDE-1 Spatiotemporal Animation" style="width:600px;">
<p><b>Fig. 5:</b> Spatiotemporal animation comparing true and approximated $u_t$ for PDE-1.</p>

<img src="PDE-1/pde_1_ut_heatmap_comparison.png" alt="PDE-1 u_t Heatmap Comparison" style="width:600px;">
<p><b>Fig. 6:</b> Heatmap comparison of $u_t$ for PDE-1.</p>

</div>



## PDE-2 Discovery

<div align="center">

<img src="PDE-2/pde_2_bar_approximated_terms.png" alt="PDE-2 Approximated Terms" style="width:600px;">
<p><b>Fig. 7:</b> Approximated PDE terms for PDE-2.</p>

<img src="PDE-2/pde_2_candidate_terms.png" alt="PDE-2 Candidate Terms" style="width:600px;">
<p><b>Fig. 8:</b> Candidate terms for PDE-2.</p>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="PDE-2/pde_2_residual_over_time.png" alt="Residual over time for PDE-2" style="width:400px;">
        <p><b>Fig. 9:</b> Residual over time for PDE-2.</p>
      </td>
      <td align="center">
        <img src="PDE-2/pde_2_scatter_true_vs_predicted_ut.png" alt="Scatter plot for PDE-2" style="width:400px;">
        <p><b>Fig. 10:</b> Scatter plot of true vs. reconstructed $u_t$ for PDE-2.<br></p>
      </td>
    </tr>
  </table>
</div>

<img src="PDE-2/pde_2_spatiotemporal_animation_comparison.gif" alt="PDE-2 Spatiotemporal Animation" style="width:600px;">
<p><b>Fig. 11:</b> Spatiotemporal animation comparing true and approximated $u_t$ for PDE-2.</p>

<img src="PDE-2/pde_2_ut_heatmap_comparison.png" alt="PDE-2 u_t Heatmap Comparison" style="width:600px;">
<p><b>Fig. 12:</b> Heatmap comparison of $u_t$ for PDE-2.</p>

</div>



## PDE-3 Discovery

#### Bar Terms (Grouped by Equation)

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="PDE-3/pde_3_bar_terms_ut.png" alt="PDE-3 Bar Terms u_t" style="width:400px;">
        <p><b>Fig. 13a:</b> Approximated PDE terms for $u_t$ (PDE-3).</p>
      </td>
      <td align="center">
        <img src="PDE-3/pde_3_bar_terms_vt.png" alt="PDE-3 Bar Terms v_t" style="width:400px;">
        <p><b>Fig. 13b:</b> Approximated PDE terms for $v_t$ (PDE-3).</p>
      </td>
    </tr>
  </table>
</div>

#### Candidate Terms

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="PDE-3/pde_3_candidate_terms.png" alt="PDE-3 Candidate Terms (Full)" style="width:400px;">
        <p><b>Fig. 14:</b> Candidate terms for PDE-3 (full data).</p>
      </td>
      <td align="center">
        <img src="PDE-3/pde_3_candidate_terms_downsampled.png" alt="PDE-3 Candidate Terms (Downsampled)" style="width:400px;">
        <p><b>Fig. 15:</b> Candidate terms for PDE-3 (downsampled).</p>
      </td>
    </tr>
    <tr>
      <td colspan="2" align="center">
        <img src="PDE-3/pde_3_candidate_terms_spatial_full.png" alt="PDE-3 Spatial Candidate Terms" style="width:400px;">
        <p><b>Fig. 16:</b> Spatial candidate terms for PDE-3 (full data, fixed time slice).</p>
      </td>
    </tr>
  </table>
</div>

#### Residual Over Time

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="PDE-3/pde_3_residual_over_time_ut.png" alt="PDE-3 Residual over time u_t" style="width:400px;">
        <p><b>Fig. 17a:</b> Residual over time for $u_t$ (PDE-3).</p>
      </td>
      <td align="center">
        <img src="PDE-3/pde_3_residual_over_time_vt.png" alt="PDE-3 Residual over time v_t" style="width:400px;">
        <p><b>Fig. 17b:</b> Residual over time for $v_t$ (PDE-3).</p>
      </td>
    </tr>
  </table>
</div>

#### Scatter Plots

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="PDE-3/pde_3_scatter_true_vs_predicted_ut.png" alt="PDE-3 Scatter u_t" style="width:400px;">
        <p><b>Fig. 18a:</b> Scatter plot for $u_t$ (PDE-3). <i>(All spatial and temporal points pooled.)</i></p>
      </td>
      <td align="center">
        <img src="PDE-3/pde_3_scatter_true_vs_predicted_vt.png" alt="PDE-3 Scatter v_t" style="width:400px;">
        <p><b>Fig. 18b:</b> Scatter plot for $v_t$ (PDE-3). <i>(All spatial and temporal points pooled.)</i></p>
      </td>
    </tr>
  </table>
</div>

#### Animations (Fixed y-Slice)

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="PDE-3/pde_3_animate_ut_slice.gif" alt="PDE-3 Animate u_t Slice" style="width:400px;">
        <p><b>Fig. 19a:</b> Animation of $u_t$ (fixed y-slice) for PDE-3.</p>
      </td>
      <td align="center">
        <img src="PDE-3/pde_3_animate_vt_slice.gif" alt="PDE-3 Animate v_t Slice" style="width:400px;">
        <p><b>Fig. 19b:</b> Animation of $v_t$ (fixed y-slice) for PDE-3.</p>
      </td>
    </tr>
  </table>
</div>

#### Overall Animations

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="PDE-3/u_t_animation.gif" alt="PDE-3 u_t Animation" style="width:400px;">
        <p><b>Fig. 20a:</b> Overall animation of $u_t$ for PDE-3.</p>
      </td>
      <td align="center">
        <img src="PDE-3/v_t_animation.gif" alt="PDE-3 v_t Animation" style="width:400px;">
        <p><b>Fig. 20b:</b> Overall animation of $v_t$ for PDE-3.</p>
      </td>
    </tr>
  </table>
</div>
