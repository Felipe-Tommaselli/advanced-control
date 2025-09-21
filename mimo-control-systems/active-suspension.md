
# Automobile Passenger Comfort Assured Through LQG/LQR Active Suspension — Text + Equations (Reconstructed)

**Authors:** Hamid D. Taghirad; E. Esmailzadeh  
**Affiliations:** Center for Intelligent Machines, McGill University; Sharif University of Technology  
**Source:** *Journal of Vibration and Control*, 4:603–618, 1998

> **Note** — This is a faithful text conversion of the provided PDF with equations reconstructed from the figures and standard LQG/LQR formulations for active suspensions. Some symbols were normalized for clarity (bold lowercase for vectors, uppercase for matrices).

---

## Abstract
An analytical investigation of a half-car model including passenger dynamics, subjected to random road disturbances is performed, and the advantage of active over conventional passive suspension systems are examined. Two different performance indices for optimal controller design are proposed. The performance index is a quantification of both ride comfort and road handling. Due to practical limitations, all the states required for the state-feedback controller are not measurable, and thus must be estimated with an observer. Stochastic inputs are applied to simulate realistic road surface conditions, and statistical comparisons between passive system and the two controllers, with and without state estimator, are carried out to gain a clearer insight into the performance of the controllers. The simulation results demonstrate that an optimal observer-based controller, when including passenger acceleration in the performance index, retains both excellent ride comfort and road handling characteristics.

**Keywords:** Active suspension; observer-based control; ride comfort; passenger dynamics.

---

## 1. Introduction
Demand for better ride comfort and controllability of road vehicles has motivated many automotive industries to consider the use of active suspensions. These electronically controlled suspension systems can potentially improve the ride comfort as well as the road handling of the vehicle. Generally, a vehicle suspension system may be categorized as passive, semi‑active, or fully‑active.

Passive systems employ conventional springs and dampers with fixed characteristics and no feedback control. Semi‑active suspensions vary damping in real time via controllable dampers in parallel with springs. Fully active suspensions add actuators (hydraulic/pneumatic) acting in parallel with the passive elements and require sensors and feedback control to command forces, with attendant power limitations that must be considered in design.

Design trade‑offs involve ride comfort, body motion, road handling (tire–road contact), and suspension travel; they cannot be minimized simultaneously. State‑feedback optimal control provides a systematic way to encode these trade‑offs via quadratic costs while respecting actuator limits. Prior work ranges from quarter‑car to full‑car models; observer‑based (output‑feedback) implementations are important because not all states are measurable in practice.

This paper contributes: (i) a half‑car model that **includes passenger dynamics** (two vertical passenger degrees‑of‑freedom) to explicitly target ride comfort; and (ii) a **statistical comparison** framework using bounds at fixed probabilities (e.g., 90%) under random road profiles (ISO PSD‑based) to compare passive vs. active (with/without observer) designs.

---

## 2. Mathematical Modeling

### 2.1 Vehicle–Passenger Model
We adopt a half‑car (longitudinal) model with **six** generalized coordinates
\[
\mathbf{q} \;=\; \begin{bmatrix} x_b & \theta & x_{t1} & x_{t2} & x_{p1} & x_{p2} \end{bmatrix}^\top,
\]
where \(x_b\) is the body bounce (heave), \(\theta\) the body pitch, \(x_{t1},x_{t2}\) the front/rear tire vertical deflections, and \(x_{p1},x_{p2}\) the vertical motions of two passengers (idealized as seat–passenger lumped masses). Suspension, tires, and seats are modeled as linear spring–damper pairs; two actuators apply controllable forces in parallel with the suspensions.

The second‑order dynamics can be written compactly as
\[
\mathbf{M}\,\ddot{\mathbf{q}} \;+\; \mathbf{C}\,\dot{\mathbf{q}} \;+\; \mathbf{K}\,\mathbf{q}
\;=\; \mathbf{B}_u\,\mathbf{u} \;+\; \mathbf{B}_w\,\mathbf{w},
\tag{1}
\]
where \(\mathbf{u}\in\mathbb{R}^2\) are the front/rear actuator forces and \(\mathbf{w}\) represents road vertical inputs at the tire contact patches (modeled as random processes derived from ISO PSD classes).

Define the state
\[
\mathbf{x} \;=\; \begin{bmatrix} \mathbf{q} \\ \dot{\mathbf{q}} \end{bmatrix} \in \mathbb{R}^{12},
\qquad
\dot{\mathbf{x}} \;=\; \mathbf{A}\mathbf{x} \;+\; \mathbf{B}\mathbf{u} \;+\; \mathbf{G}\mathbf{w},
\tag{2}
\]
with \(\mathbf{A},\mathbf{B},\mathbf{G}\) obtained by the standard first‑order augmentation of (1). Equation (2) is the basis for controller/observer design and simulation.

**States of interest.** Ride comfort is reflected in **passenger accelerations**, road handling in **tire deflections**, with body bounce/pitch and suspension deflections also monitored.

### 2.2 Road Roughness Model (ISO‑based)
Road elevation is modeled as a **zero‑mean stationary random process** characterized by its power spectral density (PSD). ISO (1982) defines road roughness classes via PSD levels. We synthesize road inputs with target PSDs using FFT‑based shaping of Gaussian noise such that the resulting profiles match the desired “poor road” class used in the simulations.

---

## 3. Optimal Controller Design

We first consider full‑state feedback of the form
\[
\mathbf{u} \;=\; -\,\mathbf{K}\,\mathbf{x}.
\tag{3}
\]

### 3.1 Conventional Method (CM)
Minimize the quadratic cost
\[
J_{\mathrm{CM}} \;=\; \int_0^\infty
\left( \mathbf{x}^\top \mathbf{Q}\,\mathbf{x} \;+\; \mathbf{u}^\top \mathbf{R}\,\mathbf{u} \right)\,dt,
\quad \mathbf{Q}\succeq 0,\; \mathbf{R}\succ 0.
\tag{4}
\]
For stabilizable \((\mathbf{A},\mathbf{B})\), the optimal gain is
\[
\mathbf{K} \;=\; \mathbf{R}^{-1}\mathbf{B}^\top \mathbf{P},
\tag{5}
\]
where \(\mathbf{P}\) solves the **algebraic Riccati equation (ARE)**
\[
\mathbf{A}^\top \mathbf{P} \;+\; \mathbf{P}\mathbf{A}
\;-\; \mathbf{P}\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^\top \mathbf{P}
\;+\; \mathbf{Q} \;=\; \mathbf{0}.
\tag{6}
\]
The closed‑loop dynamics used in simulation are
\[
\dot{\mathbf{x}} \;=\; (\mathbf{A}-\mathbf{B}\mathbf{K})\,\mathbf{x} \;+\; \mathbf{G}\mathbf{w}.
\tag{7}
\]

### 3.2 Acceleration‑Dependent Method (ADM)
To emphasize **ride comfort**, include the passengers’ accelerations in the cost. Let
\[
\mathbf{z} \;\triangleq\; \begin{bmatrix} \ddot{x}_{p1} \\[2pt] \ddot{x}_{p2} \end{bmatrix}
\;=\; \mathbf{V}\,\mathbf{x},
\tag{8}
\]
where \(\mathbf{V}\) (constant) maps states to the two seat/passenger accelerations (linear in \(\mathbf{x}\) for the adopted model). Choose a positive semidefinite \(\mathbf{S}=\mathrm{diag}(s_1,s_2)\) and minimize
\[
J_{\mathrm{ADM}} \;=\; \int_0^\infty
\left( \mathbf{x}^\top \mathbf{Q}\,\mathbf{x} \;+\; \mathbf{u}^\top \mathbf{R}\,\mathbf{u}
\;+\; \mathbf{z}^\top \mathbf{S}\,\mathbf{z} \right)\, dt.
\tag{9}
\]
Since \(\mathbf{z}=\mathbf{V}\mathbf{x}\), this is equivalent to (4) with a **modified state weight**
\[
\widetilde{\mathbf{Q}} \;=\; \mathbf{Q} \;+\; \mathbf{V}^\top \mathbf{S}\,\mathbf{V}.
\tag{10}
\]
Thus (5)–(7) hold with \(\mathbf{Q}\) replaced by \(\widetilde{\mathbf{Q}}\).

---

## 4. Optimal Observer (LQG) Design

Not all states are measurable. Let the available outputs be, e.g., **relative suspension travels** and **relative passenger bounces**:
\[
\mathbf{y} \;=\; \mathbf{C}\,\mathbf{x} \;+\; \mathbf{v},
\tag{11}
\]
with measurement noise \(\mathbf{v}\). A **Kalman observer** provides the state estimate
\[
\dot{\hat{\mathbf{x}}} \;=\; \mathbf{A}\,\hat{\mathbf{x}} \;+\; \mathbf{B}\,\mathbf{u}
\;+\; \mathbf{L}\big(\mathbf{y}-\mathbf{C}\hat{\mathbf{x}}\big),
\qquad \mathbf{u} \;=\; -\mathbf{K}\,\hat{\mathbf{x}},
\tag{12}
\]
where the **Kalman gain**
\[
\mathbf{L} \;=\; \mathbf{P}_e\,\mathbf{C}^\top \mathbf{V}_n^{-1},
\tag{13}
\]
is obtained from the estimator ARE
\[
\mathbf{A}\mathbf{P}_e \;+\; \mathbf{P}_e \mathbf{A}^\top
\;-\; \mathbf{P}_e \mathbf{C}^\top \mathbf{V}_n^{-1} \mathbf{C}\,\mathbf{P}_e
\;+\; \mathbf{W}_n \;=\; \mathbf{0},
\tag{14}
\]
with process/measurement noise covariances \(\mathbf{W}_n=\mathbb{E}[\mathbf{w}\mathbf{w}^\top]\) and \(\mathbf{V}_n=\mathbb{E}[\mathbf{v}\mathbf{v}^\top]\). The resulting output‑feedback controller is the standard **LQG** combination of (5) and (13) under detectability/observability conditions.

For simulation, plant and observer may be integrated together by augmenting (7) with (12).

---

## 5. Simulation & Statistical Comparison

### 5.1 Gaussian Bounds for Random Responses
Under Gaussian excitation, many response variables are approximately **zero‑mean Gaussian** with standard deviation \(\sigma\). Useful probability relations are
\[
\mathbb{P}\big(|x|\le X\big) \;=\; \operatorname{erf}\!\left(\frac{X}{\sqrt{2}\,\sigma}\right), 
\qquad
\mathbb{P}\big(|x|\ge X\big) \;=\; \operatorname{erfc}\!\left(\frac{X}{\sqrt{2}\,\sigma}\right).
\tag{15}
\]
Hence the **90% bound** is
\[
X_{90} \;=\; \sqrt{2}\,\sigma\,\operatorname{erf}^{-1}(0.90).
\tag{16}
\]
Design limits (on body motion, passenger acceleration, tire/suspension deflection, actuator force) are enforced by tuning weights so that the closed‑loop \(X_{90}\) values respect prescribed bounds (Table‑style limits in the paper).

### 5.2 Observer Performance
Simulations (typical mid‑size car parameters) show the Kalman observer tracks the true states closely despite the random road input, validating detectability and the chosen noise covariances. Estimated vs. true signals overlay tightly (cf. paper’s Fig. 2).

### 5.3 Active vs. Passive
Active (ADM‑tuned) vs. passive: active control markedly reduces **passenger accelerations** (ride comfort) and also reduces **tire deflections** (road handling), indicating no trade‑off degradation in this setup. Quantitatively, body bounce and passenger accelerations drop to roughly **half**, and tire deflections by **≥40%** at the 90% probability bound (paper’s Table 4).

### 5.4 CM vs. ADM Controllers
CM and ADM yield similar body/tire responses, but **ADM** achieves **significantly lower passenger accelerations** by explicitly penalizing them. This improvement comes with somewhat **larger actuator forces**, still within practical limits due to weight tuning (paper’s Fig. 5).

---

## 6. Conclusion
Optimal state‑feedback (LQR) and output‑feedback (LQG) controllers for an active half‑car suspension **including passenger dynamics** can simultaneously enhance **ride comfort** and **road handling** under realistic stochastic road profiles. A statistical (probabilistic bounds) evaluation provides a clear, quantitative basis for comparing designs and tuning weights to satisfy practical constraints, including actuator limits.

---

## References (as in the source)
Alleyne, A. & Hedrick, J.K. (1992). Nonlinear control of a quarter car active suspension. *Proc. ACC*.
Bryson, A.E. & Ho, Y. (1975). *Applied Optimal Control*. Wiley.
Caudill, R.J., Sweet, L.M., & Oda, K. (1982). Magnetic guidance of conventional railroad vehicles. *ASME JDSMC*, 104(1), 36–42.
Chen, C.T. (1970). *Introduction to Linear System Theory*. Holt, Rinehart and Winston.
Crosby, M.J. & Karnopp, D.C. (1973). The active damper. *Shock and Vibration Bulletin*, 43(2), 102–108.
Elmadany, M.M. (1992). Integral and state variable feedback controllers… *Computers & Structures*, 42(2), 237–244.
Elmadany, M.M. & Samaha, M.E. (1992). Optimum ride control… *J. Sound and Vibration*, 156(2), 269–281.
Esmailzadeh, E. (1979). Servo‑valve controlled pneumatic suspensions. *J. Mech. Eng. Sci.*, 21(1), 7–18.
Esmailzadeh, E. & Bateni, H. (1992). Optimal active vehicle suspensions… *SAE Trans.*, 101, 784–795.
Esmailzadeh, E. & Taghirad, H.D. (1995). State‑feedback control for passenger ride dynamics. *TCSME*, 19(4), 495–508.
Hrovat, D. (1990). Optimal active suspension structures for quarter‑car models. *Automatica*, 25(5), 845–860.
ISO (1982). Reporting vehicle road surface irregularities. ISO/TC108/SC2/WG4 N57.
Krtolica, R. & Hrovat, D. (1992). Optimal active suspension control based on a half‑car model. *IEEE TAC*, 37(4), 528–532.
Miller, L.R. (1988). Tuning passive, semi‑active, and fully active suspension systems. *Proc. CDC*.
Rajamani, R. & Hedrick, J.K. (1993). Adaptive observer for active automotive suspensions. *Proc. ACC*.
Shannan, J.E. & Vanderploeg, M.J. (1989). A vehicle handling model with active suspensions. *J. Mech., Trans., and Automation in Design*, 111(3), 375–381.
Sharp, R. & Hassan, S. (1986). Relative performance of passive, active, and semi‑active car suspensions. *Proc. IMechE*, 200(D3), 219–228.
Sinha, P.K., Wormley, D.N., & Hedrick, J.K. (1978). Rail passenger vehicle lateral dynamic performance improvement through active control. *ASME*, 78‑WA/DSC‑14.
Thomson, W.T. (1988). *Theory of Vibration with Applications*. Prentice‑Hall.
Wright, P.G. & Williams, D.A. (1984). The application of active suspension to high performance road vehicles. IMechE, C239/84.
Yue, C., Butsuen, T., & Hedrick, J.K. (1989). Alternative control for automotive active suspensions. *ASME JDSMC*, 111, 286–290.

---

### Appendix: Symbols (typical)
\(\mathbf{M},\mathbf{C},\mathbf{K}\): mass, damping, stiffness matrices;  
\(\mathbf{B}_u,\mathbf{B}_w\): actuator and road‑input distribution matrices;  
\(\mathbf{A},\mathbf{B},\mathbf{G}\): first‑order state‑space matrices;  
\(\mathbf{Q},\mathbf{R},\mathbf{S}\): LQR/ADM weights; \(\mathbf{V}\): passenger‑accel mapping;  
\(\mathbf{C}\): output matrix; \(\mathbf{W}_n,\mathbf{V}_n\): process/measurement noise covariances;  
\(\operatorname{erf},\operatorname{erfc}\): error and complementary error functions.
