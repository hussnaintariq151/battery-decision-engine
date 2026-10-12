# battery-decision-engine
Physics-Informed Fourier Neural Operator Decision Engine for Zero-Plating Fast Charging of Lithium-Ion Batteries via Real-Time State Estimation &amp; Barrier-Riding Control.

# Physics-Informed Battery Decision Engine

[[https://img.shields.io/badge/License-MIT-yellow.svg]](https://opensource.org/licenses/MIT)
[[https://img.shields.io/badge/python-3.10+-blue.svg]](https://www.python.org/downloads/)
[[https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg]](https://pytorch.org/)
[[https://img.shields.io/badge/PyBaMM-23.5+-green.svg]](https://pybamm.org/)

Official open-source repository for the paper:  
**"A Physics-Informed Fourier Neural Operator Framework for Zero-Plating Fast Charging of Lithium-Ion Batteries via Real-Time Electrochemical State Estimation and Barrier-Riding Optimal Control"**  
*Hussnain Tariq (Institute of Chemistry, The Islamia University of Bahawalpur)*

---

## ⚡ Key Highlights
- **4,349× Inference Acceleration**: Sub-millisecond (1.84 ms on GPU, <15 ms on CPU) surrogate evaluation over numerical Doyle–Fuller–Newman (DFN/P2D) solvers (8.0 s).
- **Physical Mass Conservation**: Explicit integration of cumulative charge $Q(t) = \int I\,dt$ resolves coordinate-invariance blindspots, plunging voltage RMSE from 108.2 mV to 5.40 mV (95% error cut).
- **Real-Time Degradation Observability**: Ensemble Smoother with Multiple Data Assimilation (ES-MDA) filters $\pm 10\,\text{mV}$ sensor noise down to 5.94 mV and tracks unobservable lithium plating overpotential $\eta_{\text{plating}}$ within $\pm 3.22\,\text{mV}$ (95% CI).
- **Zero-Plating Barrier-Riding NMPC**: Cuts 10%–80% SoC fast-charging duration by **21.7%** (33.7 min vs 43.0 min CC-CV) while strictly guaranteeing $\eta_{\text{plating}} \ge +5.0\,\text{mV}$ and reducing SEI growth by 21.5%.
- **Dual Hardware Benchmarking**: Verified against Oxford Kokam pouch cells (LCO) and commercial RWTH Aachen Sanyo 18650 cylindrical cells (NMC, 2.2C fast-charge), achieving an industrial-grade **33.02 mV tracking RMSE** (86.0% error reduction).

---

## 📁 Repository Structure
```text
├── src/
│   ├── models/          # 1D Fourier Neural Operator (FNO-1D) architecture
│   ├── observers/       # ES-MDA Bayesian data assimilation
│   ├── control/         # Barrier-Riding NMPC optimal controller
│   └── physics/         # Continuum DFN PyBaMM solver interface
├── notebooks/           # Interactive evaluation and hardware validation notebooks
├── data/                # Normalization statistics and dataset download instructions
└── figures/             # High-resolution benchmark figures
