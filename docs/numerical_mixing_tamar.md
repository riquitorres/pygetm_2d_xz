# Numerical Mixing Analysis in the Tamar 2D Slice Model

**Method:** Klingbeil et al. (2014) discrete variance decay framework  
**Implementation:** Online `salt_sq` tracer in pygetm  
**Model:** 2D x/z curvilinear sigma-coordinate slice, Tamar Estuary

---

## Table of Contents

1. [Theoretical Background](#1-theoretical-background)
2. [The Salinity Variance Budget](#2-the-salinity-variance-budget)
3. [Separating Physical and Numerical Mixing](#3-separating-physical-and-numerical-mixing)
4. [Online Diagnostic Method](#4-online-diagnostic-method)
5. [Implementation in pygetm](#5-implementation-in-pygetm)
6. [Output Variables](#6-output-variables)
7. [Post-Processing and Analysis](#7-post-processing-and-analysis)
8. [Caveats and Limitations](#8-caveats-and-limitations)
9. [Key References](#9-key-references)

---

## 1. Theoretical Background

Numerical (spurious) mixing arises in ocean and estuarine models because finite-difference and finite-volume advection schemes are not perfectly conservative of tracer variance. When a sharp salinity front is advected through a grid cell, the scheme inevitably smooths it slightly, destroying salinity variance in a way that has no physical basis. This is indistinguishable in the model output from real turbulent mixing — yet can be comparable in magnitude to it, particularly in estuaries where strong fronts and large velocity shears coexist.

The **Discrete Variance Decay (DVD) framework** of Klingbeil et al. (2014) provides a rigorous, cell-by-cell diagnostic of how much variance is destroyed by the advection scheme at each timestep, and how this compares to physically prescribed mixing via the turbulence closure. The method is non-invasive: it requires only that a second tracer tracking $s^2$ be transported by the same advection scheme as salinity.

### Why it Matters for the Tamar

The Tamar is a partially mixed, macrotidal estuary with a strong axial salinity gradient (0 PSU at Gunnislake to ~35 PSU at Plymouth Sound). Tidal excursions are large relative to the halocline thickness, meaning the salinity front is repeatedly compressed and stretched each tidal cycle. In this regime:

- Upwind-biased or flux-limited advection schemes (e.g. SUPERBEE, which pygetm uses by default) can generate numerical mixing comparable to physical mixing
- Numerical mixing is concentrated near the salt front and at the pycnocline, exactly where physical mixing is most dynamically important
- Misattributing numerical mixing as physical would lead to overestimated stratification mixing and incorrect estuarine exchange flow diagnostics

---

## 2. The Salinity Variance Budget

The evolution of salinity variance in a model grid cell $k$ of volume $V_k$ is governed by:

$$\frac{\partial}{\partial t}(V_k \langle s^2 \rangle_k) = \underbrace{-\oint s^2 \, \mathbf{u} \cdot \hat{n} \, dA}_{\text{advective flux}} + \underbrace{2 \oint s \, \kappa \nabla s \cdot \hat{n} \, dA}_{\text{diffusive flux}} - \underbrace{2 \int_{V_k} \kappa |\nabla s|^2 \, dV}_{\chi^{phy}_k}$$

The last term, $\chi^{phy}_k$, is the **physical variance dissipation rate** — the destruction of salinity variance by turbulent diffusion. It is always non-negative (diffusion destroys, not creates, variance).

In a discretised model, the advection operator does not conserve variance exactly. The **numerical variance dissipation rate** is:

$$\chi^{num}_k = -\left[\frac{d}{dt}(V_k \langle s^2 \rangle_k)\right]^{adv} + V_k \langle s \rangle_k \left[\frac{d}{dt}\langle s \rangle_k\right]^{adv}$$

This is the difference between how the advection scheme evolves $\langle s^2 \rangle$ and how it would evolve if it were exact (i.e. if $\langle s^2 \rangle_{adv} = \langle s \rangle_{adv}^2$).

---

## 3. Separating Physical and Numerical Mixing

The total mixing in the model is:

$$\chi^{tot}_k = \chi^{phy}_k + \chi^{num}_k \geq 0$$

These are diagnosed separately:

### Physical mixing $\chi^{phy}$

Computed directly from the turbulence closure output:

$$\chi^{phy}_k = 2\, \kappa_v \left(\frac{\partial s}{\partial z}\right)^2$$

where $\kappa_v$ is the vertical eddy diffusivity from GOTM (the turbulence module embedded in pygetm). In a 2D x/z slice model there is no horizontal diffusion term (no $y$-dimension), and explicit horizontal diffusivity is typically set to zero, so this captures all physical mixing.

### Numerical mixing $\chi^{num}$

Computed from the divergence between advecting $s^2$ and squaring the advected $s$:

$$\chi^{num}_k = \frac{s_{after}^2 - (s^2)_{after}}{\Delta t} \geq 0$$

where:
- $s_{after}$ = salinity after the advection step
- $(s^2)_{after}$ = the `salt_sq` tracer value after the advection step (same scheme applied to $s^2$)
- $\Delta t$ = model timestep

If the advection scheme were perfect, $s_{after}^2 \equiv (s^2)_{after}$ and $\chi^{num} = 0$ everywhere. Any positive residual is variance destroyed by numerical diffusion in the scheme.

---

## 4. Online Diagnostic Method

The key insight is that $\chi^{num}$ can be computed exactly — without any offline approximation — if we add $s^2$ as a second tracer transported by the **same advection scheme** as salinity. This is Klingbeil et al.'s Method II, and is exact at machine precision.

### Comparison with Offline Methods

| Approach | Accuracy | Cost | Notes |
|---|---|---|---|
| Online `salt_sq` tracer (this work) | Exact | +1 tracer per timestep | No offline error |
| Offline $s^2$ budget residual | ~60% overestimate at hourly output | Post-processing only | Hourly output insufficient |
| Offline at sub-tidal output (~10 min) | ~5–10% overestimate | Large storage cost | Acceptable approximation |

The offline overestimate at hourly output arises because within each output interval, variance is destroyed and partially re-created by tidal advection in ways that the hourly snapshots cannot resolve. The online method avoids this entirely because it operates at the model timestep.

---

## 5. Implementation in pygetm

The implementation is contained in `tamar_nummix_online.py`, which provides a `NumericalMixingMixin` class.

### Integration into the Run Script

The mixin is injected via Python's method resolution order (MRO) — no changes to pygetm source code are required:

```python
import pygetm
from tamar_nummix_online import NumericalMixingMixin

class TamarRun(NumericalMixingMixin, pygetm.simulation.Simulation):
    pass

sim = TamarRun(domain, ...)
sim.add_nummix_tracer()       # register salt_sq before sim.start()

output = sim.output_manager.add_netcdf_file("tamar_output.nc", interval=...)
output.request("salt", "nuh", "u", "w", "eta")
sim.add_nummix_output(output)  # adds chi_num, chi_phy, salt_sq to output

sim.output_manager.register_callback(sim._flush_nummix_output)

sim.start(t_start, timestep, ...)
while sim.time < t_end:
    sim.advance()
sim.finish()
```

### What Happens Each Timestep

```
─── timestep n ─────────────────────────────────────────────────────────────
  1. snapshot  s_before  = salt.values
  2. snapshot  sq_before = salt_sq.values
  3. sync OB:  salt_sq OBC ← (salt OBC)²        [prevents variance injection]
  4. pygetm advances salt and salt_sq through same advection scheme
  5. chi_num += (salt² - salt_sq) / dt           [accumulate]
  6. chi_phy += 2 * nuh * (ds/dz)²              [accumulate]
  7. count   += 1
─── output step (e.g. hourly) ──────────────────────────────────────────────
  8. chi_num_output = chi_num_acc / count        [time-average → NetCDF]
  9. chi_phy_output = chi_phy_acc / count
 10. reset accumulators
```

### Open Boundary Treatment

At the seaward open boundary, salinity values are prescribed from CMEMS climatology or tidal model. The `salt_sq` tracer requires a consistent boundary condition:

$$s^2_{OB} = (s_{OB})^2$$

This is enforced automatically by the mixin each timestep before the advection step. Without this, the boundary would either inject or remove spurious variance depending on the interior salinity, contaminating the diagnostic near the mouth.

### River Boundary Treatment

Rivers are set to `rivers_follow_target_cell=True` for `salt_sq`, meaning the river water is assigned $s^2$ equal to the square of the salinity in the cell the river flows into. This is physically correct: freshwater end-members (where $s \approx 0$) have $s^2 \approx 0$, and no variance correction is needed.

---

## 6. Output Variables

All variables are on the model grid `(time, z, x)` with `time` being the output averaging interval (hourly by default).

| Variable | Units | Description |
|---|---|---|
| `chi_num` | PSU² s⁻¹ | Numerical salinity variance dissipation rate (time-averaged) |
| `chi_phy` | PSU² s⁻¹ | Physical salinity variance dissipation rate (time-averaged) |
| `salt_sq` | PSU² | Advected $s^2$ tracer field |
| `salt` | PSU | Salinity (standard pygetm output) |
| `nuh` | m² s⁻¹ | Vertical eddy diffusivity from GOTM (needed to reproduce `chi_phy`) |

### Sign Conventions

- `chi_num` ≥ 0 always — numerical advection can only destroy variance
- `chi_phy` ≥ 0 always — turbulent diffusion destroys variance
- Both quantities are per unit volume (PSU² s⁻¹), not integrated

---

## 7. Post-Processing and Analysis

### Column-Integrated Mixing Profile

Integrate over depth at each along-axis position $x$:

$$M^{num}(x, t) = \int_0^{-H} \chi^{num}(x, z, t) \, B(x) \, dz \quad [\text{PSU}^2 \, \text{m} \, \text{s}^{-1}]$$

This gives a transect of mixing intensity along the estuary, useful for identifying where the salt front drives the most numerical mixing.

```python
import xarray as xr
import numpy as np

ds = xr.open_dataset("tamar_output.nc")
B  = xr.open_dataset("tamar_domain.nc")["B"]   # cross-section width (x,)

# Column-integrated numerical mixing [PSU² m s⁻¹]
M_num = (ds["chi_num"] * ds["dz"] * B).sum(dim="z")

# Column-integrated physical mixing
M_phy = (ds["chi_phy"] * ds["dz"] * B).sum(dim="z")

# Mixing ratio: fraction that is numerical
f_num = M_num / (M_num + M_phy)
```

### Domain-Integrated Totals

Integrate over the full estuary volume to get a scalar mixing budget:

$$\mathcal{M}^{num}(t) = \int_0^L \int_0^{-H} \chi^{num}(x,z,t) \, B(x) \, dz \, dx \quad [\text{PSU}^2 \, \text{m}^3 \, \text{s}^{-1}]$$

```python
dx = np.gradient(ds["x"].values)              # along-axis grid spacing (m)
M_num_total = float((M_num * dx).sum(dim="x"))
M_phy_total = float((M_phy * dx).sum(dim="x"))
print(f"Numerical fraction: {M_num_total / (M_num_total + M_phy_total):.1%}")
```

### Connection to TEF

The domain-integrated mixing $\mathcal{M}^{tot}$ can be compared to the TEF-derived mixing via the salinity variance budget of MacCready et al. (2018):

$$\mathcal{M}^{tot} = -\frac{d}{dt}\langle V \bar{s}^2 \rangle + \text{boundary salinity fluxes}$$

where the right-hand side is computed entirely from the TEF $Q(s)$ profiles and river/ocean salinities. This provides a cross-check: if $\mathcal{M}^{tot}$ from the $\chi$ fields matches the TEF-derived budget, the model's salt budget is closed and the mixing decomposition is consistent.

### Tidal Averaging

Mixing fields should be averaged over at least one full tidal cycle (M2 period ≈ 12.42 hours) before spatial analysis. The mixin accumulates at every model timestep and outputs at the requested interval; for tidal averaging set the output interval to 13 hours or use a Hanning filter in post-processing:

```python
# 30-hour low-pass Hanning filter (MacCready 2011 recommendation)
from scipy.signal import filtfilt, hann
T_filter = 30   # hours
win = hann(T_filter + 1)
win /= win.sum()
M_num_tidal_avg = xr.apply_ufunc(
    lambda x: filtfilt(win, 1, x, axis=0),
    M_num, input_core_dims=[["time"]], output_core_dims=[["time"]]
)
```

### Salinity-Class Decomposition

To understand where in salinity space mixing occurs (complementary to the spatial map), bin `chi_num` and `chi_phy` into salinity classes, weighted by cell volume:

```python
s_bins = np.linspace(0, 35, 71)
chi_by_sal = np.zeros((len(s_bins)-1, len(ds.time)))

s    = ds["salt"].values         # (time, z, x)
chi  = ds["chi_num"].values      # (time, z, x)
dz   = ds["dz"].values           # (time, z, x)
B    = B.values[np.newaxis, np.newaxis, :]   # broadcast

for ti in range(len(ds.time)):
    vol    = dz[ti] * B[0]       # (z, x)
    s_ti   = s[ti].ravel()
    chi_ti = (chi[ti] * vol).ravel()
    idx    = np.digitize(s_ti, s_bins) - 1
    valid  = (idx >= 0) & (idx < len(s_bins)-1)
    np.add.at(chi_by_sal[:, ti], idx[valid], chi_ti[valid])
```

This shows whether numerical mixing preferentially destroys variance in the freshwater end-member, the mixing zone, or near the oceanic boundary.

---

## 8. Caveats and Limitations

**Advection scheme dependency.** The diagnostic captures numerical mixing from advection only. With SUPERBEE (pygetm's default), numerical mixing is relatively low for smooth gradients but can be large at sharp fronts. Switching to a centred scheme would lower `chi_num` but increase spurious oscillations. The diagnostic is scheme-specific and should be reported alongside the scheme name.

**No horizontal diffusion term in `chi_phy`.** The 2D x/z slice has no explicit horizontal diffusion, so $\kappa_h$ does not appear in the physical mixing term. If you add horizontal diffusion to your simulation, add $2\kappa_h (\partial s / \partial x)^2$ to `chi_phy` in post-processing.

**Sigma-coordinate layer thickness.** The `chi_phy` computation uses finite differences of salinity across sigma layers. Near the free surface and in shallow regions, sigma layers compress and $\partial s / \partial z$ can be large. Check that `nuh` and `dz` are both consistent (i.e. from the same timestep) in the output.

**Width prescription.** The 2D slice does not resolve cross-estuary structure. When integrating `chi_num * B * dz` to get volumetric mixing rates, the cross-section width $B(x)$ should come from the same bathymetric dataset used to construct the domain, not from the curvilinear grid metrics alone.

**Offline comparison.** If comparing to an offline estimate (e.g. from a colleague using model output without `salt_sq`), be aware that hourly offline estimates overestimate online numerical mixing by up to 60% (Burchard & Rennau 2008). The online estimate from this implementation is the reference.

---

# Implementation of Kunt's approach to calculate and accumulate the fluxes at each timestep

This requires the addition of estimates of the salinity variance flux itself and not the difference between evolving $S^2$ and calculating $(S)^2$. 

So from the salinity equation: 

$V_k^{n+1} s_k^{n+1} = V_k^n s_k^n - \Delta t \sum\limits_{\text{faces}} F_s$

$F_{s^2}^{consistent}​=\bar{s}*F_s$​

specifically 

$\Delta (Var)_k​=s_k​−2\Delta t \sum\limits_{\text{faces}} \bar{s}_k ​F_s ​​​−[(s_k^{n+1}​)^2−(s_k^n​)^2]V_k^n​$

with the flux into the cell calculated with its own updated $\bar{S_k}$

Following a call sequence of:
Python: apply_3d()
  → self.u_3d()  / self.v_3d()  / self.w_3d()     [_pygetm compiled binding]
    → advection_uv_calculate() / advection_w_calculate()   [wrapper.F90]
      → advection%op%u2d() / v2d() / w3d()         [advection.F90.template]

... we need to modify both the fortran operators source code advection_base.F90 and advection.F90.template for both the addition of a new inout variable (chi_num) so that we can accumulate it over the advection Strangs as well as the cython fortran interface (wrapper.F90) and operators.py (the python advection class). Additionally, _pygetm.pyx also needs patching... advection.F90 and operators.F90 also need changing 


We can do this in stages to check that we are implementing things correctly... first to do the simple S2 advection but at this level rather than at the run script as it was done earlier. 

In 
```bash 
./src/operators/advection_schemes/advection_base.F90
```

I added the optional chi_num variable to the abstract interface that drops in to advection.F90_template. As per suggestion by Jorn chi_num is no longer an optional argument and instead the implementation follows apply_diffusion with a logical variable passed at the same time as the chi_num array. 

For the template
```bash 
./src/operators/advection_schemes/advection.F90_Template
```
the subroutines u2d, v2d and w3d need extending to take calculate_nummix and chi_num, intercept the fluxes to calculate $F_sq$ and accumulate the variance change associated with the fluxes directly.
This was more complicated... and prone to errors!

And horizontal diffusion was not implemented... but the logic should be the same as for the advection already in the code.
Definition of the additional fortran subroutine arguments were included in operators.F90 and noted in the procedure definition in advection.F90

Next is patching the operators.py to pass chi_num (with a default value of NoNe)
tracers.py only calls apply_3d_batch so the hack only needs to be included there. There we include the variable to apply the numerical calculations to (eg sim.salt). This is what is intercepted in the runscript (run2DsliceWithMixingestimates.py)

in _pygetm.pyx is the c interface to the fortran routines. Chi_num is included in both 2d and 3d advection definitions. 

Here in _pygetm.pyx i need to pass a NULL for chi_num as advection_uv_calculate always requires a chi_num argument!
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    def u_2d(self, Array u not None, double timestep, Array var not None, Array Ah=None, Array chi_num=None):
        cdef double * pAh = <double *>Ah.p if Ah is not None else NULL
        cdef double * pchi_num = <double *>chi_num.p if chi_num is not None else NULL
        advection_uv_calculate(1, 1, self.p, self.grid.p, self.ugrid.p, <double *>u.p, pAh, timestep, self.ph, self.pDU, <double *>var.p, pchi_nume)
And what about class AdvectionVertical? is that used by tracers?
No... Jorn did mention when this is used but I can't remember now  

cdef class VerticalAdvection:
    @cython.initializedcheck(False)
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    def w_3d(self, Array w not None, Array w_var not None, double timestep, Array var not None):
        advection_w_calculate(self.p, self.grid.p, <double *>w.p, <double *>w_var.p, timestep, self.ph, <double *>var.p)


---

### Implementation review (by chatGPT): Klingbeil et al. (2014) DVD diagnostic in pygetm

### Reference

Klingbeil, Mohammadi-Aragh, Graewe & Burchard (2014)

DOI: 10.1016/j.ocemod.2014.06.001

### 1. Objective of the paper

The paper derives a discrete numerical mixing diagnostic based on the decay of tracer variance caused by the advection scheme. The key quantity is

Paper definition

### χ_num = (f²_ideal − f²_after) / Δt

where f²_ideal is obtained by transporting f² with exactly the same discrete transport operator used for f.

### 2. Correspondence of variables

| Paper                       | Fortran                        |
| --------------------------- | ------------------------------ |
| Tracer f                    | f(i,j)or f(i,j,k)               |
| Cell thickness/volume h     | h(i,j)or haux(k)                |
| Transport Q                 | QU, QV, w                      |
| Face value f̄               | fu(after limiter)              |
| Donor value f_d             | f_donor(or fu before correction) |
| Second-moment flux Q f̄ f_d | flux_sq                        |
| Numerical mixing χ_num      | chi_num                        |

### 3. Horizontal u-advection

### Paper

The tracer update is

Tracer update

### (h f)ⁿ⁺¹ = (h f)ⁿ − Δt ∇·(Q f̄)

The ideal transport of f² is

Second-moment transport

### (h f²)_ideal = (h f²)ⁿ − Δt ∇·(Q f̄ f_d)

### Code

Lines 151–155

exact match

Second-moment face flux

flux_sq(i,j) = QU(i,j) * fu * f_donor

Lines 190–193

exact match

Divergence of second-moment flux

advn_sq = (flux_sq(i,j)-flux_sq(i-1,j))*iA(i,j)

f2_advected = (h_old*f_before² - dt*advn_sq) / h_new

Lines 198–199

exact match

Numerical mixing diagnostic

chi_num += max(0, (f2_advected - f_after²)/dt)

### Assessment

This is the discrete variance decay diagnostic of the paper.

### 4. Horizontal v-advection

The implementation is the direct y-direction analogue.

| Paper term | Code                        |
| ---------- | --------------------------- |
| Q f̄ f_d   | flux_sq = QV * fu * f_donor |
| ∂/∂y       | flux_sq(i,j)-flux_sq(i,j-1) |
| χ_num      | same expression as u2d      |

Agreement: exact directional analogue of the paper.

### 5. Vertical advection

### Paper

For vertical transport the same operator must be applied during each sub-step.

### Code

Lines 497–509

face flux

Donor value stored before TVD correction

f_donor = fu

Limited face value used for tracer flux

fu = fu + 0.5*limiter*(1-cfl)*deltaf

Second-moment flux

flux_sq(k) = w_var(i,j,k) * fu * f_donor

Lines 524–529

diagnostic

Numerical mixing accumulation

f2_advected = (haux*f_before² - dtk*advn_sq) / h_new

chi_num += max(0, (f2_advected - f_after²)/dtk)

### Assessment

This is consistent with the paper and correctly accounts for vertical sub-stepping.

### 6. Important implementation detail

The paper requires that the second-moment flux use:

Required form

### Q · (limited face value) · (donor value before limiting)

Your code does exactly that:

| Quantity                    | Code variable   |
| --------------------------- | --------------- |
| Limited face value          | fu              |
| Donor value before limiting | f_donor         |
| Transport                   | QU / QV / w_var |

This is the strongest indication that the implementation follows the derivation correctly.

### 7. One notable difference from the paper

### Horizontal diffusion is excluded from χ_num

Intentional

Lines 169–170 contain the comment:

diffusive contribution to flux_sq intentionally left out

Implication:

* chi_num diagnoses only numerical mixing from advection

* Physical horizontal diffusion Ah is not included in the variance budget

* This is acceptable if Ah = 0 (as your comment suggests)

If Ah is non-zero and you want total variance decay, this is a discrepancy with the full variance budget.

### 8. Minor numerical difference

### Non-negativity clipping

The paper derives χ_num ≥ 0 analytically.

Your implementation uses

### MAX(0, χ_num)

to remove small negative values caused by round-off.

This is a sensible and common practical modification.

### 9. Potential issue to verify

### Use of h_old in u2d/v2d

Check

In u2d:

h(i,j) = h(i,j) - dt*(...)

f2_advected = ((h(i,j)+dt*(...))*f_before² - dt*advn_sq)/h(i,j)

The expression reconstructs h_old from h_new + dt*div(Q), which is algebraically correct.

I checked the sign: it is consistent with the tracer update.

No inconsistency found.

### 10. Final verdict

### Core DVD formulation

Correct

The implementation matches the Klingbeil et al. (2014) discrete variance decay formulation.

### Face-flux construction

Correct

The second-moment flux uses Q · f_limited · f_donor, exactly as required.

### Volume normalization

Correct

The normalization by the updated cell thickness/volume is handled consistently.

### Vertical sub-stepping

Correct

The vertical implementation applies the diagnostic during each sub-step consistently.

### Only substantive difference

Documented

The diagnostic excludes horizontal diffusive variance dissipation from chi_num. If Ah = 0, this is effectively identical to the paper’s intended use for diagnosing numerical mixing by advection.

### Conclusion

I do not see any major inconsistency between the paper and your Fortran implementation. The code appears to be a faithful implementation of the DVD numerical mixing diagnostic, with the deliberate choice of omitting explicit horizontal diffusion from the diagnosed variance decay.



---

## 9. Key References

**Klingbeil, K., Mohammadi-Aragh, M., Gräwe, U., & Burchard, H. (2014).** Quantification of spurious dissipation and mixing — Discrete Variance Decay in a finite-volume framework. *Ocean Modelling*, 81, 49–64. https://doi.org/10.1016/j.ocemod.2014.06.001

> The foundational paper. Introduces the DVD framework, defines $\chi^{num}$ and $\chi^{phy}$ rigorously, and presents both the online Method II (used here) and an offline Method I. Demonstrates application in GETM.

**Burchard, H., & Rennau, H. (2008).** Comparative quantification of physically and numerically induced mixing in ocean models. *Ocean Modelling*, 20(3), 293–311. https://doi.org/10.1016/j.ocemod.2007.10.003

> Original proposal of the $s^2$ tracer approach for quantifying numerical mixing online. Precursor to Klingbeil et al. (2014).

**MacCready, P., Geyer, W. R., & Burchard, H. (2018).** Estuarine exchange flow is related to mixing through the salinity variance budget. *Journal of Physical Oceanography*, 48(6), 1251–1271. https://doi.org/10.1175/JPO-D-17-0266.1

> Links the variance budget to TEF, enabling comparison of $\mathcal{M}^{tot}$ from this diagnostic with the TEF-derived exchange flow.

**MacCready, P. (2011).** Calculating estuarine exchange flow using isohaline coordinates. *Journal of Physical Oceanography*, 41(8), 1116–1124. https://doi.org/10.1175/2011JPO4517.1

> The original TEF paper, providing the framework with which the mixing diagnostic connects.

**Lorenz, M., Klingbeil, K., MacCready, P., & Burchard, H. (2019).** Numerical issues of the Total Exchange Flow (TEF) analysis framework for quantifying estuarine circulation. *Ocean Science*, 15(2), 601–614. https://doi.org/10.5194/os-15-601-2019

> Discusses numerical artefacts in TEF analysis, directly relevant when using TEF alongside numerical mixing diagnostics.
