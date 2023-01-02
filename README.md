# omega_gw-constraints

## Cosmic Microwave Background
The Cosmic Microwave Background (CMB) is the radiation remnant from the Big Bang. After the big bang, the early Universe was denser and much hotter. The tightly coupled system of photons, electrons, and protons behaved as a "single gas" (matter), and it trapped radiation (light).

As the Universe expanded and cooled, matter lost its ability to ensnare light. Around 380,000 years after the big bang— the temperature of the gas decreased enough for the protons to capture the electrons and become atoms. This phase is called recombination.

After recombination, the photons (light) were no longer scattered by collisions with charged particles and are free to travel largely unimpeded through space. After 14 billion years, this release of photons became the CMB we see today.

Small disturbances in the early Universe made the "single gas" propagate as sound waves. The gas from the rarefied (cooling the gas) and compressed (heating the gas) regions froze into the CMB and created temperature anisotropies. 


## Cosmological Perturbation
According to the Inflation theory, the Universe expanded exponentially from a microscopic scale in its first few moments after the Big Bang. The quantum fluctuations accelerated by this rapid expansion has created some initial cosmological perturbations, which has two component, tensor (Gravitational Waves) and scalar (Density Variations).

## Primordial Gravitational Waves
Gravitational waves are waves of the intensity of gravity generated by the accelerated masses that propagate as waves outward from their source at the speed of light. The Primordial Gravitational Waves (PGWs) created after inflation contributed to the tensor component of perturbations and the CMB Anisotropy

## Tensor-to-Scalar Ratio
The tensor-to-scaler ratio, r, was proposed to measure the proportion of the two components of fluctuations. It could also measure the Universe's expansion rate during inflation since the tensor component of perturbations only depends on the expansion rate.

## B Mode Power Spectrum
There are two-component in the CMB Anisotropies, temperature and polarization. And the polarization has an E mode and a B mode. While the EE, TE (Cross section of the Temperature and E-mode polarization), and BB power spectra (ClBB) contain the primordial tensor fluctuation signals, only the BB power spectrum does not contain any primordial scalar fluctuation signals. Therefore, the BB power spectrum could be used as a signature to the tensor perturbations that resulted from the PGWs.

## Analysing Method
We forecast constraints achievable with the CMB-S4 experiment [1].  CMB-S4 is a cosmic microwave background experiment that will probe gravitational waves through the B-mode signature in CMB polarization that uniquely resulted from the Primordial Gravitational Wave.

Forecasts typically focus on scale-invariant power spectrum for the inflationary gravitational waves (tensor), parameterized by r, the tensor-to-scalar ratio. However, gravitational waves can be produced by other sources, such as phase transitions, and are characterized by a scale-dependent power spectrum, Pt(k). 

The tensor primordial power spectrum - Pt(k) is parametrized as: 

$$P_{t}(k) = A_{t}\left(\frac{k}{k_p}\right)^{n_t}, \, r = \frac{A_t}{A_s}$$

where $A_s$ is the scalar amplitude and $A_t$ is the tensor amplitude.

For the inflationary case: $n_t \approx 0$.

We created our initial power spectrum as equally-binned step functions of the Fourier wavenumber (k) and its corresponding scale-dependent Pt(k) parametrized by r and the well-determined As value (from the CMB experiment) on different logarithmic scales (from 1e-6 to 10).

Below is the equation of how we vary the scale-dependent Pt(k):

$$P_{t}(k) = erfc\left(\frac{(i-1)\times\delta - \ln(k)}{\sigma\times\sqrt{2}}\right) - erfc\left(\frac{i\times\delta - \ln(k)}{\sigma\times\sqrt{2}}\right)$$

## Forecasting Method
We used the public Boltzmann code CLASS to transform our initial general power spectrum (scale-dependent) to the CMB B-mode power spectrum. We varied the r value for the bins on each scale and generated the corresponding B-mode polarization power spectrum.
We used a standard Fisher matrix method in cosmology [2] with the transformed B-mode power spectrum to forecast a constraint for r and Pt(k) for PGWs. We also considered the sky coverage, galactic foreground noise, and instrumentation noise.
Fisher Matrix Equation: 

$$F_{r} = \sum_{l=2}^{l_{max}}\left[C_l^{BB} + \frac{4\pi\sigma^2}{N}e^{\theta^2_{b}l(l+1)}\right]^{-2}\left(\frac{\partial C_l}{\partial r}\right)^2$$

The $\frac{4\pi\sigma^2}{N}e^{\theta^2_{b}l(l+1)}$ part is the foreground signal and noise simulated. After getting the Fiducial value for r in each bin, we can just multiply it with the determined $A_s$ value to get the constraint for the Primordial Power Spectra in each bin.

## Delensing Process

The E mode polarization, which also contains the scalar signals, could be transformed into the B mode power through gravitational lensing. This means the Inflationary Gravitational Waves signals (in the low multipole region) and the signals from Gravitational Lensing (in the high multipole region) are intertwined at some level in the low multipole region. Therefore, it is essential to do a delensing process to improve the accuracy of the B-mode polarization data.

The equation for the power spectrum that includes lensing effects is:

$$C_l = C_l^{r} + A_l \times C_l^{lensed(A_l)} = r \times \frac{\partial{C_l}}{\partial{r}} + A_l \times \frac{\partial{C_l}}{\partial{A_l}}$$ 


# Using the code

## constraints_tools.py
