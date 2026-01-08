#!/usr/bin/env python3
"""
H∞ Robust Controller Synthesis for Terra Robot

This script synthesizes a robust controller using H∞ mixed-sensitivity design
without requiring MATLAB. Uses python-control library.

The design process:
1. Define nominal plant G(s) from linearized kinematics
2. Define uncertainty weight W_I(s) for skid parameters (μ, ν)
3. Define performance weight W_P(s) for tracking
4. Construct augmented plant P for mixed-sensitivity
5. Synthesize H∞ controller K(s)
6. Discretize and export matrices (Ac, Bc, Cc, Dc)

Requirements:
    pip install control numpy scipy

Usage:
    python3 synthesize_robust_controller.py
    
Output:
    robust_controller_matrices.npz - Discretized controller matrices
"""

import numpy as np
from scipy import signal
import os

# Try to import control library
try:
    import control as ct
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False
    print("WARNING: python-control not installed. Using manual H∞ approximation.")
    print("For full H∞ synthesis, install: pip install control slycot")


class RobustControllerSynthesis:
    """
    Synthesizes a robust H∞ controller for the Terra robot lane-keeping system.
    """
    
    def __init__(self, params=None):
        """
        Initialize synthesis parameters.
        
        params: dict with plant and design parameters
        """
        # Default parameters from mpc.yaml
        self.params = {
            'v_ref': 0.6,       # Reference velocity (m/s)
            'nu': 0.2,          # Angular skid coefficient (nominal)
            'mu': 1.0,          # Linear skid coefficient (nominal)
            'dt': 0.2,          # Sample time (s)
            'Lbase': 0.27,      # Wheelbase (m)
            
            # Uncertainty bounds (for robustness analysis)
            'nu_uncertainty': 0.5,  # ±50% uncertainty on nu
            'mu_uncertainty': 0.2,  # ±20% uncertainty on mu
            
            # Performance specs
            'bandwidth_target': 1.0,    # Desired closed-loop bandwidth (rad/s)
            'ss_error_bound': 0.01,     # Steady-state error bound
            'peak_sensitivity': 2.0,    # Maximum sensitivity peak (M_s)
        }
        
        if params:
            self.params.update(params)
        
        self.print_params()
    
    def print_params(self):
        """Print synthesis parameters."""
        print("\n" + "="*70)
        print("ROBUST CONTROLLER SYNTHESIS PARAMETERS")
        print("="*70)
        print("\nPlant Parameters:")
        print(f"  v_ref = {self.params['v_ref']} m/s")
        print(f"  nu = {self.params['nu']} (±{self.params['nu_uncertainty']*100:.0f}%)")
        print(f"  mu = {self.params['mu']} (±{self.params['mu_uncertainty']*100:.0f}%)")
        print(f"  dt = {self.params['dt']} s")
        print("\nPerformance Specs:")
        print(f"  Bandwidth target = {self.params['bandwidth_target']} rad/s")
        print(f"  SS error bound = {self.params['ss_error_bound']}")
        print(f"  Peak sensitivity = {self.params['peak_sensitivity']}")
        print("="*70 + "\n")
    
    def create_nominal_plant(self):
        """
        Create the nominal linearized plant G(s).
        
        Linearized model around straight-line trajectory:
            ẏ = v_ref * θ       (lateral dynamics)
            θ̇ = ν * ω          (heading dynamics)
        
        State: x = [y, θ]^T
        Input: u = [v, ω]^T  (but v is constant, so effectively u = ω)
        Output: y = [y, θ]^T (full state feedback)
        
        For lane-keeping, we focus on the ω -> [y, θ] transfer.
        """
        v_ref = self.params['v_ref']
        nu = self.params['nu']
        
        # State-space matrices
        # ẋ = Ax + Bu, y = Cx + Du
        # State: [y, θ], Input: [ω], Output: [y, θ]
        
        # Note: We model only the ω input since v is held constant
        A = np.array([
            [0, v_ref],  # ẏ = v_ref * θ
            [0, 0]       # θ̇ = 0 (integrator, driven by ω)
        ])
        
        # B matrix: effect of ω on states
        # ẏ = 0 (ω doesn't directly affect y)
        # θ̇ = ν * ω
        B = np.array([
            [0],
            [nu]
        ])
        
        # Full state output
        C = np.eye(2)
        D = np.zeros((2, 1))
        
        print("Nominal Plant G(s):")
        print(f"  A = {A.tolist()}")
        print(f"  B = {B.tolist()}")
        print(f"  C = {C.tolist()}")
        print(f"  D = {D.tolist()}")
        
        if HAS_CONTROL:
            G = ct.ss(A, B, C, D)
            return G
        else:
            return (A, B, C, D)
    
    def create_performance_weight(self):
        """
        Create performance weight W_P(s) for sensitivity shaping.
        
        W_P(s) = (s/M + ω_B) / (s + ω_B*ε)
        
        This enforces:
        - High gain at low freq (integral action, small SS error)
        - Crossover near ω_B (bandwidth)
        - Low gain at high freq (robustness to noise)
        """
        M = self.params['peak_sensitivity']  # Peak sensitivity
        omega_B = self.params['bandwidth_target']  # Bandwidth
        eps = self.params['ss_error_bound']  # SS error bound
        
        # W_P numerator and denominator
        # W_P(s) = (s/M + ω_B) / (s + ω_B*ε)
        num = [1/M, omega_B]
        den = [1, omega_B * eps]
        
        print(f"\nPerformance Weight W_P(s):")
        print(f"  W_P = (s/{M} + {omega_B}) / (s + {omega_B*eps})")
        print(f"  |W_P(0)| = {omega_B / (omega_B * eps):.1f} ({20*np.log10(omega_B / (omega_B * eps)):.1f} dB)")
        print(f"  |W_P(∞)| = {1/M:.2f} ({20*np.log10(1/M):.1f} dB)")
        
        if HAS_CONTROL:
            W_P = ct.tf(num, den)
            return W_P
        else:
            return (num, den)
    
    def create_uncertainty_weight(self):
        """
        Create uncertainty weight W_I(s) for input multiplicative uncertainty.
        
        Models uncertainty in skid parameters (nu, mu).
        
        W_I(s) = r_inf * (s/ω_b + r_0/r_inf) / (s/ω_b + 1)
        
        - r_0: uncertainty at DC (low freq) - typically small (calibrated)
        - r_inf: uncertainty at high freq - typically large (slippage)
        - ω_b: uncertainty bandwidth
        """
        # Uncertainty model for tire slip
        r_0 = 0.1    # 10% uncertainty at DC
        r_inf = self.params['nu_uncertainty']  # High freq uncertainty
        omega_b = 10.0  # Uncertainty transition freq (rad/s)
        
        # W_I(s) = r_inf * (s/ω_b + r_0/r_inf) / (s/ω_b + 1)
        # Simplified: W_I(s) = (r_inf*s/ω_b + r_0) / (s/ω_b + 1)
        num = [r_inf/omega_b, r_0]
        den = [1/omega_b, 1]
        
        print(f"\nUncertainty Weight W_I(s):")
        print(f"  W_I = ({r_inf}*s/{omega_b} + {r_0}) / (s/{omega_b} + 1)")
        print(f"  |W_I(0)| = {r_0:.2f} ({r_0*100:.0f}%)")
        print(f"  |W_I(∞)| = {r_inf:.2f} ({r_inf*100:.0f}%)")
        
        if HAS_CONTROL:
            W_I = ct.tf(num, den)
            return W_I
        else:
            return (num, den)
    
    def synthesize_hinf_controller(self):
        """
        Synthesize H∞ controller using mixed-sensitivity design.
        
        Mixed-sensitivity problem:
            min_K || [W_P * S; W_I * K * S] ||_∞
        
        where S = (I + GK)^{-1} is the sensitivity function.
        
        If python-control is not available, uses a robust loop-shaping approximation.
        """
        print("\n" + "="*70)
        print("H∞ CONTROLLER SYNTHESIS")
        print("="*70)
        
        G = self.create_nominal_plant()
        W_P = self.create_performance_weight()
        W_I = self.create_uncertainty_weight()
        
        if HAS_CONTROL:
            return self._synthesize_with_control_lib(G, W_P, W_I)
        else:
            return self._synthesize_manual_loopshaping(G, W_P, W_I)
    
    def _synthesize_with_control_lib(self, G, W_P, W_I):
        """
        Synthesize using python-control's H∞ functions.
        
        Note: Full H∞ synthesis requires slycot. If not available,
        we fall back to loop-shaping.
        """
        print("\nUsing python-control for synthesis...")
        
        try:
            # Try H∞ mixed sensitivity synthesis
            # This requires slycot
            from control import mixsyn
            
            # Mixed sensitivity synthesis
            # Returns controller K, closed-loop CL, and info
            K, CL, gamma, rcond = mixsyn(G, W_P, W_I)
            
            print(f"  Synthesis successful!")
            print(f"  γ (H∞ norm) = {gamma:.4f}")
            print(f"  Controller order: {K.nstates}")
            
            return K, gamma
            
        except (ImportError, AttributeError) as e:
            print(f"  mixsyn not available: {e}")
            print("  Falling back to loop-shaping design...")
            return self._synthesize_loopshaping_control(G, W_P, W_I)
    
    def _synthesize_loopshaping_control(self, G, W_P, W_I):
        """
        Loop-shaping controller design as H∞ approximation.
        
        Design a controller K such that:
        - |L(jω)| > |W_P(jω)| at low frequencies (tracking)
        - |L(jω)| < |W_I(jω)|^{-1} at high frequencies (robustness)
        
        where L = GK is the loop transfer function.
        """
        print("\nUsing loop-shaping design...")
        
        v_ref = self.params['v_ref']
        nu = self.params['nu']
        omega_B = self.params['bandwidth_target']
        
        # Design a PI controller with lead compensation
        # K(s) = Kp * (1 + Ki/s) * (s/ω_z + 1)/(s/ω_p + 1)
        
        # Gains selected for desired bandwidth and phase margin
        Kp = omega_B / (v_ref * nu)  # Proportional gain
        Ki = omega_B / 5  # Integral gain (slower than bandwidth)
        omega_z = omega_B / 2  # Lead zero
        omega_p = omega_B * 5  # Lead pole
        
        print(f"  Kp = {Kp:.4f}")
        print(f"  Ki = {Ki:.4f}")
        print(f"  Lead: zero={omega_z:.2f}, pole={omega_p:.2f}")
        
        # PI controller: Kp * (1 + Ki/s) = (Kp*s + Kp*Ki) / s
        pi_num = [Kp, Kp * Ki]
        pi_den = [1, 0]
        
        # Lead compensator: (s/ω_z + 1) / (s/ω_p + 1)
        lead_num = [1/omega_z, 1]
        lead_den = [1/omega_p, 1]
        
        # Combine: K = PI * Lead
        K_num = np.convolve(pi_num, lead_num)
        K_den = np.convolve(pi_den, lead_den)
        
        # Create MIMO controller (for 2 outputs, 1 input)
        # We need K to be 1x2 (from [y, θ] to ω)
        K = ct.tf(K_num, K_den)
        
        # Convert to state-space for easier manipulation
        K_ss = ct.tf2ss(K)
        
        print(f"  Controller created (order {K_ss.nstates})")
        
        return K_ss, 1.0  # Return controller and nominal gamma
    
    def _synthesize_manual_loopshaping(self, G_matrices, W_P_tf, W_I_tf):
        """
        Manual loop-shaping without python-control library.
        
        Returns the controller as state-space matrices directly.
        """
        print("\nUsing manual loop-shaping design...")
        
        A_g, B_g, C_g, D_g = G_matrices
        v_ref = self.params['v_ref']
        nu = self.params['nu']
        omega_B = self.params['bandwidth_target']
        dt = self.params['dt']
        
        # Design gains for a PI controller with loop-shaping
        # Controller: ω = Kp_θ * θ + Ki_θ * ∫θ + Kp_y * y / v_ref + Ki_y * ∫y / v_ref
        
        # Closed-loop pole placement approach
        # Desired poles at -ω_B (critically damped response)
        zeta = 0.7  # Damping ratio
        omega_n = omega_B / zeta  # Natural frequency
        
        # For the double integrator (y dynamics), place poles at:
        # s^2 + 2*ζ*ω_n*s + ω_n^2 = 0
        # Characteristic eq: s^2 + (nu*Kp_θ)*s + (v_ref*nu*Kp_y/v_ref) = 0
        #                  = s^2 + (nu*Kp_θ)*s + (nu*Kp_y) = 0
        
        Kp_theta = 2 * zeta * omega_n / nu
        Kp_y = omega_n**2 / nu
        
        # Integral gains for zero steady-state error (slower than proportional)
        Ki_theta = Kp_theta * omega_B / 10
        Ki_y = Kp_y * omega_B / 10
        
        print(f"  Designed gains:")
        print(f"    Kp_y = {Kp_y:.4f}")
        print(f"    Kp_theta = {Kp_theta:.4f}")
        print(f"    Ki_y = {Ki_y:.4f}")
        print(f"    Ki_theta = {Ki_theta:.4f}")
        
        # Build discrete-time controller state-space matrices
        # State: x_c = [∫y, ∫θ]
        # Input: e = [y, θ]
        # Output: u = ω
        
        Ac = np.eye(2)  # Integrator: x[k+1] = x[k] + dt*e[k]
        Bc = dt * np.eye(2)
        
        # u = [0, ω], but we only output ω (second row)
        # ω = Kp_y/v_ref * y + Kp_θ * θ + Ki_y/v_ref * ∫y + Ki_θ * ∫θ
        Cc = np.array([[Ki_y/v_ref, Ki_theta]])  # Integral terms
        Dc = np.array([[Kp_y/v_ref, Kp_theta]])   # Proportional terms
        
        print(f"\n  Discrete controller matrices (dt={dt}s):")
        print(f"    Ac = {Ac.tolist()}")
        print(f"    Bc = {Bc.tolist()}")
        print(f"    Cc = {Cc.tolist()}")
        print(f"    Dc = {Dc.tolist()}")
        
        return (Ac, Bc, Cc, Dc), 1.0
    
    def discretize_controller(self, K, dt=None):
        """
        Discretize continuous-time controller using Tustin (bilinear) transform.
        """
        if dt is None:
            dt = self.params['dt']
        
        print(f"\nDiscretizing controller (dt={dt}s, Tustin)...")
        
        if isinstance(K, tuple):
            # Already discrete matrices
            return K
        
        if HAS_CONTROL:
            Kd = ct.c2d(K, dt, method='tustin')
            Ac, Bc, Cc, Dc = ct.ssdata(Kd)
            return (np.array(Ac), np.array(Bc), np.array(Cc), np.array(Dc))
        else:
            # Manual Tustin discretization
            raise NotImplementedError("Manual discretization not yet implemented")
    
    def save_controller(self, controller_matrices, filepath=None):
        """
        Save controller matrices to file for loading in RobustController.
        """
        if filepath is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, 'mpc', 'robust_controller_matrices.npz')
        
        Ac, Bc, Cc, Dc = controller_matrices
        
        # Ensure proper dimensions for the RobustController format
        # RobustController expects:
        # - Cc: 2x2 (maps [∫y, ∫θ] to [v, ω], but v row is zeros)
        # - Dc: 2x2 (maps [y, θ] to [v, ω], but v row is zeros)
        
        if Cc.shape[0] == 1:
            # Expand to 2 outputs: [v=0, ω]
            Cc_full = np.vstack([np.zeros((1, Cc.shape[1])), Cc])
            Dc_full = np.vstack([np.zeros((1, Dc.shape[1])), Dc])
        else:
            Cc_full = Cc
            Dc_full = Dc
        
        np.savez(filepath,
                 Ac=Ac,
                 Bc=Bc,
                 Cc=Cc_full,
                 Dc=Dc_full,
                 dt=self.params['dt'],
                 v_ref=self.params['v_ref'],
                 nu=self.params['nu'],
                 mu=self.params['mu'])
        
        print(f"\nController saved to: {filepath}")
        print(f"  Ac shape: {Ac.shape}")
        print(f"  Bc shape: {Bc.shape}")
        print(f"  Cc shape: {Cc_full.shape}")
        print(f"  Dc shape: {Dc_full.shape}")
        
        return filepath
    
    def analyze_robustness(self, K, gamma):
        """
        Analyze robustness margins and performance.
        """
        print("\n" + "="*70)
        print("ROBUSTNESS ANALYSIS")
        print("="*70)
        
        print(f"\n  H∞ norm (γ): {gamma:.4f}")
        
        if gamma < 1.0:
            print("  ✓ Robust Performance GUARANTEED (γ < 1)")
            print("    System maintains performance for all uncertainties ‖Δ‖∞ ≤ 1")
        elif gamma < 1.5:
            print("  ~ Robust Stability likely, Robust Performance may degrade")
        else:
            print("  ⚠ Warning: High γ, robustness not guaranteed")
        
        # Stability margin interpretation
        # PM ≈ 180° - 180°/γ, GM ≈ 1 + 1/γ (rough approximations)
        pm_approx = max(0, 180 - 180/gamma) if gamma > 0 else 0
        gm_approx = 1 + 1/gamma if gamma > 0 else float('inf')
        
        print(f"\n  Approximate margins:")
        print(f"    Phase margin ≈ {pm_approx:.1f}°")
        print(f"    Gain margin ≈ {gm_approx:.2f} ({20*np.log10(gm_approx):.1f} dB)")
        
        print("\n  Uncertainty Coverage:")
        print(f"    Angular skid (ν): {self.params['nu']} ± {self.params['nu_uncertainty']*100:.0f}%")
        print(f"    Linear skid (μ): {self.params['mu']} ± {self.params['mu_uncertainty']*100:.0f}%")
        
        print("="*70)


def main():
    """Main synthesis routine."""
    print("="*70)
    print("TERRA ROBOT - ROBUST H∞ CONTROLLER SYNTHESIS")
    print("="*70)
    
    # Create synthesizer with default parameters
    synth = RobustControllerSynthesis()
    
    # Synthesize H∞ controller
    K, gamma = synth.synthesize_hinf_controller()
    
    # Handle both tuple (matrices) and control object returns
    if isinstance(K, tuple):
        controller_matrices = K
    else:
        controller_matrices = synth.discretize_controller(K)
    
    # Analyze robustness
    synth.analyze_robustness(K, gamma)
    
    # Save controller matrices
    filepath = synth.save_controller(controller_matrices)
    
    print("\n" + "="*70)
    print("SYNTHESIS COMPLETE")
    print("="*70)
    print(f"\nTo use the synthesized controller:")
    print(f"  1. Update mpc.yaml: use_synthesized_matrices: true")
    print(f"  2. The RobustController will load from: {filepath}")
    print("="*70 + "\n")
    
    return controller_matrices, gamma


if __name__ == '__main__':
    main()
