"""
Robust Linear Controller for Terra Robot

This implements a linear state-space controller as an alternative to the nonlinear MPC.
The controller is designed based on the linearized kinematic model:
    ẏ = v_ref * θ       (lateral error dynamics)
    θ̇ = ν * ω          (heading dynamics with skid coefficient)

The controller uses state feedback with integral action for zero steady-state error.
Controller gains can be synthesized offline via H∞/μ-synthesis methods.

The state-space controller form is:
    u[k] = Cc * x_c[k] + Dc * e[k]
    x_c[k+1] = Ac * x_c[k] + Bc * e[k]

Where e = [y_error, θ_error]^T are the tracking errors.
"""

import numpy as np
import time


class RobustController:
    def __init__(self, params):
        """
        Initialize the robust linear controller.
        
        params: dict with keys:
            - v_ref: reference linear velocity
            - nu: angular skid coefficient
            - mu: linear skid coefficient  
            - dt: time step
            - v_lin_max: max linear velocity
            - v_ang_max: max angular velocity
            - Kp_y: proportional gain for lateral error
            - Kp_theta: proportional gain for heading error
            - Ki_y: integral gain for lateral error (optional)
            - Ki_theta: integral gain for heading error (optional)
        """
        self.p = {}
        self.p['v_ref'] = params.get('v_ref', 0.6)
        self.p['nu'] = params.get('nu', 0.2)
        self.p['mu'] = params.get('mu', 1.0)
        self.p['dt'] = params.get('dt', 0.2)
        self.p['v_lin_max'] = params.get('v_lin_max', 1.0)
        self.p['v_ang_max'] = params.get('v_ang_max', 1.0)
        self.p['v_max'] = params.get('v_max', 1.0)
        self.p['Lbase'] = params.get('Lbase', 0.27)
        self.p['verbose'] = params.get('verbose', False)
        
        # Controller gains - can be synthesized via H∞/μ-synthesis offline
        # Default gains are designed for the linearized system
        self.p['Kp_y'] = params.get('Kp_y', 2.0)       # Lateral error proportional
        self.p['Kp_theta'] = params.get('Kp_theta', 1.5) # Heading error proportional
        self.p['Ki_y'] = params.get('Ki_y', 0.1)       # Lateral error integral
        self.p['Ki_theta'] = params.get('Ki_theta', 0.05) # Heading error integral
        
        # Option to use synthesized H∞/μ controller matrices
        self.p['use_synthesized_matrices'] = params.get('use_synthesized_matrices', False)
        self.p['controller_matrices_file'] = params.get('controller_matrices_file', None)
        
        # State-space controller matrices (can be loaded from external synthesis)
        self.matrices_source = 'PI'  # Track where matrices came from
        if self.p['use_synthesized_matrices']:
            loaded = self._load_synthesized_matrices()
            if loaded:
                self.matrices_source = 'H∞-synthesized'
            else:
                print("[RobustController] Synthesized matrices not found, using PI fallback")
                self._setup_controller_matrices()
        else:
            self._setup_controller_matrices()
        
        # Integral states - size depends on controller order
        n_states = self.Ac.shape[0] if hasattr(self, 'Ac') else 2
        self.x_c = np.zeros(n_states)
        
        # Diagnostic counters
        self.step_count = 0
        self.start_time = time.time()
        
        self.printParams()
        self.printPlantModel()
        self.printControllerMatrices()
    
    def _load_synthesized_matrices(self):
        """
        Load controller matrices from synthesized H∞/μ controller.
        
        Returns True if successful, False otherwise.
        """
        import os
        
        # Determine file path
        if self.p['controller_matrices_file']:
            filepath = self.p['controller_matrices_file']
        else:
            # Default location: same directory as this file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, 'robust_controller_matrices.npz')
        
        if not os.path.exists(filepath):
            print(f"[RobustController] Matrix file not found: {filepath}")
            return False
        
        try:
            data = np.load(filepath)
            self.Ac = data['Ac']
            self.Bc = data['Bc']
            self.Cc = data['Cc']
            self.Dc = data['Dc']
            
            # Validate dimensions
            if self.Ac.shape[0] != self.Ac.shape[1]:
                raise ValueError(f"Ac must be square, got {self.Ac.shape}")
            if self.Bc.shape[0] != self.Ac.shape[0]:
                raise ValueError(f"Bc rows must match Ac, got {self.Bc.shape}")
            
            print(f"[RobustController] Loaded synthesized matrices from: {filepath}")
            print(f"  Controller order: {self.Ac.shape[0]}")
            print(f"  Ac: {self.Ac.shape}, Bc: {self.Bc.shape}, Cc: {self.Cc.shape}, Dc: {self.Dc.shape}")
            
            return True
            
        except Exception as e:
            print(f"[RobustController] Error loading matrices: {e}")
            return False
    
    def _setup_controller_matrices(self):
        """
        Setup the state-space controller matrices.
        
        For a PI controller on each channel:
            ω = Kp_theta * θ_error + Ki_theta * ∫θ_error dt + Kp_y * y_error / v_ref
            v = v_ref (constant forward velocity)
        
        State-space form:
            x_c = [∫y_error, ∫θ_error]^T  (integral states)
            e = [y_error, θ_error]^T       (tracking errors)
        
            x_c[k+1] = Ac * x_c[k] + Bc * e[k]
            u = Cc * x_c + Dc * e
        """
        dt = self.p['dt']
        
        # Discrete-time integrator: x_c[k+1] = x_c[k] + dt * e[k]
        self.Ac = np.eye(2)
        self.Bc = dt * np.eye(2)
        
        # Output: u = [v, ω]^T
        # v = v_ref (we'll add this separately)
        # ω = Kp_theta * θ + Ki_theta * ∫θ + Kp_y * y / v_ref + Ki_y * ∫y / v_ref
        
        v_ref = max(self.p['v_ref'], 0.1)  # Avoid division by zero
        
        # Cc maps integral states to control
        # [v; ω] = Cc * [∫y; ∫θ]
        # Positive y_error/theta_error means path is to the LEFT -> need positive omega to turn left
        self.Cc = np.array([
            [0.0, 0.0],                                           # v: no integral action
            [self.p['Ki_y'] / v_ref, self.p['Ki_theta']]          # ω: integral terms (positive)
        ])
        
        # Dc maps errors to control
        # [v; ω] = Dc * [y; θ]
        self.Dc = np.array([
            [0.0, 0.0],                                           # v: constant v_ref
            [self.p['Kp_y'] / v_ref, self.p['Kp_theta']]          # ω: proportional terms (positive)
        ])
    
    def printPlantModel(self):
        """Print the linearized plant model for μ-synthesis."""
        v_ref = self.p['v_ref']
        nu = self.p['nu']
        mu = self.p['mu']
        
        print('\n' + '='*70)
        print('LINEARIZED PLANT MODEL (for μ-synthesis)')
        print('='*70)
        print(f'Reference velocity (v_ref): {v_ref} m/s')
        print(f'Angular skid coefficient (nu): {nu}')
        print(f'Linear skid coefficient (mu): {mu}')
        print(f'Sample time (dt): {self.p["dt"]} s')
        print()
        print('Continuous-time state-space (from [v, ω] to [y, θ]):')
        print('  ẏ = v_ref * θ')
        print('  θ̇ = nu * ω')
        print()
        print('Plant matrices A, B, C, D:')
        A_plant = np.array([[0, v_ref], [0, 0]])
        B_plant = np.array([[0, 0], [0, nu]])
        C_plant = np.eye(2)
        D_plant = np.zeros((2, 2))
        print(f'  A = {A_plant.tolist()}')
        print(f'  B = {B_plant.tolist()}')
        print(f'  C = {C_plant.tolist()}')
        print(f'  D = {D_plant.tolist()}')
        print()
        print('MATLAB code to create plant:')
        print(f'  v_ref = {v_ref}; nu = {nu};')
        print(f'  A = [0, {v_ref}; 0, 0];')
        print(f'  B = [0, 0; 0, {nu}];')
        print('  C = eye(2); D = zeros(2);')
        print('  G = ss(A, B, C, D);')
        print('='*70 + '\n')
    
    def printControllerMatrices(self):
        """Print current controller matrices for verification."""
        print('\n' + '='*70)
        print('CONTROLLER STATE-SPACE MATRICES (K)')
        print('='*70)
        print('Controller order: 2 (PI structure)')
        print()
        print('Discrete-time controller:')
        print('  x_c[k+1] = Ac * x_c[k] + Bc * e[k]')
        print('  u[k] = Cc * x_c[k] + Dc * e[k]')
        print()
        print(f'  Ac = {self.Ac.tolist()}')
        print(f'  Bc = {self.Bc.tolist()}')
        print(f'  Cc = {self.Cc.tolist()}')
        print(f'  Dc = {self.Dc.tolist()}')
        print()
        print('To replace with synthesized controller:')
        print('  1. Run μ-synthesis in MATLAB: [K,~,~] = dksyn(P,2,2);')
        print('  2. Discretize: Kd = c2d(K, dt, "tustin");')
        print('  3. Export: [Ac,Bc,Cc,Dc] = ssdata(Kd);')
        print('  4. Load into this controller')
        print('='*70 + '\n')
    
    def updateParams(self, params):
        """Update controller parameters dynamically."""
        for param in params:
            if param in self.p:
                if self.p[param] != params[param]:
                    print(f'robust updating parameter [{param}] from {self.p[param]} to {params[param]}')
                    self.p[param] = params[param]
        self._setup_controller_matrices()
        return True
    
    def printParams(self):
        print('\n' + '='*70)
        print('ROBUST CONTROLLER PARAMETERS')
        print('='*70)
        print(f'  v_ref: {self.p["v_ref"]} m/s')
        print(f'  nu (angular skid): {self.p["nu"]}')
        print(f'  mu (linear skid): {self.p["mu"]}')
        print(f'  dt: {self.p["dt"]} s')
        print(f'  v_lin_max: {self.p["v_lin_max"]} m/s')
        print(f'  v_ang_max: {self.p["v_ang_max"]} rad/s')
        print(f'  v_max (wheel): {self.p["v_max"]} m/s')
        print(f'  Lbase: {self.p["Lbase"]} m')
        print('Controller gains:')
        print(f'  Kp_y: {self.p["Kp_y"]}')
        print(f'  Kp_theta: {self.p["Kp_theta"]}')
        print(f'  Ki_y: {self.p["Ki_y"]}')
        print(f'  Ki_theta: {self.p["Ki_theta"]}')
        print('='*70 + '\n')
    
    def reset(self):
        """Reset integral states."""
        self.x_c = np.zeros(2)
        self.step_count = 0
        print('[RobustController] Reset integral states')
    
    def solve_mpc(self, mpc_reference, omega=0):
        """
        Compute control action using robust linear controller.
        
        This method has the same signature as MPC_CONTROLLER.solve_mpc()
        for drop-in compatibility.
        
        Args:
            mpc_reference: dict with keys 'x', 'y', 'theta', 'speed'
                - x[0], y[0], theta[0] are the reference at current time
            omega: current angular velocity (unused in this simple controller)
        
        Returns:
            u: control [v, ω] as 2xN array (only first column used)
            pred_vals: predicted states (simplified - just current reference)
            ss_error: steady-state error norm
        """
        self.step_count += 1
        
        # Extract tracking errors from reference
        # In robot frame, reference x[0], y[0] represent where we want to be
        # y_error = y[0] (lateral deviation from robot's x-axis)
        # theta_error = theta[0] (heading error)
        
        if len(mpc_reference['x']) == 0:
            # No reference, stop
            N = 15  # Default horizon
            u = np.zeros((2, N))
            pred_vals = np.zeros((3, N + 1))
            print(f'[Step {self.step_count}] No reference, stopping')
            return u, pred_vals, 0.0
        
        y_error = mpc_reference['y'][0]
        theta_error = mpc_reference['theta'][0]
        e = np.array([y_error, theta_error])
        
        # Save previous state for logging
        x_c_prev = self.x_c.copy()
        
        # State-space control law: u = Cc * x_c + Dc * e
        u_integral = self.Cc @ self.x_c  # Contribution from integral states
        u_proportional = self.Dc @ e      # Contribution from error (proportional)
        u_raw = u_integral + u_proportional
        
        # Add reference velocity
        v_cmd = self.p['v_ref'] + u_raw[0]
        omega_cmd = u_raw[1]
        
        # Store pre-saturation values for logging
        v_cmd_presat = v_cmd
        omega_cmd_presat = omega_cmd
        
        # Saturate angular velocity
        omega_cmd = np.clip(omega_cmd, -self.p['v_ang_max'], self.p['v_ang_max'])
        
        # Saturate linear velocity
        v_cmd = np.clip(v_cmd, -self.p['v_lin_max'], self.p['v_lin_max'])
        
        # Enforce motor speed constraints: v ± L/2 * ω ≤ v_max
        v_left = v_cmd + 0.5 * self.p['Lbase'] * omega_cmd
        v_right = v_cmd - 0.5 * self.p['Lbase'] * omega_cmd
        
        max_wheel = max(abs(v_left), abs(v_right))
        wheel_saturated = False
        if max_wheel > self.p['v_max']:
            scale = self.p['v_max'] / max_wheel
            v_cmd *= scale
            omega_cmd *= scale
            wheel_saturated = True
        
        # Update integral states: x_c[k+1] = Ac * x_c + Bc * e
        self.x_c = self.Ac @ self.x_c + self.Bc @ e
        
        # Anti-windup: clamp integral states
        windup_clamped = False
        x_c_unclamped = self.x_c.copy()
        self.x_c = np.clip(self.x_c, -10.0, 10.0)
        if not np.allclose(self.x_c, x_c_unclamped):
            windup_clamped = True
        
        # Detailed logging every step (or periodically)
        if self.p['verbose'] or self.step_count <= 10 or self.step_count % 20 == 0:
            print(f'\n[RobustController Step {self.step_count}]')
            print(f'  Errors:      y_err={y_error:+.4f} m, theta_err={theta_error:+.4f} rad ({np.degrees(theta_error):+.2f} deg)')
            print(f'  Integral:    ∫y={x_c_prev[0]:+.4f}, ∫θ={x_c_prev[1]:+.4f}')
            print(f'  u_prop:      [{u_proportional[0]:+.4f}, {u_proportional[1]:+.4f}] (Dc @ e)')
            print(f'  u_int:       [{u_integral[0]:+.4f}, {u_integral[1]:+.4f}] (Cc @ x_c)')
            print(f'  u_raw:       v={v_cmd_presat:+.4f}, ω={omega_cmd_presat:+.4f}')
            print(f'  u_final:     v={v_cmd:+.4f} m/s, ω={omega_cmd:+.4f} rad/s')
            print(f'  Wheel v:     L={v_left:+.4f}, R={v_right:+.4f}')
            print(f'  Saturation:  wheel={wheel_saturated}, windup={windup_clamped}')
            print(f'  New ∫:       [∫y={self.x_c[0]:+.4f}, ∫θ={self.x_c[1]:+.4f}]')
        
        # Format output to match MPC interface
        N = len(mpc_reference['x']) - 1 if len(mpc_reference['x']) > 1 else 15
        u = np.zeros((2, N))
        u[0, :] = v_cmd  # Same control for whole horizon (linear controller)
        u[1, :] = omega_cmd
        
        # Simplified predicted vals (just propagate with constant control)
        n_states = 3
        pred_vals = np.zeros((n_states, N + 1))
        
        ss_error = np.sqrt(mpc_reference['x'][0]**2 + mpc_reference['y'][0]**2)
        
        return u, pred_vals, ss_error

