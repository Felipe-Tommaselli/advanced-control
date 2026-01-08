#!/usr/bin/env python3
"""
Offline Analysis Script for Robust Control Data

Processes CSV logs from control_data_logger.py and computes metrics for μ-synthesis.

Usage:
    python3 analyze_control_data.py <csv_file>
    python3 analyze_control_data.py  # Uses latest file in ~/.ros/control_logs/

Outputs:
    - Summary statistics (console)
    - Performance metrics for report
    - Plots (if matplotlib available)
"""

import os
import sys
import glob
import math
from pathlib import Path

import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping plots")


def load_csv(filepath):
    """Load CSV data into numpy arrays."""
    data = np.genfromtxt(filepath, delimiter=',', names=True, dtype=None, encoding='utf-8')
    return data


def compute_metrics(data):
    """Compute performance metrics from logged data."""
    metrics = {}
    
    # Time vector
    time = data['time_s'].astype(float)
    dt = np.mean(np.diff(time))
    
    # Extract signals
    heading_err = data['heading_err_rad'].astype(float)
    distance_err = data['distance_err_m'].astype(float)
    cmd_lin = data['cmd_lin_mps'].astype(float)
    cmd_ang = data['cmd_ang_radps'].astype(float)
    vel_lin = data['vel_lin_mps'].astype(float)
    vel_ang = data['vel_ang_radps'].astype(float)
    tracking_err = data['tracking_err_m'].astype(float)
    
    # ===== BASIC STATISTICS =====
    metrics['duration_s'] = time[-1] - time[0]
    metrics['sample_count'] = len(time)
    metrics['sample_rate_hz'] = 1.0 / dt if dt > 0 else 0
    
    # ===== STEADY-STATE ANALYSIS =====
    # Use last 50% of data for steady-state
    ss_start = len(time) // 2
    
    metrics['ss_heading_mean'] = np.mean(heading_err[ss_start:])
    metrics['ss_heading_std'] = np.std(heading_err[ss_start:])
    metrics['ss_heading_max'] = np.max(np.abs(heading_err[ss_start:]))
    
    metrics['ss_distance_mean'] = np.mean(distance_err[ss_start:])
    metrics['ss_distance_std'] = np.std(distance_err[ss_start:])
    metrics['ss_distance_max'] = np.max(np.abs(distance_err[ss_start:]))
    
    metrics['ss_tracking_rms'] = np.sqrt(np.mean(tracking_err[ss_start:]**2))
    
    # ===== TRANSIENT ANALYSIS =====
    # Find settling time (when error stays below 5% of initial)
    if len(tracking_err) > 0 and tracking_err[0] > 0.001:
        threshold = 0.05 * tracking_err[0]
        threshold = max(threshold, 0.005)  # Minimum threshold 5mm
        
        settled = np.abs(tracking_err) < threshold
        if np.any(settled):
            # Find first index where it stays settled
            for i in range(len(settled)):
                if np.all(settled[i:min(i+10, len(settled))]):
                    metrics['settling_time_s'] = time[i]
                    break
            else:
                metrics['settling_time_s'] = float('nan')
        else:
            metrics['settling_time_s'] = float('nan')
    else:
        metrics['settling_time_s'] = 0.0
    
    # Peak error
    metrics['peak_heading_err'] = np.max(np.abs(heading_err))
    metrics['peak_distance_err'] = np.max(np.abs(distance_err))
    metrics['peak_tracking_err'] = np.max(tracking_err)
    
    # ===== CONTROL EFFORT =====
    metrics['cmd_ang_rms'] = np.sqrt(np.mean(cmd_ang**2))
    metrics['cmd_ang_max'] = np.max(np.abs(cmd_ang))
    metrics['cmd_lin_mean'] = np.mean(cmd_lin)
    
    # Control rate (derivative of command)
    cmd_ang_rate = np.diff(cmd_ang) / dt
    metrics['cmd_ang_rate_rms'] = np.sqrt(np.mean(cmd_ang_rate**2))
    
    # ===== VELOCITY TRACKING =====
    metrics['vel_lin_mean'] = np.mean(vel_lin)
    metrics['vel_lin_std'] = np.std(vel_lin)
    
    # ===== FOR μ-SYNTHESIS =====
    # Estimate closed-loop bandwidth from step response
    # (rough approximation: bandwidth ≈ 0.35 / rise_time)
    if len(tracking_err) > 0 and tracking_err[0] > 0.001:
        initial_err = tracking_err[0]
        # Find 10% to 90% rise time (or fall time for error)
        try:
            idx_90 = np.where(tracking_err < 0.1 * initial_err)[0]
            idx_10 = np.where(tracking_err < 0.9 * initial_err)[0]
            if len(idx_10) > 0 and len(idx_90) > 0:
                rise_time = time[idx_90[0]] - time[idx_10[0]]
                if rise_time > 0:
                    metrics['est_bandwidth_hz'] = 0.35 / rise_time
                else:
                    metrics['est_bandwidth_hz'] = float('nan')
            else:
                metrics['est_bandwidth_hz'] = float('nan')
        except:
            metrics['est_bandwidth_hz'] = float('nan')
    else:
        metrics['est_bandwidth_hz'] = float('nan')
    
    return metrics, time, heading_err, distance_err, cmd_ang, tracking_err


def print_metrics(metrics):
    """Print formatted metrics summary."""
    print("\n" + "="*70)
    print("ROBUST CONTROL PERFORMANCE ANALYSIS")
    print("="*70)
    
    print("\n--- Data Summary ---")
    print(f"  Duration:        {metrics['duration_s']:.2f} s")
    print(f"  Samples:         {metrics['sample_count']}")
    print(f"  Sample Rate:     {metrics['sample_rate_hz']:.1f} Hz")
    
    print("\n--- Steady-State Performance ---")
    print(f"  Heading Error:   {metrics['ss_heading_mean']*1000:.3f} ± {metrics['ss_heading_std']*1000:.3f} mrad")
    print(f"  Distance Error:  {metrics['ss_distance_mean']*1000:.3f} ± {metrics['ss_distance_std']*1000:.3f} mm")
    print(f"  Tracking RMS:    {metrics['ss_tracking_rms']*1000:.3f} mm")
    
    print("\n--- Transient Performance ---")
    print(f"  Settling Time:   {metrics['settling_time_s']:.2f} s")
    print(f"  Peak Heading:    {metrics['peak_heading_err']*1000:.2f} mrad ({np.degrees(metrics['peak_heading_err']):.2f}°)")
    print(f"  Peak Distance:   {metrics['peak_distance_err']*1000:.2f} mm")
    
    print("\n--- Control Effort ---")
    print(f"  Cmd ω RMS:       {metrics['cmd_ang_rms']:.4f} rad/s")
    print(f"  Cmd ω Max:       {metrics['cmd_ang_max']:.4f} rad/s")
    print(f"  Cmd v Mean:      {metrics['cmd_lin_mean']:.3f} m/s")
    print(f"  Cmd ω Rate RMS:  {metrics['cmd_ang_rate_rms']:.4f} rad/s²")
    
    print("\n--- For μ-Synthesis ---")
    print(f"  Est. Bandwidth:  {metrics.get('est_bandwidth_hz', float('nan')):.2f} Hz")
    print(f"  Velocity Mean:   {metrics['vel_lin_mean']:.3f} m/s")
    
    print("="*70)


def plot_results(time, heading_err, distance_err, cmd_ang, tracking_err, output_dir):
    """Generate analysis plots."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Error plot
    ax1 = axes[0]
    ax1.plot(time, np.degrees(heading_err), 'b-', label='Heading (°)', alpha=0.8)
    ax1.plot(time, distance_err * 100, 'r-', label='Distance (cm)', alpha=0.8)
    ax1.set_ylabel('Error')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Tracking Errors')
    
    # Combined tracking error
    ax2 = axes[1]
    ax2.plot(time, tracking_err * 1000, 'g-', linewidth=1.5)
    ax2.set_ylabel('Tracking Error (mm)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Combined Tracking Error')
    ax2.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5mm threshold')
    ax2.legend(loc='upper right')
    
    # Control command
    ax3 = axes[2]
    ax3.plot(time, cmd_ang, 'k-', linewidth=1.0)
    ax3.set_ylabel('ω cmd (rad/s)')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Angular Velocity Command')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'control_analysis.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")
    
    plt.show()


def export_for_matlab(metrics, output_dir):
    """Export key parameters for MATLAB μ-synthesis."""
    matlab_params_path = os.path.join(output_dir, 'plant_params_for_matlab.m')
    
    with open(matlab_params_path, 'w') as f:
        f.write("%% Plant Parameters from ROS Data\n")
        f.write("%% Auto-generated by analyze_control_data.py\n\n")
        
        f.write("% Nominal parameters\n")
        f.write("v_ref = 0.6;      % m/s (reference velocity)\n")
        f.write("nu = 0.2;         % angular skid coefficient\n")
        f.write("mu = 1.0;         % linear skid coefficient\n")
        f.write("dt = 0.2;         % sample time (s)\n")
        f.write("L = 0.27;         % wheelbase (m)\n\n")
        
        f.write("% Nominal plant (linearized)\n")
        f.write("A = [0, v_ref; 0, 0];\n")
        f.write("B = [0, 0; 0, nu];\n")
        f.write("C = eye(2);\n")
        f.write("D = zeros(2);\n")
        f.write("G = ss(A, B, C, D);\n\n")
        
        f.write("% Measured performance\n")
        f.write(f"settling_time = {metrics['settling_time_s']:.3f};  % seconds\n")
        f.write(f"ss_error_rms = {metrics['ss_tracking_rms']*1000:.3f};  % mm\n")
        f.write(f"bandwidth_est = {metrics.get('est_bandwidth_hz', 1.0):.2f};  % Hz\n\n")
        
        f.write("% Uncertainty weights (suggested)\n")
        f.write("% Low uncertainty at low freq, high at high freq (slip model)\n")
        f.write("s = tf('s');\n")
        f.write("W_I = 0.5 * (s/10 + 0.1) / (s/10 + 1);  % Input uncertainty\n\n")
        
        f.write("% Performance weights (suggested)\n")
        f.write("% Enforce integral action for tracking\n")
        f.write("M = 2;        % Peak sensitivity\n")
        f.write("omega_B = 1;  % Bandwidth (rad/s)\n")
        f.write("eps = 0.01;   % Steady-state error bound\n")
        f.write("W_P = (s/M + omega_B) / (s + omega_B*eps);\n")
    
    print(f"MATLAB parameters saved to: {matlab_params_path}")


def main():
    # Find CSV file
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        log_dir = os.path.expanduser('~/.ros/control_logs')
        csv_files = glob.glob(os.path.join(log_dir, 'control_data_*.csv'))
        if not csv_files:
            print(f"Error: No CSV files found in {log_dir}")
            print("Usage: python3 analyze_control_data.py <csv_file>")
            sys.exit(1)
        csv_path = max(csv_files, key=os.path.getmtime)
        print(f"Using latest file: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    # Load and analyze data
    print(f"\nLoading data from: {csv_path}")
    data = load_csv(csv_path)
    
    metrics, time, heading_err, distance_err, cmd_ang, tracking_err = compute_metrics(data)
    
    # Print results
    print_metrics(metrics)
    
    # Export for MATLAB
    output_dir = os.path.dirname(csv_path)
    export_for_matlab(metrics, output_dir)
    
    # Generate plots
    plot_results(time, heading_err, distance_err, cmd_ang, tracking_err, output_dir)


if __name__ == '__main__':
    main()
