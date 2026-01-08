#!/usr/bin/env python3
"""
Robust Control Analysis - Theoretical Metrics and Visualization

Este script implementa análises de controle robusto conforme a teoria da disciplina:
- Resposta em frequência do sistema
- Análise de estabilidade nominal (NS)
- Análise de desempenho nominal (NP)
- Análise de estabilidade robusta (RS) via Small Gain Theorem
- Análise de desempenho robusto (RP)
- Margens de ganho e fase
- Visualização de Bode, Nyquist e μ-plots

Requer: numpy, scipy, matplotlib
Opcional: python-control (para funções avançadas)

Usage:
    python3 robust_analysis.py                    # Usa parâmetros default
    python3 robust_analysis.py --csv <arquivo>    # Carrega dados de CSV para validação
"""

import numpy as np
from scipy import signal
import os
import argparse

# Matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed, plots will be skipped")

# Optional: python-control for advanced analysis
try:
    import control as ct
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False


class RobustControlAnalysis:
    """
    Classe para análise de controle robusto conforme teoria da disciplina.
    
    Implementa:
    - Modelagem da planta nominal G(s)
    - Pesos de incerteza W_I(s)
    - Pesos de desempenho W_P(s)
    - Análise de NS, NP, RS, RP
    """
    
    def __init__(self, params=None):
        """
        Inicializa parâmetros do sistema.
        
        Args:
            params: dicionário com parâmetros da planta e controlador
        """
        # Parâmetros default do sistema Terra
        self.params = {
            'v_ref': 0.6,       # Velocidade de referência (m/s)
            'nu': 0.2,          # Coeficiente de skid angular (nominal)
            'mu': 1.0,          # Coeficiente de skid linear (nominal)
            'dt': 0.2,          # Período de amostragem (s)
            'Lbase': 0.27,      # Base da roda (m)
            
            # Incertezas
            'nu_uncertainty': 0.5,   # ±50% incerteza em nu
            'mu_uncertainty': 0.2,   # ±20% incerteza em mu
            
            # Especificações de desempenho
            'omega_B': 1.0,          # Largura de banda desejada (rad/s)
            'M_s': 2.0,              # Pico de sensibilidade máximo
            'epsilon': 0.01,         # Erro em regime permanente
            
            # Ganhos do controlador (PI ou H∞)
            'Kp_y': 10.2,
            'Kp_theta': 10.0,
            'Ki_y': 1.02,
            'Ki_theta': 1.0,
        }
        
        if params:
            self.params.update(params)
        
        # Frequências para análise
        self.omega = np.logspace(-2, 2, 500)  # 0.01 a 100 rad/s
        
        # Diretório para salvar resultados
        self.output_dir = os.path.expanduser('~/.ros/robust_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
    
    # =========================================================================
    # SEÇÃO 1: MODELAGEM DO SISTEMA
    # =========================================================================
    
    def create_plant(self):
        """
        Cria a planta nominal linearizada G(s).
        
        Modelo linearizado ao redor de trajetória reta:
            ẏ = v_ref * θ
            θ̇ = ν * ω
        
        Returns:
            Sistema em espaço de estados ou função de transferência
        """
        v_ref = self.params['v_ref']
        nu = self.params['nu']
        
        # Espaço de estados: x = [y, θ], u = ω, y = [y, θ]
        A = np.array([[0, v_ref], [0, 0]])
        B = np.array([[0], [nu]])
        C = np.eye(2)
        D = np.zeros((2, 1))
        
        self.plant_ss = (A, B, C, D)
        
        print("="*60)
        print("PLANTA NOMINAL G(s)")
        print("="*60)
        print(f"Velocidade: v_ref = {v_ref} m/s")
        print(f"Skid angular: ν = {nu}")
        print(f"\nMatrizes do espaço de estados:")
        print(f"  A = {A.tolist()}")
        print(f"  B = {B.tolist()}")
        print(f"  C = {C.tolist()}")
        print(f"  D = {D.tolist()}")
        
        # Funções de transferência individuais
        # G_y(s) = v_ref * nu / s^2  (de ω para y)
        # G_theta(s) = nu / s         (de ω para θ)
        
        self.G_y_num = [v_ref * nu]
        self.G_y_den = [1, 0, 0]
        
        self.G_theta_num = [nu]
        self.G_theta_den = [1, 0]
        
        print(f"\nFunções de transferência:")
        print(f"  G_y(s) = {v_ref*nu} / s²")
        print(f"  G_θ(s) = {nu} / s")
        print("="*60 + "\n")
        
        return self.plant_ss
    
    def create_controller(self):
        """
        Cria o controlador K(s).
        
        Controlador PI com estrutura:
            ω = Kp_θ*θ + Ki_θ*∫θ + Kp_y*y/v_ref + Ki_y*∫y/v_ref
        
        Returns:
            Tupla (numerador, denominador) da função de transferência
        """
        v_ref = self.params['v_ref']
        Kp_y = self.params['Kp_y']
        Kp_theta = self.params['Kp_theta']
        Ki_y = self.params['Ki_y']
        Ki_theta = self.params['Ki_theta']
        
        # Para a análise SISO de ω para θ, consideramos apenas o canal θ
        # K_theta(s) = Kp_theta + Ki_theta/s = (Kp_theta*s + Ki_theta) / s
        self.K_theta_num = [Kp_theta, Ki_theta]
        self.K_theta_den = [1, 0]
        
        # Para o canal y
        # K_y(s) = (Kp_y + Ki_y/s) / v_ref = (Kp_y*s + Ki_y) / (v_ref*s)
        self.K_y_num = [Kp_y, Ki_y]
        self.K_y_den = [v_ref, 0]
        
        print("="*60)
        print("CONTROLADOR K(s)")
        print("="*60)
        print(f"Ganhos (H∞ sintetizado):")
        print(f"  Kp_y = {Kp_y:.4f}")
        print(f"  Kp_θ = {Kp_theta:.4f}")
        print(f"  Ki_y = {Ki_y:.4f}")
        print(f"  Ki_θ = {Ki_theta:.4f}")
        print(f"\nFunção de transferência (canal θ):")
        print(f"  K_θ(s) = ({Kp_theta}s + {Ki_theta}) / s")
        print("="*60 + "\n")
        
        return (self.K_theta_num, self.K_theta_den)
    
    def create_weights(self):
        """
        Cria os pesos de desempenho W_P(s) e incerteza W_I(s).
        
        W_P(s): Peso de sensibilidade para tracking
        W_I(s): Peso de incerteza multiplicativa de entrada
        """
        M = self.params['M_s']
        omega_B = self.params['omega_B']
        eps = self.params['epsilon']
        r_0 = 0.1
        r_inf = self.params['nu_uncertainty']
        omega_b = 10.0
        
        # W_P(s) = (s/M + omega_B) / (s + omega_B*eps)
        self.W_P_num = [1/M, omega_B]
        self.W_P_den = [1, omega_B * eps]
        
        # W_I(s) = (r_inf*s/omega_b + r_0) / (s/omega_b + 1)
        self.W_I_num = [r_inf/omega_b, r_0]
        self.W_I_den = [1/omega_b, 1]
        
        print("="*60)
        print("PESOS DE SÍNTESE")
        print("="*60)
        print(f"Peso de Desempenho W_P(s):")
        print(f"  W_P = (s/{M} + {omega_B}) / (s + {omega_B*eps})")
        print(f"  |W_P(0)| = {omega_B/(omega_B*eps):.1f} ({20*np.log10(omega_B/(omega_B*eps)):.1f} dB)")
        print(f"  |W_P(∞)| = {1/M:.2f} ({20*np.log10(1/M):.1f} dB)")
        print(f"\nPeso de Incerteza W_I(s):")
        print(f"  W_I = ({r_inf}s/{omega_b} + {r_0}) / (s/{omega_b} + 1)")
        print(f"  |W_I(0)| = {r_0:.2f} ({r_0*100:.0f}% incerteza)")
        print(f"  |W_I(∞)| = {r_inf:.2f} ({r_inf*100:.0f}% incerteza)")
        print("="*60 + "\n")
    
    # =========================================================================
    # SEÇÃO 2: RESPOSTA EM FREQUÊNCIA
    # =========================================================================
    
    def compute_frequency_response(self, num, den, omega=None):
        """
        Calcula a resposta em frequência de uma função de transferência.
        
        Args:
            num: coeficientes do numerador
            den: coeficientes do denominador
            omega: vetor de frequências (rad/s)
        
        Returns:
            magnitude (dB), fase (graus), omega
        """
        if omega is None:
            omega = self.omega
        
        s = 1j * omega
        
        # Avalia numerador e denominador
        num_eval = np.polyval(num, s)
        den_eval = np.polyval(den, s)
        
        # Evita divisão por zero
        den_eval = np.where(np.abs(den_eval) < 1e-10, 1e-10, den_eval)
        
        H = num_eval / den_eval
        
        mag_db = 20 * np.log10(np.abs(H))
        phase_deg = np.angle(H, deg=True)
        
        return mag_db, phase_deg, omega
    
    def compute_loop_transfer(self):
        """
        Calcula a função de transferência de malha aberta L(s) = G(s)*K(s).
        
        Para o canal θ:
            L_θ(s) = G_θ(s) * K_θ(s) = (ν/s) * (Kp*s + Ki)/s
                   = ν*(Kp*s + Ki) / s²
        """
        nu = self.params['nu']
        Kp = self.params['Kp_theta']
        Ki = self.params['Ki_theta']
        
        # L(s) = nu * (Kp*s + Ki) / s^2
        self.L_num = [nu * Kp, nu * Ki]
        self.L_den = [1, 0, 0]
        
        return (self.L_num, self.L_den)
    
    def compute_sensitivity(self):
        """
        Calcula a função de sensibilidade S(s) = 1 / (1 + L(s)).
        
        S(s) = s² / (s² + ν*Kp*s + ν*Ki)
        """
        nu = self.params['nu']
        Kp = self.params['Kp_theta']
        Ki = self.params['Ki_theta']
        
        # S(s) = s² / (s² + nu*Kp*s + nu*Ki)
        self.S_num = [1, 0, 0]
        self.S_den = [1, nu * Kp, nu * Ki]
        
        return (self.S_num, self.S_den)
    
    def compute_complementary_sensitivity(self):
        """
        Calcula a função de sensibilidade complementar T(s) = L(s) / (1 + L(s)).
        
        T(s) = ν*(Kp*s + Ki) / (s² + ν*Kp*s + ν*Ki)
        """
        nu = self.params['nu']
        Kp = self.params['Kp_theta']
        Ki = self.params['Ki_theta']
        
        # T(s) = nu*(Kp*s + Ki) / (s² + nu*Kp*s + nu*Ki)
        self.T_num = [nu * Kp, nu * Ki]
        self.T_den = [1, nu * Kp, nu * Ki]
        
        return (self.T_num, self.T_den)
    
    # =========================================================================
    # SEÇÃO 3: ANÁLISE DE ESTABILIDADE E DESEMPENHO
    # =========================================================================
    
    def analyze_nominal_stability(self):
        """
        Análise de Estabilidade Nominal (NS).
        
        Verifica se os polos de malha fechada estão no semiplano esquerdo.
        """
        print("="*60)
        print("ANÁLISE DE ESTABILIDADE NOMINAL (NS)")
        print("="*60)
        
        # Polos do sistema de malha fechada
        poles = np.roots(self.S_den)
        
        print(f"Polos de malha fechada:")
        stable = True
        for i, p in enumerate(poles):
            status = "✓" if np.real(p) < 0 else "✗"
            if np.real(p) >= 0:
                stable = False
            print(f"  p{i+1} = {p:.4f}  (Re = {np.real(p):.4f}) {status}")
        
        # Frequência natural e amortecimento
        if len(poles) == 2:
            omega_n = np.sqrt(np.abs(poles[0] * poles[1]))
            zeta = -np.real(poles[0] + poles[1]) / (2 * omega_n)
            print(f"\nCaracterísticas da resposta:")
            print(f"  Frequência natural: ωn = {omega_n:.3f} rad/s")
            print(f"  Amortecimento: ζ = {zeta:.3f}")
        
        result = "✓ ESTÁVEL" if stable else "✗ INSTÁVEL"
        print(f"\nResultado NS: {result}")
        print("="*60 + "\n")
        
        return stable, poles
    
    def analyze_nominal_performance(self):
        """
        Análise de Desempenho Nominal (NP).
        
        Verifica: ||W_P * S||∞ < 1
        
        Ou seja, a sensibilidade ponderada deve ser menor que 1 em todas as frequências.
        """
        print("="*60)
        print("ANÁLISE DE DESEMPENHO NOMINAL (NP)")
        print("="*60)
        
        # Calcula |W_P(jω) * S(jω)|
        mag_WP, _, omega = self.compute_frequency_response(self.W_P_num, self.W_P_den)
        mag_S, _, _ = self.compute_frequency_response(self.S_num, self.S_den)
        
        # |W_P * S| em dB
        mag_WPS = mag_WP + mag_S  # soma de dB = produto
        
        # Pico
        peak_WPS = np.max(mag_WPS)
        peak_WPS_linear = 10**(peak_WPS/20)
        peak_freq = omega[np.argmax(mag_WPS)]
        
        print(f"Critério: ||W_P * S||∞ < 1")
        print(f"\nResultados:")
        print(f"  ||W_P * S||∞ = {peak_WPS_linear:.4f} ({peak_WPS:.2f} dB)")
        print(f"  Frequência do pico: ω = {peak_freq:.3f} rad/s")
        
        np_satisfied = peak_WPS_linear < 1
        result = "✓ SATISFEITO" if np_satisfied else "✗ NÃO SATISFEITO"
        print(f"\nResultado NP: {result}")
        print("="*60 + "\n")
        
        return np_satisfied, peak_WPS_linear
    
    def analyze_robust_stability(self):
        """
        Análise de Estabilidade Robusta (RS).
        
        Small Gain Theorem para incerteza multiplicativa de entrada:
            RS ⟺ ||W_I * T||∞ < 1
        
        Para perturbações estruturadas, usamos μ:
            RS ⟺ μ_Δ(M) < 1, ∀ω
        """
        print("="*60)
        print("ANÁLISE DE ESTABILIDADE ROBUSTA (RS)")
        print("="*60)
        
        # Small Gain Theorem: ||W_I * T||∞ < 1
        mag_WI, _, omega = self.compute_frequency_response(self.W_I_num, self.W_I_den)
        mag_T, _, _ = self.compute_frequency_response(self.T_num, self.T_den)
        
        # |W_I * T|
        mag_WIT = mag_WI + mag_T
        
        peak_WIT = np.max(mag_WIT)
        peak_WIT_linear = 10**(peak_WIT/20)
        peak_freq = omega[np.argmax(mag_WIT)]
        
        print(f"Small Gain Theorem: ||W_I * T||∞ < 1")
        print(f"\nResultados:")
        print(f"  ||W_I * T||∞ = {peak_WIT_linear:.4f} ({peak_WIT:.2f} dB)")
        print(f"  Frequência do pico: ω = {peak_freq:.3f} rad/s")
        
        # Interpretação da margem de estabilidade robusta
        margin = 1 / peak_WIT_linear if peak_WIT_linear > 0 else float('inf')
        print(f"\nMargem de estabilidade robusta: {margin:.2f}")
        print(f"(Sistema tolera incerteza {margin*100:.0f}% do modelo)")
        
        rs_satisfied = peak_WIT_linear < 1
        result = "✓ ESTABILIDADE ROBUSTA GARANTIDA" if rs_satisfied else "✗ RS NÃO GARANTIDA"
        print(f"\nResultado RS: {result}")
        print("="*60 + "\n")
        
        return rs_satisfied, peak_WIT_linear, margin
    
    def analyze_robust_performance(self):
        """
        Análise de Desempenho Robusto (RP).
        
        Para sistemas SISO com incerteza multiplicativa:
            RP ⟺ ||W_P * S||∞ + ||W_I * T||∞ < 1
        
        Para sistemas MIMO, usamos μ com bloco de desempenho fictício:
            RP ⟺ μ_Δ̃(N) < 1, onde Δ̃ = diag{Δ, Δ_P}
        """
        print("="*60)
        print("ANÁLISE DE DESEMPENHO ROBUSTO (RP)")
        print("="*60)
        
        # Cálculo de |W_P * S| e |W_I * T|
        mag_WP, _, omega = self.compute_frequency_response(self.W_P_num, self.W_P_den)
        mag_S, _, _ = self.compute_frequency_response(self.S_num, self.S_den)
        mag_WI, _, _ = self.compute_frequency_response(self.W_I_num, self.W_I_den)
        mag_T, _, _ = self.compute_frequency_response(self.T_num, self.T_den)
        
        # Conversão de dB para linear
        WPS_linear = 10**(mag_WP/20) * 10**(mag_S/20)
        WIT_linear = 10**(mag_WI/20) * 10**(mag_T/20)
        
        # Para SISO: RP ⟺ max(|W_P*S| + |W_I*T|) < 1
        # Isso é uma aproximação conservadora
        RP_metric = WPS_linear + WIT_linear
        peak_RP = np.max(RP_metric)
        peak_freq = omega[np.argmax(RP_metric)]
        
        print(f"Critério SISO (conservador):")
        print(f"  ||W_P*S|| + ||W_I*T|| < 1")
        print(f"\nResultados:")
        print(f"  max(|W_P*S| + |W_I*T|) = {peak_RP:.4f}")
        print(f"  Frequência do pico: ω = {peak_freq:.3f} rad/s")
        
        # Para MIMO, precisamos calcular μ
        # Aqui aproximamos usando σ̄(N)
        print(f"\nNota: Para análise exata de RP em MIMO, calcular μ_Δ̃(N)")
        
        rp_satisfied = peak_RP < 1
        result = "✓ DESEMPENHO ROBUSTO GARANTIDO" if rp_satisfied else "✗ RP NÃO GARANTIDO"
        print(f"\nResultado RP: {result}")
        print("="*60 + "\n")
        
        return rp_satisfied, peak_RP
    
    def compute_stability_margins(self):
        """
        Calcula margens de estabilidade clássicas (ganho e fase).
        """
        print("="*60)
        print("MARGENS DE ESTABILIDADE")
        print("="*60)
        
        # Encontra frequência de cruzamento de fase (L = -180°)
        _, phase_L, omega = self.compute_frequency_response(self.L_num, self.L_den)
        
        # Frequência de ganho unitário (|L| = 0 dB)
        mag_L, _, _ = self.compute_frequency_response(self.L_num, self.L_den)
        
        # Margem de fase: fase em |L| = 0 dB + 180°
        idx_crossover = np.argmin(np.abs(mag_L))
        omega_c = omega[idx_crossover]
        phase_at_crossover = phase_L[idx_crossover]
        phase_margin = 180 + phase_at_crossover
        
        # Margem de ganho: |L| em fase = -180°
        phase_unwrapped = np.unwrap(np.deg2rad(phase_L))
        idx_180 = np.argmin(np.abs(phase_unwrapped + np.pi))
        omega_180 = omega[idx_180]
        gain_at_180 = mag_L[idx_180]
        gain_margin = -gain_at_180
        
        print(f"Margem de Fase:")
        print(f"  Frequência de cruzamento: ωc = {omega_c:.3f} rad/s")
        print(f"  Margem de fase: PM = {phase_margin:.1f}°")
        
        print(f"\nMargem de Ganho:")
        print(f"  Frequência -180°: ω-180 = {omega_180:.3f} rad/s")
        print(f"  Margem de ganho: GM = {gain_margin:.1f} dB ({10**(gain_margin/20):.2f}x)")
        
        # Interpretação
        if phase_margin > 30 and gain_margin > 6:
            stability = "✓ Margens adequadas para robustez"
        else:
            stability = "⚠ Margens baixas, sensível a incertezas"
        
        print(f"\nInterpretação: {stability}")
        print("="*60 + "\n")
        
        return phase_margin, gain_margin, omega_c
    
    # =========================================================================
    # SEÇÃO 4: VISUALIZAÇÃO
    # =========================================================================
    
    def plot_bode(self):
        """
        Gera diagrama de Bode para L(s), S(s), T(s).
        """
        if not HAS_MATPLOTLIB:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        omega = self.omega
        
        # L(s)
        mag_L, phase_L, _ = self.compute_frequency_response(self.L_num, self.L_den)
        # S(s)
        mag_S, phase_S, _ = self.compute_frequency_response(self.S_num, self.S_den)
        # T(s)
        mag_T, phase_T, _ = self.compute_frequency_response(self.T_num, self.T_den)
        
        # Magnitude
        ax1 = axes[0]
        ax1.semilogx(omega, mag_L, 'b-', linewidth=2, label='L(s) - Malha aberta')
        ax1.semilogx(omega, mag_S, 'r-', linewidth=2, label='S(s) - Sensibilidade')
        ax1.semilogx(omega, mag_T, 'g-', linewidth=2, label='T(s) - Sens. compl.')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title('Diagrama de Bode - Funções de Malha')
        ax1.legend(loc='best')
        ax1.grid(True, which='both', alpha=0.3)
        ax1.set_ylim([-60, 60])
        
        # Fase
        ax2 = axes[1]
        ax2.semilogx(omega, phase_L, 'b-', linewidth=2, label='L(s)')
        ax2.semilogx(omega, phase_S, 'r-', linewidth=2, label='S(s)')
        ax2.semilogx(omega, phase_T, 'g-', linewidth=2, label='T(s)')
        ax2.axhline(y=-180, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Frequência (rad/s)')
        ax2.set_ylabel('Fase (graus)')
        ax2.legend(loc='best')
        ax2.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'bode_diagram.png')
        plt.savefig(filepath, dpi=150)
        print(f"Bode diagram saved to: {filepath}")
        plt.show()
    
    def plot_robustness(self):
        """
        Gera gráficos de análise de robustez (W_P*S, W_I*T, μ-plot).
        """
        if not HAS_MATPLOTLIB:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        omega = self.omega
        
        # Pesos e funções de sensibilidade
        mag_WP, _, _ = self.compute_frequency_response(self.W_P_num, self.W_P_den)
        mag_WI, _, _ = self.compute_frequency_response(self.W_I_num, self.W_I_den)
        mag_S, _, _ = self.compute_frequency_response(self.S_num, self.S_den)
        mag_T, _, _ = self.compute_frequency_response(self.T_num, self.T_den)
        
        # Conversão para linear
        WPS_linear = 10**((mag_WP + mag_S)/20)
        WIT_linear = 10**((mag_WI + mag_T)/20)
        
        # Plot 1: |W_P| e |S|
        ax1 = axes[0, 0]
        ax1.semilogx(omega, mag_S, 'r-', linewidth=2, label='|S(jω)|')
        ax1.semilogx(omega, -mag_WP, 'b--', linewidth=2, label='1/|W_P(jω)|')
        ax1.fill_between(omega, -100, -mag_WP, alpha=0.2, color='blue')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title('Desempenho Nominal: |S| < 1/|W_P|')
        ax1.legend()
        ax1.grid(True, which='both', alpha=0.3)
        ax1.set_ylim([-60, 20])
        
        # Plot 2: |W_I| e |T|
        ax2 = axes[0, 1]
        ax2.semilogx(omega, mag_T, 'g-', linewidth=2, label='|T(jω)|')
        ax2.semilogx(omega, -mag_WI, 'm--', linewidth=2, label='1/|W_I(jω)|')
        ax2.fill_between(omega, -100, -mag_WI, alpha=0.2, color='magenta')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_title('Estabilidade Robusta: |T| < 1/|W_I|')
        ax2.legend()
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_ylim([-60, 20])
        
        # Plot 3: |W_P*S| para NP
        ax3 = axes[1, 0]
        ax3.semilogx(omega, WPS_linear, 'r-', linewidth=2, label='|W_P·S|')
        ax3.axhline(y=1, color='k', linestyle='--', linewidth=2, label='Limite (γ=1)')
        ax3.fill_between(omega, 0, 1, alpha=0.2, color='green')
        ax3.set_xlabel('Frequência (rad/s)')
        ax3.set_ylabel('Magnitude')
        ax3.set_title('Desempenho Nominal: ||W_P·S||∞ < 1')
        ax3.legend()
        ax3.grid(True, which='both', alpha=0.3)
        ax3.set_ylim([0, 2])
        
        # Plot 4: |W_I*T| para RS
        ax4 = axes[1, 1]
        ax4.semilogx(omega, WIT_linear, 'g-', linewidth=2, label='|W_I·T|')
        ax4.axhline(y=1, color='k', linestyle='--', linewidth=2, label='Limite (γ=1)')
        ax4.fill_between(omega, 0, 1, alpha=0.2, color='green')
        ax4.set_xlabel('Frequência (rad/s)')
        ax4.set_ylabel('Magnitude')
        ax4.set_title('Estabilidade Robusta: ||W_I·T||∞ < 1')
        ax4.legend()
        ax4.grid(True, which='both', alpha=0.3)
        ax4.set_ylim([0, 2])
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'robustness_analysis.png')
        plt.savefig(filepath, dpi=150)
        print(f"Robustness analysis saved to: {filepath}")
        plt.show()
    
    def plot_mu_approximation(self):
        """
        Gera μ-plot aproximado para análise de Desempenho Robusto.
        
        Para sistemas SISO, μ ≈ |W_P*S| + |W_I*T| (limite superior conservador).
        """
        if not HAS_MATPLOTLIB:
            return
        
        omega = self.omega
        
        # Cálculo de μ aproximado
        mag_WP, _, _ = self.compute_frequency_response(self.W_P_num, self.W_P_den)
        mag_WI, _, _ = self.compute_frequency_response(self.W_I_num, self.W_I_den)
        mag_S, _, _ = self.compute_frequency_response(self.S_num, self.S_den)
        mag_T, _, _ = self.compute_frequency_response(self.T_num, self.T_den)
        
        WPS_linear = 10**((mag_WP + mag_S)/20)
        WIT_linear = 10**((mag_WI + mag_T)/20)
        
        # μ upper bound
        mu_upper = WPS_linear + WIT_linear
        
        # σ̄(N) - maximum singular value approximation
        sigma_bar = np.sqrt(WPS_linear**2 + WIT_linear**2)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.semilogx(omega, mu_upper, 'r-', linewidth=2, 
                    label='μ upper bound: |W_P·S| + |W_I·T|')
        ax.semilogx(omega, sigma_bar, 'b--', linewidth=2,
                    label='σ̄(N): √(|W_P·S|² + |W_I·T|²)')
        ax.semilogx(omega, WPS_linear, 'g:', linewidth=1.5, label='|W_P·S|')
        ax.semilogx(omega, WIT_linear, 'm:', linewidth=1.5, label='|W_I·T|')
        ax.axhline(y=1, color='k', linestyle='--', linewidth=2, label='γ = 1 (RP limit)')
        
        ax.fill_between(omega, 0, 1, alpha=0.1, color='green')
        
        ax.set_xlabel('Frequência (rad/s)')
        ax.set_ylabel('Magnitude')
        ax.set_title('μ-Plot Aproximado para Desempenho Robusto (RP)\nRP garantido se μ < 1 ∀ω')
        ax.legend(loc='best')
        ax.grid(True, which='both', alpha=0.3)
        ax.set_ylim([0, max(2, np.max(mu_upper) * 1.1)])
        
        # Marca pico
        peak_idx = np.argmax(mu_upper)
        ax.annotate(f'Peak: {mu_upper[peak_idx]:.3f}',
                    xy=(omega[peak_idx], mu_upper[peak_idx]),
                    xytext=(omega[peak_idx]*2, mu_upper[peak_idx]*0.8),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'mu_plot.png')
        plt.savefig(filepath, dpi=150)
        print(f"μ-plot saved to: {filepath}")
        plt.show()
    
    # =========================================================================
    # SEÇÃO 5: RELATÓRIO
    # =========================================================================
    
    def generate_report(self):
        """
        Gera relatório completo de análise.
        """
        print("\n" + "="*70)
        print("RELATÓRIO DE ANÁLISE DE CONTROLE ROBUSTO")
        print("="*70)
        
        # Cria sistema completo
        self.create_plant()
        self.create_controller()
        self.create_weights()
        
        # Calcula funções de transferência
        self.compute_loop_transfer()
        self.compute_sensitivity()
        self.compute_complementary_sensitivity()
        
        # Análises
        ns_ok, poles = self.analyze_nominal_stability()
        np_ok, np_peak = self.analyze_nominal_performance()
        rs_ok, rs_peak, rs_margin = self.analyze_robust_stability()
        rp_ok, rp_peak = self.analyze_robust_performance()
        pm, gm, wc = self.compute_stability_margins()
        
        # Resumo
        print("\n" + "="*70)
        print("RESUMO")
        print("="*70)
        print(f"  Estabilidade Nominal (NS):    {'✓' if ns_ok else '✗'}")
        print(f"  Desempenho Nominal (NP):      {'✓' if np_ok else '✗'} (peak = {np_peak:.3f})")
        print(f"  Estabilidade Robusta (RS):    {'✓' if rs_ok else '✗'} (peak = {rs_peak:.3f})")
        print(f"  Desempenho Robusto (RP):      {'✓' if rp_ok else '✗'} (peak = {rp_peak:.3f})")
        print(f"\n  Margem de Fase:               {pm:.1f}°")
        print(f"  Margem de Ganho:              {gm:.1f} dB")
        print(f"  Frequência de Cruzamento:     {wc:.2f} rad/s")
        print("="*70)
        
        # Gera plots
        if HAS_MATPLOTLIB:
            print("\nGerando gráficos...")
            self.plot_bode()
            self.plot_robustness()
            self.plot_mu_approximation()
        
        # Salva resultados em CSV
        self.save_results_csv(ns_ok, np_ok, rs_ok, rp_ok, pm, gm)
        
        return {
            'NS': ns_ok,
            'NP': (np_ok, np_peak),
            'RS': (rs_ok, rs_peak, rs_margin),
            'RP': (rp_ok, rp_peak),
            'PM': pm,
            'GM': gm,
            'omega_c': wc
        }
    
    def save_results_csv(self, ns, np_ok, rs, rp, pm, gm):
        """
        Salva resultados numéricos em CSV.
        """
        filepath = os.path.join(self.output_dir, 'robust_analysis_results.csv')
        
        with open(filepath, 'w') as f:
            f.write("metric,value,unit,status\n")
            f.write(f"Nominal_Stability,{1 if ns else 0},,{'OK' if ns else 'FAIL'}\n")
            f.write(f"Nominal_Performance,{1 if np_ok else 0},,{'OK' if np_ok else 'FAIL'}\n")
            f.write(f"Robust_Stability,{1 if rs else 0},,{'OK' if rs else 'FAIL'}\n")
            f.write(f"Robust_Performance,{1 if rp else 0},,{'OK' if rp else 'FAIL'}\n")
            f.write(f"Phase_Margin,{pm:.2f},degrees,\n")
            f.write(f"Gain_Margin,{gm:.2f},dB,\n")
            f.write(f"v_ref,{self.params['v_ref']},m/s,\n")
            f.write(f"nu,{self.params['nu']},,\n")
            f.write(f"nu_uncertainty,{self.params['nu_uncertainty']*100:.0f},%,\n")
        
        print(f"\nResults saved to: {filepath}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Análise de Controle Robusto')
    parser.add_argument('--csv', type=str, help='Arquivo CSV com dados experimentais')
    parser.add_argument('--output', type=str, help='Diretório de saída')
    args = parser.parse_args()
    
    # Cria analisador
    analyzer = RobustControlAnalysis()
    
    if args.output:
        analyzer.output_dir = args.output
    
    # Executa análise completa
    results = analyzer.generate_report()
    
    print("\n" + "="*70)
    print("ANÁLISE COMPLETA!")
    print(f"Resultados salvos em: {analyzer.output_dir}")
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    main()
