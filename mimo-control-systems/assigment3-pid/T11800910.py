#!/usr/bin/env python3
"""
T11800910 - Tarefa 2 (PID contínuo e discreto)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import control


FIG_DIR = Path("figures")
TIME_VECTOR = np.linspace(0.0, 40.0, 2000)


def pid_tf(kp: float, ki: float, kd: float) -> control.TransferFunction:
    """PID ideal em domínio contínuo."""
    return control.tf([kd, kp, ki], [1.0, 0.0])


def step_metrics(sys: control.TransferFunction) -> dict[str, float]:
    info = control.step_info(sys)
    return {k: float(v) for k, v in info.items()}


def freq_metrics(open_loop: control.TransferFunction, closed_loop: control.TransferFunction) -> dict[str, float]:
    gm, pm, wg, wp = control.margin(open_loop)
    gm_db = 20.0 * math.log10(gm) if math.isfinite(gm) and gm > 0.0 else math.inf
    bw = control.bandwidth(closed_loop)
    return {
        "gain_margin_db": float(gm_db) if math.isfinite(gm_db) else math.inf,
        "phase_margin_deg": float(pm),
        "wg_rad_s": float(wg) if math.isfinite(wg) else math.nan,
        "wp_rad_s": float(wp) if math.isfinite(wp) else math.nan,
        "bandwidth_rad_s": float(bw),
    }


def plot_step(systems: list[tuple[control.TransferFunction, str]], filename: str) -> None:
    plt.figure()
    for sys, label in systems:
        if getattr(sys, "dt", None):
            t = np.arange(0.0, TIME_VECTOR[-1] + float(sys.dt), float(sys.dt))
        else:
            t = TIME_VECTOR
        t, y = control.step_response(sys, T=t)
        plt.plot(t, y, label=label)
    plt.plot(TIME_VECTOR, np.ones_like(TIME_VECTOR), "k--", linewidth=1.0, label="referência")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Saída")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename)
    plt.close()


def plot_bode(open_loops: list[tuple[control.TransferFunction, str]], filename: str) -> None:
    mag_ax = plt.subplot(2, 1, 1)
    phase_ax = plt.subplot(2, 1, 2)
    for sys, label in open_loops:
        if getattr(sys, "dt", None):
            nyquist = math.pi / float(sys.dt)
            upper = max(nyquist * 0.99, 1e-2)
            w = np.logspace(-2, math.log10(upper), 800)
        else:
            w = np.logspace(-2, 2, 800)
        fr = control.frequency_response(sys, w)
        mag = fr[0].squeeze()
        phase = np.rad2deg(fr[1].squeeze())
        mag_ax.semilogx(w, 20.0 * np.log10(np.abs(mag)), label=label)
        phase_ax.semilogx(w, phase)
    mag_ax.set_ylabel("Magnitude (dB)")
    phase_ax.set_ylabel("Fase (graus)")
    phase_ax.set_xlabel("Frequência (rad/s)")
    mag_ax.grid(True, which="both")
    phase_ax.grid(True, which="both")
    mag_ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename)
    plt.close()


def print_block(title: str, metrics: dict[str, float | int | str]) -> None:
    print(f"--- {title} ---")
    for key, value in metrics.items():
        if isinstance(value, str):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {float(value):.2f}")
    print()


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    print("1) Seleção da planta com dinâmica oscilatória")
    G = control.tf([1.0], [1.0, 2.0, 2.0, 1.0])
    print("G(s) = 1 / (s^3 + 2s^2 + 2s + 1)\n")

    print("2) PID contínuo obtido via Ziegler-Nichols (ganho limite)")
    kp_zn, ki_zn, kd_zn = 1.8, 0.81, 1.0
    C_zn = pid_tf(kp_zn, ki_zn, kd_zn)
    print_block(
        "Parâmetros do PID Z-N",
        {
            "Kp": kp_zn,
            "Ki": ki_zn,
            "Kd": kd_zn,
            "Ku": 3.0,
            "Tu_s": 2.0 * math.pi / math.sqrt(2.0),
        },
    )

    L_zn = C_zn * G
    T_zn = control.feedback(L_zn, 1)
    zn_step = step_metrics(T_zn)
    zn_freq = freq_metrics(L_zn, T_zn)
    print_block("3) PID Z-N - índices no domínio do tempo", zn_step)
    print_block("3) PID Z-N - índices no domínio da frequência", zn_freq)
    plot_step([(T_zn, "PID Z-N")], "item3_step_pid_zn.png")
    plot_bode([(L_zn, "PID Z-N")], "item3_bode_pid_zn.png")
    print("Figuras salvas: item3_step_pid_zn.png, item3_bode_pid_zn.png\n")

    print("4) Ajuste manual do PID para limitar Mp ≤ 20 %")
    C_tuned = pid_tf(1.6, 0.72, 0.9)
    print_block(
        "Ganho ajustado",
        {"Kp": 1.6, "Ki": 0.72, "Kd": 0.9},
    )

    L_tuned = C_tuned * G
    T_tuned = control.feedback(L_tuned, 1)
    tuned_step = step_metrics(T_tuned)
    tuned_freq = freq_metrics(L_tuned, T_tuned)
    print_block("5) PID ajustado - índices no domínio do tempo", tuned_step)
    print_block("5) PID ajustado - índices no domínio da frequência", tuned_freq)
    plot_step([(T_zn, "PID Z-N"), (T_tuned, "PID Ajustado")], "item5_step_comparacao.png")
    plot_bode([(L_zn, "PID Z-N"), (L_tuned, "PID Ajustado")], "item5_bode_comparacao.png")
    print("Figuras salvas: item5_step_comparacao.png, item5_bode_comparacao.png\n")

    print("6) Projeto dos compensadores em avanço e em atraso")
    alpha_lead = 0.1
    wc_target = 2.0
    T_lead = 1.0 / (math.sqrt(alpha_lead) * wc_target)
    lead = control.tf([T_lead, 1.0], [alpha_lead * T_lead, 1.0])
    C_lead = C_tuned * lead
    alpha_lag = 1.1
    T_lag = 0.3
    lag = control.tf([T_lag, 1.0], [alpha_lag * T_lag, 1.0])
    C_lag = pid_tf(1.8, 0.65, 1.0) * lag
    print_block(
        "6) Parâmetros",
        {
            "Lead_alpha": alpha_lead,
            "Lead_T_s": T_lead,
            "Lag_alpha": alpha_lag,
            "Lag_T_s": T_lag,
        },
    )

    L_lead = C_lead * G
    T_lead_cl = control.feedback(L_lead, 1)
    lead_step = step_metrics(T_lead_cl)
    lead_freq = freq_metrics(L_lead, T_lead_cl)
    L_lag = C_lag * G
    T_lag_cl = control.feedback(L_lag, 1)
    lag_step = step_metrics(T_lag_cl)
    lag_freq = freq_metrics(L_lag, T_lag_cl)
    print_block("8) Compensador em avanço - desempenho", lead_step | lead_freq)
    print_block("8) Compensador em atraso - desempenho", lag_step | lag_freq)
    plot_step(
        [(T_tuned, "PID Ajustado"), (T_lead_cl, "Avanço"), (T_lag_cl, "Atraso")],
        "item8_step_avanco_atraso.png",
    )
    plot_bode(
        [(L_tuned, "PID Ajustado"), (L_lead, "Avanço"), (L_lag, "Atraso")],
        "item8_bode_avanco_atraso.png",
    )
    print("Figuras salvas: item8_step_avanco_atraso.png, item8_bode_avanco_atraso.png\n")

    print("9) Definição da taxa de amostragem e discretização")
    wc = zn_freq["bandwidth_rad_s"]
    fc = wc / (2.0 * math.pi)
    fs = 20.0 * fc
    Ts = 1.0 / fs
    N_derivative = 10.0
    C_cont_filtered = control.tf([1.6], [1.0]) + control.tf([0.72], [1.0, 0.0]) + 0.9 * control.tf(
        [N_derivative, 0.0], [1.0, N_derivative]
    )
    C_discrete = control.c2d(C_cont_filtered, Ts, method="tustin")
    G_discrete = control.c2d(G, Ts, method="zoh")
    print_block(
        "9) Amostragem",
        {"wc_rad_s": wc, "fc_hz": fc, "fs_hz": fs, "Ts_s": Ts},
    )

    T_disc = control.feedback(C_discrete * G_discrete, 1)
    disc_step = step_metrics(T_disc)
    omega_disc = np.linspace(0.01, math.pi / Ts, 2500)
    disc_mag = control.frequency_response(T_disc, omega_disc)[0].squeeze()
    idx_disc = int(np.argmin(np.abs(disc_mag - 1.0 / math.sqrt(2.0))))
    wc_disc = float(omega_disc[idx_disc])
    poles_disc = control.poles(T_disc)
    num_d = np.squeeze(C_discrete.num)
    den_d = np.squeeze(C_discrete.den)
    a_terms = -den_d[1:]
    b_terms = num_d
    eq_str = f"u[k] = {a_terms[0]:.4f}·u[k-1]"
    eq_str += f" {'+' if a_terms[1] >= 0 else '-'} {abs(a_terms[1]):.4f}·u[k-2]"
    eq_str += f" {'+' if b_terms[0] >= 0 else '-'} {abs(b_terms[0]):.4f}·e[k]"
    eq_str += f" {'+' if b_terms[1] >= 0 else '-'} {abs(b_terms[1]):.4f}·e[k-1]"
    eq_str += f" {'+' if b_terms[2] >= 0 else '-'} {abs(b_terms[2]):.4f}·e[k-2]"

    plot_step(
        [(T_tuned, "Contínuo (PID Ajustado)"), (T_disc, "Discreto (PID+ZOH)")],
        "item12_step_cont_x_disc.png",
    )

    plot_bode(
        [
            (L_tuned, "Contínuo (PID Ajustado)"),
            (control.series(C_discrete, G_discrete), "Discreto (PID Filtrado)"),
        ],
        "item12_bode_cont_x_disc.png",
    )

    print_block(
        "10) Equação a diferenças do controlador discretizado",
        {"u[k]": eq_str},
    )
    print_block(
        "11-12) Comparação contínuo x discreto",
        {
            "wc_disc_rad_s": wc_disc,
            "overshoot_cont_%": tuned_step["Overshoot"],
            "overshoot_disc_%": disc_step["Overshoot"],
            "settling_cont_s": tuned_step["SettlingTime"],
            "settling_disc_s": disc_step["SettlingTime"],
            "max_pole_abs": float(np.max(np.abs(poles_disc))),
        },
    )
    print("Figuras salvas: item12_step_cont_x_disc.png, item12_bode_cont_x_disc.png\n")


if __name__ == "__main__":
    main()
