#!/usr/bin/env python3
"""
(pêndulo invertido).
"""

import math
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# Funções auxiliares compartilhadas pelos itens da lista
# ---------------------------------------------------------------------------

def evaluate_tf(num: np.ndarray, den: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Retorna G(jw) para um TF com coeficientes em potências decrescentes."""
    return np.polyval(num, 1j * w) / np.polyval(den, 1j * w)


def closed_loop_tf(num_g: np.ndarray, den_g: np.ndarray, num_c: np.ndarray, den_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retorna o numerador e denominador da malha fechada com realimentação unitária."""
    num_l = np.polymul(num_c, num_g)
    den_l = np.polymul(den_c, den_g)
    if len(den_l) > len(num_l):
        num_l = np.pad(num_l, (len(den_l) - len(num_l), 0))
    else:
        den_l = np.pad(den_l, (len(num_l) - len(den_l), 0))
    den_t = np.polyadd(den_l, num_l)
    num_t = num_l.copy()
    # remove zeros em excesso nas extremidades
    num_t = np.trim_zeros(num_t, "f")
    den_t = np.trim_zeros(den_t, "f")
    num_t = np.trim_zeros(num_t, "b")
    den_t = np.trim_zeros(den_t, "b")
    return num_t, den_t


def unity_prefilter(num_t: np.ndarray, den_t: np.ndarray) -> tuple[np.ndarray, float]:
    """Retorna numerador escalonado para ganho unitário (aproximação DC)."""
    s_eval = 1e-6
    dc_gain = np.polyval(num_t, s_eval) / np.polyval(den_t, s_eval)
    scale = 1.0 / dc_gain if dc_gain != 0 else 1.0
    return num_t * scale, float(dc_gain)


def time_metrics(t: np.ndarray, y: np.ndarray, target: float | None = 1.0) -> dict:
    """Calcula overshoot (%), tempos de subida e acomodação (2%), valor final e erro."""
    final = y[-1]
    ref = final if target is None else target
    peak = np.max(y)
    overshoot = max(0.0, (peak - ref) / ref * 100.0)
    band = 0.02 * ref
    settling_time = math.nan
    out_of_band = np.where(np.abs(y - ref) > band)[0]
    if out_of_band.size:
        last_out = out_of_band[-1]
        if last_out + 1 < t.size:
            settling_time = t[last_out + 1]
    else:
        settling_time = t[0]
    rise_time = math.nan
    above = np.where(y >= 0.9 * ref)[0]
    if above.size:
        rise_time = t[above[0]]
    return {
        "final_value": final,
        "overshoot_%": overshoot,
        "settling_time_s": settling_time,
        "rise_time_s": rise_time,
        "steady_state_error": ref - final,
    }


def is_stable_continuous(den: np.ndarray) -> bool:
    return np.all(np.real(np.roots(den)) < 0)


def is_stable_discrete(den: np.ndarray) -> bool:
    return np.all(np.abs(np.roots(den)) < 1)


def frequency_metrics(num_l: np.ndarray, den_l: np.ndarray) -> dict:
    """Calcula margens de ganho/fase e frequência de corte."""
    w = np.logspace(-2, 3, 8000)
    l_vals = evaluate_tf(num_l, den_l, w)
    mag = np.abs(l_vals)
    phase = np.unwrap(np.angle(l_vals))
    wc = math.nan
    idx = np.where(np.diff(np.sign(mag - 1.0)) != 0)[0]
    if idx.size:
        i = idx[0]
        w1, w2 = w[i], w[i + 1]
        m1, m2 = mag[i], mag[i + 1]
        wc = w1 + (1.0 - m1) * (w2 - w1) / (m2 - m1)
    else:
        wc = w[np.argmin(np.abs(mag - 1.0))]
    phase_deg = phase * 180.0 / np.pi
    pm = math.nan
    if not math.isnan(wc):
        phase_wc = np.interp(wc, w, phase_deg)
        pm = 180.0 + phase_wc
    target = phase + np.pi
    gm = math.nan
    gm_db = math.nan
    idx_p = np.where(np.diff(np.sign(target)) != 0)[0]
    if idx_p.size:
        i = idx_p[0]
        w1, w2 = w[i], w[i + 1]
        v1, v2 = target[i], target[i + 1]
        wp = w1 - v1 * (w2 - w1) / (v2 - v1)
        l_wp = evaluate_tf(num_l, den_l, np.array([wp]))[0]
        gm = 1.0 / abs(l_wp)
        gm_db = 20.0 * math.log10(gm)
    return {
        "wc_rad_s": wc,
        "phase_margin_deg": pm,
        "gain_margin": gm,
        "gain_margin_db": gm_db,
        "frequencies_rad_s": w,
        "magnitude": mag,
        "phase_deg": phase_deg,
    }


def bode_plot(path: Path, w: np.ndarray, mag: np.ndarray, phase_deg: np.ndarray, title: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.semilogx(w, 20.0 * np.log10(mag))
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both")
    ax2.semilogx(w, phase_deg)
    ax2.set_ylabel("Fase (graus)")
    ax2.set_xlabel("Frequência (rad/s)")
    ax2.grid(True, which="both")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_step(path: Path, t: np.ndarray, responses: dict, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (time_vec, y) in responses.items():
        ax.plot(time_vec, y, label=label)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Saída θ (rad)")
    ax.grid(True)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def difference_equation(num: np.ndarray, den: np.ndarray) -> str:
    """Retorna string com a equação a diferenças y[k] para TF discreta num/den."""
    n = num.flatten()
    d = den.flatten()
    assert abs(d[0] - 1.0) < 1e-9, "Denominador deve ser moníc."
    a1 = -d[1]
    a2 = -d[2]
    b0, b1, b2 = n
    return f"u[k] = {a1:.6f}·u[k-1] + {a2:.6f}·u[k-2] + {b0:.6f}·e[k] + {b1:.6f}·e[k-1] + {b2:.6f}·e[k-2]"


# ---------------------------------------------------------------------------
# 1) Seleção do sistema
# ---------------------------------------------------------------------------
# Modelo simplificado do pêndulo invertido linearizado (θ como saída) com duas
# dinâmicas de 1ª ordem para atuar-servo e sensor. O ganho AC final possui
# polo no semiplano direito, com tendência oscilatória/instável aberta.
g = 9.81  # gravidade (m/s²)
l = 0.5   # meia haste (m)
omega0 = math.sqrt(g / l)
tau_act = 0.1
tau_sensor = 0.2
num_plant = np.array([1.0, 0.0])
den_plant = np.polymul(np.polymul([tau_act, 1.0], [tau_sensor, 1.0]), [1.0, 0.0, -(omega0**2)])


# ---------------------------------------------------------------------------
# 2) PID pela regra de Ziegler-Nichols (método da amplitude crítica)
# ---------------------------------------------------------------------------
def ultimate_gain(num: np.ndarray, den: np.ndarray) -> tuple[float, float, float]:
    w = np.logspace(-2, 3, 10000)
    g_vals = evaluate_tf(num, den, w)
    phase = np.unwrap(np.angle(g_vals))
    target = phase + np.pi
    idx = np.where(np.diff(np.sign(target)) != 0)[0]
    if idx.size:
        i = idx[0]
        w1, w2 = w[i], w[i + 1]
        v1, v2 = target[i], target[i + 1]
        w180 = w1 - v1 * (w2 - w1) / (v2 - v1)
    else:
        w180 = w[np.argmin(np.abs(target))]
    g_at = evaluate_tf(num, den, np.array([w180]))[0]
    ku = 1.0 / abs(g_at)
    tu = 2.0 * np.pi / w180
    return ku, tu, w180


ku, tu, w_crit = ultimate_gain(num_plant, den_plant)
kp_zn = 0.6 * ku
ti_zn = 0.5 * tu
td_zn = 0.125 * tu
ki_zn = kp_zn / ti_zn
kd_zn = kp_zn * td_zn
filter_n = 10.0 / td_zn
pid_zn_num = np.array([filter_n * (kp_zn + kd_zn), kp_zn + ki_zn * filter_n, ki_zn])
pid_zn_den = np.array([filter_n, 1.0, 0.0])


# ---------------------------------------------------------------------------
# 3) Avaliação da malha com PID de ZN (índices tempo/frequência)
# ---------------------------------------------------------------------------
num_t_zn, den_t_zn = closed_loop_tf(num_plant, den_plant, pid_zn_num, pid_zn_den)
time_vec = np.linspace(0, 10, 4000)
loop_metrics_zn = frequency_metrics(np.polymul(pid_zn_num, num_plant), np.polymul(pid_zn_den, den_plant))
if is_stable_continuous(den_t_zn):
    num_t_zn_scaled, _ = unity_prefilter(num_t_zn, den_t_zn)
    sys_zn = signal.TransferFunction(num_t_zn_scaled, den_t_zn)
    _, y_zn = signal.step(sys_zn, T=time_vec)
    metrics_zn = time_metrics(time_vec, y_zn)
else:
    y_zn = np.zeros_like(time_vec)
    metrics_zn = {
        "final_value": math.nan,
        "overshoot_%": math.nan,
        "settling_time_s": math.nan,
        "rise_time_s": math.nan,
        "steady_state_error": math.nan,
    }


# ---------------------------------------------------------------------------
# 4) Reajuste dos ganhos para limitar overshoot em 20%
# ---------------------------------------------------------------------------
scale_k = (0.6631578947368421, 0.7894736842105263, 1.0105263157894737)
kp_tuned = kp_zn * scale_k[0]
ki_tuned = ki_zn * scale_k[1]
kd_tuned = kd_zn * scale_k[2]
pid_tuned_num = np.array([filter_n * (kp_tuned + kd_tuned), kp_tuned + ki_tuned * filter_n, ki_tuned])
pid_tuned_den = pid_zn_den.copy()
num_t_tuned, den_t_tuned = closed_loop_tf(num_plant, den_plant, pid_tuned_num, pid_tuned_den)
num_t_tuned_scaled, final_raw_tuned = unity_prefilter(num_t_tuned, den_t_tuned)
sys_tuned = signal.TransferFunction(num_t_tuned_scaled, den_t_tuned)
_, y_tuned = signal.step(sys_tuned, T=time_vec)
metrics_tuned = time_metrics(time_vec, y_tuned)


# ---------------------------------------------------------------------------
# 5) Comparação dos índices de desempenho
# ---------------------------------------------------------------------------
def compare_metrics(m1: dict, m2: dict) -> dict:
    return {key: (m1[key], m2[key]) for key in m1}


comparison_pid = compare_metrics(metrics_zn, metrics_tuned)


# ---------------------------------------------------------------------------
# 6) Compensadores em avanço e atraso com ωc ≥ ωc(ZN)
# ---------------------------------------------------------------------------
wc_target_lead = 1.2 * loop_metrics_zn["wc_rad_s"]
phi_max = 45.0
alpha_lead = (1 - math.sin(math.radians(phi_max))) / (1 + math.sin(math.radians(phi_max)))
tau_lead = 1.0 / (wc_target_lead * math.sqrt(alpha_lead))
g_wc = evaluate_tf(num_plant, den_plant, np.array([wc_target_lead]))[0]
mag_lead = math.sqrt((1 + (wc_target_lead * tau_lead) ** 2) / (1 + (wc_target_lead * alpha_lead * tau_lead) ** 2))
k_lead = 1.0 / (abs(g_wc) * mag_lead)
lead_num = k_lead * np.array([tau_lead, 1.0])
lead_den = np.array([alpha_lead * tau_lead, 1.0])

beta_lag = 5.0
wc_target_lag = loop_metrics_zn["wc_rad_s"]
tau_lag = 10.0 / wc_target_lag
mag_lag = math.sqrt((1 + (wc_target_lag * tau_lag) ** 2) / (1 + (wc_target_lag * beta_lag * tau_lag) ** 2))
g_wlag = evaluate_tf(num_plant, den_plant, np.array([wc_target_lag]))[0]
k_lag = 1.0 / (abs(g_wlag) * mag_lag)
lag_num = k_lag * np.array([tau_lag, 1.0])
lag_den = np.array([beta_lag * tau_lag, 1.0])


# ---------------------------------------------------------------------------
# 7) Comentários embutidos (ver código) sobre escolhas dos parâmetros
# ---------------------------------------------------------------------------
# - φ_max de 45° fornece reforço de fase suficiente com α ≈ 0.1716.
# - τ_lead posiciona zero/polo simetricamente em torno de ωc desejado.
# - β=5 e τ_lag deslocam zero 1 década abaixo de ωc para reforço de ganho DC.


# ---------------------------------------------------------------------------
# 8) Comparação de desempenho para avanço/atraso
# ---------------------------------------------------------------------------
num_t_lead, den_t_lead = closed_loop_tf(num_plant, den_plant, lead_num, lead_den)
if is_stable_continuous(den_t_lead):
    sys_lead = signal.TransferFunction(num_t_lead, den_t_lead)
    _, y_lead = signal.step(sys_lead, T=time_vec)
    metrics_lead = time_metrics(time_vec, y_lead, target=None)
else:
    y_lead = np.zeros_like(time_vec)
    metrics_lead = {
        "final_value": math.nan,
        "overshoot_%": math.nan,
        "settling_time_s": math.nan,
        "rise_time_s": math.nan,
        "steady_state_error": math.nan,
    }
loop_metrics_lead = frequency_metrics(np.polymul(lead_num, num_plant), np.polymul(lead_den, den_plant))

num_t_lag, den_t_lag = closed_loop_tf(num_plant, den_plant, lag_num, lag_den)
if is_stable_continuous(den_t_lag):
    sys_lag = signal.TransferFunction(num_t_lag, den_t_lag)
    _, y_lag = signal.step(sys_lag, T=time_vec)
    metrics_lag = time_metrics(time_vec, y_lag, target=None)
else:
    y_lag = np.zeros_like(time_vec)
    metrics_lag = {
        "final_value": math.nan,
        "overshoot_%": math.nan,
        "settling_time_s": math.nan,
        "rise_time_s": math.nan,
        "steady_state_error": math.nan,
    }
loop_metrics_lag = frequency_metrics(np.polymul(lag_num, num_plant), np.polymul(lag_den, den_plant))


# ---------------------------------------------------------------------------
# 9) Discretização do controlador sintonizado – Ts assumido 10 ms
# ---------------------------------------------------------------------------
ts = 0.01
cd_result = signal.cont2discrete((pid_tuned_num, pid_tuned_den), ts, method="tustin")
if len(cd_result) == 3:
    num_cd, den_cd, _ = cd_result
else:  # SciPy >=1.10
    num_cd, den_cd, _, _, _ = cd_result
gd_result = signal.cont2discrete((num_plant, den_plant), ts, method="zoh")
if len(gd_result) == 3:
    num_gd, den_gd, _ = gd_result
else:
    num_gd, den_gd, _, _, _ = gd_result
num_cd = num_cd.flatten()
den_cd = den_cd.flatten()
num_gd = num_gd.flatten()
den_gd = den_gd.flatten()


# ---------------------------------------------------------------------------
# 10) Equação recursiva do controlador discreto
# ---------------------------------------------------------------------------
difference_eq = difference_equation(num_cd, den_cd)


# ---------------------------------------------------------------------------
# 11) Simulação em tempo discreto (malha fechada com planta discretizada)
# ---------------------------------------------------------------------------
num_ld = np.polymul(num_cd, num_gd)
den_ld = np.polymul(den_cd, den_gd)
num_td = np.polymul(num_cd, num_gd)
den_td = np.polyadd(den_ld, num_ld)
num_td = np.trim_zeros(num_td, "f")
den_td = np.trim_zeros(den_td, "f")
if is_stable_discrete(den_td):
    num_td_scaled, _ = unity_prefilter(num_td, den_td)
    step_d = signal.dstep((num_td_scaled, den_td, ts), n=400)
    time_d = step_d[0]
    y_d = np.squeeze(step_d[1])
    metrics_discrete = time_metrics(time_d, y_d)
else:
    time_d = np.arange(400) * ts
    y_d = np.zeros_like(time_d)
    metrics_discrete = {
        "final_value": math.nan,
        "overshoot_%": math.nan,
        "settling_time_s": math.nan,
        "rise_time_s": math.nan,
        "steady_state_error": math.nan,
    }


# ---------------------------------------------------------------------------
# 12) Comparação entre a resposta contínua e discreta
# ---------------------------------------------------------------------------
comparison_discrete = compare_metrics(metrics_tuned, metrics_discrete)


# ---------------------------------------------------------------------------
# Geração dos gráficos solicitados
# ---------------------------------------------------------------------------
here = Path(__file__).resolve().parent
fig_dir = here / "figures"

plot_step(
    fig_dir / "step_pid_zn.png",
    time_vec,
    {"PID ZN (com pré-filtro)": (time_vec, y_zn)},
    "Item 3 – Resposta com PID Ziegler-Nichols",
)

plot_step(
    fig_dir / "step_pid_comparison.png",
    time_vec,
    {
        "PID ZN": (time_vec, y_zn),
        "PID Ajustado": (time_vec, y_tuned),
    },
    "Itens 3–5 – Comparação de controladores",
)

plot_step(
    fig_dir / "step_compensators.png",
    time_vec,
    {
        "Avanço": (time_vec, y_lead),
        "Atraso": (time_vec, y_lag),
    },
    "Itens 6–8 – Compensadores de avanço/atraso",
)

bode_plot(
    fig_dir / "bode_pid_zn.png",
    loop_metrics_zn["frequencies_rad_s"],
    loop_metrics_zn["magnitude"],
    loop_metrics_zn["phase_deg"],
    "Item 3 – Bode da malha com PID ZN",
)

plot_step(
    fig_dir / "step_continuo_vs_discreto.png",
    time_vec,
    {
        "Contínuo (PID Ajustado)": (time_vec, y_tuned),
        "Discreto (Ts=10 ms)": (time_d, y_d),
    },
    "Itens 11–12 – Comparação contínua x discreta",
)


# ---------------------------------------------------------------------------
# Relatório textual no terminal
# ---------------------------------------------------------------------------
print("# Item 2 – Ganho e período críticos (método Ziegler-Nichols)")
print(f"Ku = {ku:.4f}, Tu = {tu:.4f} s, ω180 = {w_crit:.4f} rad/s")
print(f"Kp_ZN = {kp_zn:.4f}, Ki_ZN = {ki_zn:.4f}, Kd_ZN = {kd_zn:.4f}")

print("\n# Item 3 – Índices com PID ZN (após pré-filtro para ganho unitário)")
for k, v in metrics_zn.items():
    print(f"{k}: {v:.5f}")
print(f"Frequência de corte ≈ {loop_metrics_zn['wc_rad_s']:.4f} rad/s")
print(f"Margem de fase ≈ {loop_metrics_zn['phase_margin_deg']:.2f}°")
print(f"Margem de ganho ≈ {loop_metrics_zn['gain_margin_db']:.2f} dB")

print("\n# Item 4 – PID ajustado (sobressinal ≤ 20%)")
print(f"Escalas adotadas (Kp, Ki, Kd) = {scale_k}")
for k, v in metrics_tuned.items():
    print(f"{k}: {v:.5f}")

print("\n# Item 5 – Comparação PID ZN x Ajustado")
for k, (v1, v2) in comparison_pid.items():
    print(f"{k}: ZN={v1:.5f}, Ajustado={v2:.5f}")

print("\n# Item 8 – Desempenho dos compensadores")
print("Compensador em avanço:")
for k, v in metrics_lead.items():
    print(f"  {k}: {v:.5f}")
print(f"  Margem de fase: {loop_metrics_lead['phase_margin_deg']:.2f}°")
print("Compensador em atraso:")
for k, v in metrics_lag.items():
    print(f"  {k}: {v:.5f}")
print(f"  Margem de fase: {loop_metrics_lag['phase_margin_deg']:.2f}°")

print("\n# Item 9 – Controlador discreto (Tustin, Ts = 10 ms)")
print("Numerador:", num_cd)
print("Denominador:", den_cd)

print("\n# Item 10 – Equação a diferenças do PID discreto")
print(difference_eq)

print("\n# Item 11 – Desempenho em tempo discreto")
for k, v in metrics_discrete.items():
    print(f"{k}: {v:.5f}")

print("\n# Item 12 – Comparação contínuo x discreto")
for k, (v1, v2) in comparison_discrete.items():
    print(f"{k}: contínuo={v1:.5f}, discreto={v2:.5f}")

print("\nFiguras salvas em:", fig_dir)
