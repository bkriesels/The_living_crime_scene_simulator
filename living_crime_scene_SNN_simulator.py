# living_crime_scene_snn_speedversion7.py
# ---------------------------------------------------------------------------
# "The Living Crime Scene" – SNN + aanvalsscenario's + forensische metriek
#
# Auteur: Berry Kriesels
#
# Versie 7:
#   - Gebaseerd op stabiele v4b-script
#   - Extra variatie: lichte achtergrond-jitter op input-rate
#   - Nieuwe feature: detection_delay (eerste afwijking t.o.v. baseline)
#   - Nieuwe features: vroege/late attack-metrics (early/late mean & entropy)
#   - Compatibel met bestaande analyse-scripts (alleen extra kolommen)
# ---------------------------------------------------------------------------

from brian2 import *
import numpy as np
import pandas as pd
import os
import time
from datetime import timedelta
import multiprocessing as mp

################################################################################
# Globale configuratie
################################################################################

# Zet codegen target. Standaard op numpy voor stabiliteit en makkelijker debuggen.
CODEGEN_TARGET = "numpy"
prefs.codegen.target = CODEGEN_TARGET

defaultclock.dt = 0.1 * ms

# Netwerk grootte en connectiviteit (sparse houden voor performance)
N = 1000
p_connect = 0.05

# STDP instellingen (plasticiteit)
tau_pre = 20 * ms
tau_post = 20 * ms
Apre = 0.01
Apost = -Apre * tau_pre / tau_post * 1.05
wmax = 1.0

# Standaard Neuron parameters (LIF model)
EL_val = -65 * mV
Vth = -50 * mV
Vreset = -65 * mV
tau_m = 20 * ms
R_m = 100 * Mohm
refr = 2 * ms

# Tijdlijnen voor de simulatie fases
duration_baseline = 5 * second
duration_attack = 5 * second
duration_mitig = 20 * second

# Recovery window: we kijken alleen naar het laatste stukje van de mitigatie
RECOVERY_EVAL_WINDOW = 5 * second

# Nominale input rates. Baseline is rustig, mitigatie schroeven we terug.
inp_rate_baseline_nominal = 8 * Hz
inp_rate_mitig_base_nominal = 4 * Hz

# ---------------- Attack-intensity basiswaarden (Default = HIGH) -------------
# Hier definiëren we de "zwaarte" van de aanvallen.
# Deze waarden worden later overschreven als we --attack-intensity low/medium kiezen.

# Scenario 1: Overstimulation (DDoS-achtig)
BASE_inp_rate_attack_overstim = 200 * Hz
BASE_OVERSTIM_PULSE_FRAC = 0.4  # 40% van de neuronen krijgt een klap
BASE_OVERSTIM_PULSE_I = 1.5 * nA  # Amplitude van de extra puls

# Scenario 2: Poisoning (Backdoor/Data manipulatie)
BASE_poison_fraction = 0.10  # Update kans
BASE_poison_target_frac = 0.15  # Percentage neuronen met backdoor
BASE_poison_weight_factor = 2.0  # Forceer gewichten omhoog

# Scenario 3: Drift (Supply chain / aging)
BASE_drift_step = 0.03  # ~3% verloop per interval
drift_interval = 200 * ms

# Scenario 4: sVCE (Virus verspreiding - SEIR model op de graaf)
svce_update_dt = 20 * ms
BASE_svce_beta = 0.10  # Infectiekans
BASE_svce_sigma = 0.08  # E -> I
BASE_svce_gamma = 0.03  # I -> R
BASE_svce_initial_frac_I = 0.03  # Patient zero populatie

# Runtime-variabelen (deze worden dynamisch gezet door configure_attack_intensity)
inp_rate_attack_overstim = BASE_inp_rate_attack_overstim
OVERSTIM_PULSE_FRAC = BASE_OVERSTIM_PULSE_FRAC
OVERSTIM_PULSE_I = BASE_OVERSTIM_PULSE_I

poison_fraction = BASE_poison_fraction
poison_target_frac = BASE_poison_target_frac
poison_weight_factor = BASE_poison_weight_factor

drift_step = BASE_drift_step

svce_beta = BASE_svce_beta
svce_sigma = BASE_svce_sigma
svce_gamma = BASE_svce_gamma
svce_initial_frac_I = BASE_svce_initial_frac_I

# Hoe vaak samplen we voor statistiek?
stat_bin = 10 * ms

# Homeostase targets (om het netwerk stabiel te houden buiten attacks om)
target_rate = 20 * Hz
homeo_window = 50 * ms

# Globale flags
MITIGATION_LEVEL = "high"
ATTACK_INTENSITY = "high"


################################################################################
# Attack-intensity configuratie
################################################################################

def configure_attack_intensity(level: str):
    """
    Stelt de globale parameters in op basis van low/medium/high.
    High is de default, low is handig om te zien of metrics gevoelig genoeg zijn.
    """
    global ATTACK_INTENSITY
    global inp_rate_attack_overstim, OVERSTIM_PULSE_FRAC, OVERSTIM_PULSE_I
    global poison_fraction, poison_target_frac, poison_weight_factor
    global drift_step
    global svce_beta, svce_sigma, svce_gamma, svce_initial_frac_I

    ATTACK_INTENSITY = level

    if level == "low":
        # Alles op een laag pitje.
        inp_rate_attack_overstim = BASE_inp_rate_attack_overstim * 0.4
        OVERSTIM_PULSE_FRAC = BASE_OVERSTIM_PULSE_FRAC * 0.4
        OVERSTIM_PULSE_I = BASE_OVERSTIM_PULSE_I * 0.6

        poison_fraction = BASE_poison_fraction * 0.5
        poison_target_frac = BASE_poison_target_frac * 0.5
        poison_weight_factor = 1.0 + (BASE_poison_weight_factor - 1.0) * 0.5

        drift_step = BASE_drift_step * 0.4

        svce_beta = BASE_svce_beta * 0.5
        svce_sigma = BASE_svce_sigma * 0.5
        svce_gamma = BASE_svce_gamma * 0.6
        svce_initial_frac_I = BASE_svce_initial_frac_I * 0.5

    elif level == "medium":
        # Tussenweg
        inp_rate_attack_overstim = BASE_inp_rate_attack_overstim * 0.7
        OVERSTIM_PULSE_FRAC = BASE_OVERSTIM_PULSE_FRAC * 0.7
        OVERSTIM_PULSE_I = BASE_OVERSTIM_PULSE_I * 0.8

        poison_fraction = BASE_poison_fraction * 0.8
        poison_target_frac = BASE_poison_target_frac * 0.8
        poison_weight_factor = 1.0 + (BASE_poison_weight_factor - 1.0) * 0.8

        drift_step = BASE_drift_step * 0.7

        svce_beta = BASE_svce_beta * 0.8
        svce_sigma = BASE_svce_sigma * 0.8
        svce_gamma = BASE_svce_gamma * 0.8
        svce_initial_frac_I = BASE_svce_initial_frac_I * 0.8

    else:  # high
        # Gewoon de basiswaarden gebruiken (full impact)
        inp_rate_attack_overstim = BASE_inp_rate_attack_overstim
        OVERSTIM_PULSE_FRAC = BASE_OVERSTIM_PULSE_FRAC
        OVERSTIM_PULSE_I = BASE_OVERSTIM_PULSE_I

        poison_fraction = BASE_poison_fraction
        poison_target_frac = BASE_poison_target_frac
        poison_weight_factor = BASE_poison_weight_factor

        drift_step = BASE_drift_step

        svce_beta = BASE_svce_beta
        svce_sigma = BASE_svce_sigma
        svce_gamma = BASE_svce_gamma
        svce_initial_frac_I = BASE_svce_initial_frac_I


################################################################################
# Forensische helpers
################################################################################

def metrics_from_spikes(pop_rate_mon, bin_size=10 * ms,
                        t_start=0 * ms, t_stop=None):
    """
    Haalt de belangrijkste stats uit de rate monitor.
    Std_rate is toegevoegd om detection threshold beter te kunnen bepalen.
    """
    if len(pop_rate_mon.t) == 0:
        return dict(mean_rate=np.nan, burstiness=np.nan,
                    synchrony=np.nan, entropy=np.nan,
                    std_rate=np.nan)

    ts = pop_rate_mon.t
    rs = pop_rate_mon.smooth_rate(window='flat', width=bin_size)  # Hz
    rs = np.asarray(rs)

    if t_stop is None:
        t_stop = ts[-1]

    # Filter op het gevraagde tijdsvenster
    mask = (ts >= t_start) & (ts <= t_stop)
    if not np.any(mask):
        return dict(mean_rate=np.nan, burstiness=np.nan,
                    synchrony=np.nan, entropy=np.nan,
                    std_rate=np.nan)

    rs_seg = rs[mask]
    mean_rate = float(np.mean(rs_seg))
    std_rate = float(np.std(rs_seg))

    # Burstiness (Coefficient of Variation)
    if mean_rate > 1e-9:
        burstiness = std_rate / mean_rate
    else:
        burstiness = np.nan

    # Synchrony: var/mean. Simpele maatstaf, maar werkt.
    synchrony = (std_rate ** 2) / (mean_rate + 1e-9)

    # Entropie berekening (Shannon)
    hist, _ = np.histogram(rs_seg, bins=20, density=True)
    p = hist / (np.sum(hist) + 1e-12)
    entropy = -np.sum(p * np.log2(p + 1e-12))
    entropy_max = np.log2(len(p)) if len(p) > 0 else 1.0
    entropy_norm = float(entropy / entropy_max) if entropy_max > 0 else np.nan

    return dict(mean_rate=mean_rate,
                burstiness=burstiness,
                synchrony=synchrony,
                entropy=entropy_norm,
                std_rate=std_rate)


def compute_rate_slope(pop_rate_mon, t_start, t_stop):
    """
    Kijkt of de rate stijgt of daalt in het interval (Linear Least Squares).
    Handig om drift of langzame crashes te spotten.
    """
    ts = pop_rate_mon.t
    rs = np.asarray(pop_rate_mon.smooth_rate(window='flat', width=stat_bin))

    mask = (ts >= t_start) & (ts <= t_stop)
    if not np.any(mask):
        return np.nan

    x = (ts[mask] - ts[mask][0]) / second
    y = rs[mask]
    if len(x) < 2:
        return np.nan

    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)


def compute_heterogeneity(spike_mon, t_start, t_stop, n_neurons):
    """
    Heterogeniteit = CV van de firing rates per neuron.
    Als alle neuronen synchroon vuren, is dit laag. Als de boel chaotisch is of
    sommige neuronen kapot zijn, is dit hoog.
    """
    if len(spike_mon.i) == 0:
        return np.nan

    # Brian2 quantities even strippen naar floats
    t = np.asarray(spike_mon.t / second)
    i = np.asarray(spike_mon.i, dtype=int)
    t0 = float(t_start / second)
    t1 = float(t_stop / second)

    mask = (t >= t0) & (t <= t1)
    if not np.any(mask):
        return np.nan

    i = i[mask]
    window = t1 - t0
    if window <= 0:
        return np.nan

    counts = np.bincount(i, minlength=n_neurons)
    rates = counts / window

    mean_rate = np.mean(rates)
    std_rate = np.std(rates)

    if mean_rate <= 1e-9:
        return np.nan

    return float(std_rate / mean_rate)


def compute_detection_delay(pop_rate_mon,
                            t_start_atk,
                            t_stop_atk,
                            base_mean,
                            base_burstiness,
                            k_sigma=3.0):
    """
    Wanneer gaat het alarm af?
    We zoeken het eerste punt in de attack-fase waar rate > mean + 3*sigma.
    """
    if np.isnan(base_mean) or np.isnan(base_burstiness):
        return np.nan, np.nan

    base_std = base_burstiness * base_mean
    threshold = base_mean + k_sigma * base_std

    ts = np.asarray(pop_rate_mon.t / second)
    rs = np.asarray(pop_rate_mon.smooth_rate(window='flat', width=stat_bin))

    t0 = float(t_start_atk / second)
    t1 = float(t_stop_atk / second)

    mask = (ts >= t0) & (ts <= t1)
    if not np.any(mask):
        return np.nan, threshold

    idx = np.where((rs > threshold) & mask)[0]
    if len(idx) == 0:
        return np.nan, threshold

    first_t = ts[idx[0]]
    delay = first_t - t0
    return float(delay), threshold


################################################################################
# Helper: fase-detectie
################################################################################

def _in_attack_phase(t):
    return (t >= duration_baseline) and (t < duration_baseline + duration_attack)


################################################################################
# Netwerkconstructie
################################################################################

def build_network(seed_val=42,
                  trial_jitter_factor=1.0):
    """
    Bouwt het complete LIF netwerk inclusief STDP en monitors.
    Trial_jitter_factor zorgt ervoor dat niet elke run exact dezelfde input krijgt.
    """
    np.random.seed(seed_val)
    seed(seed_val)

    # LIF vergelijkingen + SEIR state tracking
    eqs = '''
    dv/dt = (-(v - EL) + R*I_syn)/tau : volt (unless refractory)
    I_syn : amp
    EL    : volt
    R     : ohm
    tau   : second
    seir_state : integer  # 0=S,1=E,2=I,3=R
    '''

    G = NeuronGroup(
        N, eqs,
        threshold='v > Vth',
        reset='v = Vreset',
        refractory=refr,
        method='euler'
    )
    # Init met wat random ruis op voltage
    G.v = EL_val + (np.random.rand(N) * 5 - 2.5) * mV

    # Heterogeniteit in hardware parameters simuleren (niet elk neuron is gelijk)
    G.EL = EL_val + (np.random.randn(N) * 1.0) * mV
    G.R = R_m
    G.tau = tau_m * (1.0 + 0.1 * np.random.randn(N))
    G.seir_state = 0

    # Synapsen met STDP (Plasticiteit)
    S = Synapses(
        G, G,
        model='''
        w : 1
        dapre/dt = -apre/taupre : 1 (event-driven)
        dapost/dt = -apost/taupost : 1 (event-driven)
        ''',
        on_pre='''
        I_syn_post += w * 0.5*nA
        apre += sim_Apre
        w = clip(w + apost, 0, sim_wmax)
        ''',
        on_post='''
        apost += sim_Apost
        w = clip(w + apre, 0, sim_wmax)
        '''
    )
    # Namespaces koppelen voor de C++ codegen
    S.namespace['taupre'] = tau_pre
    S.namespace['taupost'] = tau_post
    S.namespace['sim_Apre'] = Apre
    S.namespace['sim_Apost'] = Apost
    S.namespace['sim_wmax'] = wmax

    S.connect(p=p_connect, condition='i != j')
    S.w = '0.1 + 0.1 * rand()'

    # Poisson-input (de "zintuigen" van het netwerk)
    baseline_rate = inp_rate_baseline_nominal * trial_jitter_factor
    P = PoissonGroup(N, rates=baseline_rate)
    Sinp = Synapses(P, G, model='w:1', on_pre='I_syn_post += w * 0.8*nA')
    Sinp.connect(j='i')
    Sinp.w = '0.2 + 0.05 * rand()'

    # Monitors aanzetten
    M_sp = SpikeMonitor(G)
    M_rate = PopulationRateMonitor(G)

    # Homeostase mechanisme
    # Zorgt dat het netwerk niet ontploft of doodvalt als er geen attack is.
    @network_operation(dt=500 * ms)
    def homeostasis():
        if len(M_rate.t) == 0:
            return
        t = defaultclock.t
        # Tijdens attack staat homeostase uit (simuleert overweldiging)
        if _in_attack_phase(t):
            return
        current_rate = float(M_rate.smooth_rate(
            window='flat', width=homeo_window)[-1])
        scale = 1.0
        if current_rate > (target_rate / Hz + 5.0):
            scale = 0.98
        elif current_rate < (target_rate / Hz - 5.0):
            scale = 1.02
        Sinp.w[:] = clip(Sinp.w[:] * scale, 0, 1.5)

    # Achtergrond Jitter
    # Maakt de input iets 'natuurlijker' en minder statisch.
    @network_operation(dt=200 * ms)
    def background_rate_jitter():
        jitter = np.random.normal(1.0, 0.05)
        new_rates = P.rates * jitter
        P.rates = clip(new_rates, 0 * Hz, 250 * Hz)

    return dict(
        G=G,
        P=P,
        S=S,
        Sinp=Sinp,
        M_sp=M_sp,
        M_rate=M_rate,
        homeostasis=homeostasis,
        background_rate_jitter=background_rate_jitter,
    )


################################################################################
# Scenario-specifieke operators
################################################################################

def make_overstim_pulse_operator(G):
    """
    Overstimulation: willekeurige neuronen krijgen extra stroomstoten.
    Simuleert een DDoS of flooding attack.
    """

    @network_operation(dt=20 * ms)
    def overstim_pulse():
        t = defaultclock.t
        if not _in_attack_phase(t):
            return
        n = len(G)
        k = int(OVERSTIM_PULSE_FRAC * n)
        if k <= 0:
            return
        idx = np.random.choice(np.arange(n), size=k, replace=False)
        G.I_syn[idx] += OVERSTIM_PULSE_I

    return overstim_pulse


def make_poisoning_operator(Sinp, poison_targets):
    """
    Poisoning: versterkt synapsen van een specifieke set neuronen.
    Probeert een backdoor patroon erin te branden.
    """

    @network_operation(dt=50 * ms)
    def poisoning():
        t = defaultclock.t
        if not _in_attack_phase(t):
            return

        if np.random.rand() < poison_fraction:
            Sinp.w[poison_targets] = np.clip(
                Sinp.w[poison_targets] * poison_weight_factor,
                0, 1.5
            )

    return poisoning


def make_drift_operator(G, S):
    """
    Drift: langzaam verlopen van tijdconstantes (tau) en gewichten.
    Lastig te detecteren op korte termijn.
    """

    @network_operation(dt=drift_interval)
    def drift():
        t = defaultclock.t
        if not _in_attack_phase(t):
            return

        G.tau[:] = G.tau[:] * (1.0 + drift_step)
        S.w[:] = clip(S.w[:] * (1.0 + drift_step), 0, 1.5)

    return drift


def make_svce_operator(G, S):
    """
    sVCE (Spiking Virus): SEIR logica.
    Geïnfecteerde nodes (I) besmetten buren via synapsen.
    Infectie zorgt voor degradatie van hardware params (EL, tau).
    """
    i_arr = np.array(S.i[:])
    j_arr = np.array(S.j[:])

    @network_operation(dt=svce_update_dt)
    def svce():
        t = defaultclock.t
        if not _in_attack_phase(t):
            return

        state = G.seir_state[:]

        # Infecteer buren (S -> E)
        infectious = np.where(state == 2)[0]
        if len(infectious) > 0:
            for i in infectious:
                neigh = j_arr[i_arr == i]
                if len(neigh) == 0:
                    continue
                mask = np.random.rand(len(neigh)) < svce_beta
                newly = neigh[mask]
                mask_S = (state[newly] == 0)
                state[newly[mask_S]] = 1

                # Incubatie (E -> I)
        exposed = np.where(state == 1)[0]
        if len(exposed) > 0:
            mask_EI = np.random.rand(len(exposed)) < svce_sigma
            state[exposed[mask_EI]] = 2

        # Recovery / Death (I -> R)
        infectious = np.where(state == 2)[0]
        if len(infectious) > 0:
            mask_IR = np.random.rand(len(infectious)) < svce_gamma
            removed = infectious[mask_IR]
            state[removed] = 3

        G.seir_state[:] = state

        # Hier passen we de schade toe (hardware degradatie)
        G.EL[state == 2] = -55 * mV
        G.tau[state == 2] = 6 * ms

        # R staat voor Removed/Dead, dus die doen bijna niks meer
        G.EL[state == 3] = -82 * mV
        G.tau[state == 3] = 3 * ms

    return svce


################################################################################
# Mitigatie-acties
################################################################################

def apply_rate_limiting(P):
    """
    Noodrem: input rate omlaag gooien afhankelijk van mitigation level.
    """
    if MITIGATION_LEVEL == "low":
        rate = inp_rate_mitig_base_nominal
    elif MITIGATION_LEVEL == "medium":
        rate = inp_rate_mitig_base_nominal * 0.5
    else:
        rate = inp_rate_mitig_base_nominal * 0.25
    P.rates = rate


def reset_input(P, trial_jitter_factor):
    """
    Terug naar normaal (met de trial-jitter).
    """
    P.rates = inp_rate_baseline_nominal * trial_jitter_factor


def mitig_poisoning(Sinp):
    """
    Gewichten 'clippen' om de backdoors onschadelijk te maken.
    """
    if MITIGATION_LEVEL == "low":
        max_w = 0.8
    elif MITIGATION_LEVEL == "medium":
        max_w = 0.5
    else:
        max_w = 0.3
    Sinp.w[:] = np.minimum(Sinp.w[:], max_w)


def mitig_drift(G, S):
    """
    Opnieuw kalibreren van tau en gewichten om drift tegen te gaan.
    """
    G.tau[:] = clip(G.tau[:], 10 * ms, 50 * ms)

    if MITIGATION_LEVEL == "low":
        S.w[:] = clip(S.w[:], 0, 1.2)
    elif MITIGATION_LEVEL == "medium":
        S.w[:] = clip(S.w[:] * 0.8, 0, 1.0)
    else:
        S.w[:] = clip(S.w[:] * 0.6, 0, 1.0)


def mitig_svce(G):
    """
    Harde reset voor sVCE.
    Zet voltages terug en geneest nodes (afhankelijk van level).
    """
    G.EL[:] = EL_val
    G.tau[:] = clip(G.tau[:], 10 * ms, 30 * ms)

    state = G.seir_state[:]
    if MITIGATION_LEVEL == "medium":
        state[state == 1] = 0  # E -> S
    elif MITIGATION_LEVEL == "high":
        state[(state == 1) | (state == 2)] = 0  # E en I -> S (agressieve clean-up)
    G.seir_state[:] = state


################################################################################
# Eén trial draaien
################################################################################

def run_single_trial(scenario="overstimulation",
                     seed_val=42, trial_id=0, verbose=True):
    """
    Draait de volledige simulatie voor 1 scenario.
    Baseline -> Attack -> Mitigation.
    Geeft een dict terug met alle verzamelde metrics.
    """
    start_scope()

    # Zorgen dat niet elke run exact identiek is
    trial_jitter_factor = np.random.uniform(0.8, 1.2)

    net_objs = build_network(seed_val=seed_val,
                             trial_jitter_factor=trial_jitter_factor)
    G = net_objs["G"]
    P = net_objs["P"]
    S = net_objs["S"]
    Sinp = net_objs["Sinp"]
    M_sp = net_objs["M_sp"]
    M_rate = net_objs["M_rate"]
    homeo = net_objs["homeostasis"]
    jitter_op = net_objs["background_rate_jitter"]

    ops = [homeo, jitter_op]

    # Koppel de juiste attack operator aan het scenario
    if scenario == "overstimulation":
        overstim_op = make_overstim_pulse_operator(G)
        ops.append(overstim_op)
    elif scenario == "poisoning":
        poison_targets = np.random.choice(
            np.arange(N),
            size=int(N * poison_target_frac),
            replace=False
        )
        poison_op = make_poisoning_operator(Sinp, poison_targets)
        ops.append(poison_op)
    elif scenario == "drift":
        drift_op = make_drift_operator(G, S)
        ops.append(drift_op)
    elif scenario == "svce":
        G.seir_state[:] = 0
        init_I = np.random.choice(
            np.arange(N),
            size=max(1, int(N * svce_initial_frac_I)),
            replace=False
        )
        G.seir_state[init_I] = 2  # Patient zero aanwijzen
        svce_op = make_svce_operator(G, S)
        ops.append(svce_op)

    net = Network(collect())
    for op in ops:
        net.add(op)

    # --------------------------- Fase 1: Baseline ---------------------------
    homeo.active = True
    reset_input(P, trial_jitter_factor)

    if verbose:
        print(f"[{scenario.upper()} | TRIAL {trial_id}] Phase 1/3: BASELINE")

    net.run(duration_baseline, report=None)

    # Metrics opslaan van baseline
    t0_base = 0 * ms
    t1_base = duration_baseline
    met_base = metrics_from_spikes(M_rate, bin_size=stat_bin,
                                   t_start=t0_base, t_stop=t1_base)
    base_hetero = compute_heterogeneity(M_sp, t0_base, t1_base, N)
    base_slope = compute_rate_slope(M_rate, t0_base, t1_base)

    # --------------------------- Fase 2: Attack -------------------------
    if verbose:
        desc = {
            "baseline": "CONTROL WINDOW (no attack)",
            "overstimulation": "ATTACK (DDoS-like)",
            "poisoning": "ATTACK (Backdoor)",
            "drift": "ATTACK (Supply-chain drift)",
            "svce": "ATTACK (Virus spread)",
        }.get(scenario, "ATTACK")
        print(f"[{scenario.upper()} | TRIAL {trial_id}] Phase 2/3: {desc}")

    # Homeostase uit tijdens attack, anders vecht het netwerk terug
    homeo.active = False

    if scenario == "overstimulation":
        P.rates = inp_rate_attack_overstim
        net.run(duration_attack, report=None)
    elif scenario == "baseline":
        reset_input(P, trial_jitter_factor)
        net.run(duration_attack, report=None)
    else:
        reset_input(P, trial_jitter_factor)
        net.run(duration_attack, report=None)

    t0_atk = duration_baseline
    t1_atk = duration_baseline + duration_attack
    met_atk = metrics_from_spikes(M_rate, bin_size=stat_bin,
                                  t_start=t0_atk, t_stop=t1_atk)
    atk_hetero = compute_heterogeneity(M_sp, t0_atk, t1_atk, N)
    atk_slope = compute_rate_slope(M_rate, t0_atk, t1_atk)

    # Opsplitsen in early/late attack metrics (sommige attacks groeien langzaam)
    t_mid_atk = t0_atk + (t1_atk - t0_atk) / 2.0
    met_atk_early = metrics_from_spikes(M_rate, bin_size=stat_bin,
                                        t_start=t0_atk, t_stop=t_mid_atk)
    met_atk_late = metrics_from_spikes(M_rate, bin_size=stat_bin,
                                       t_start=t_mid_atk, t_stop=t1_atk)

    # Bereken detection delay
    if scenario == "baseline":
        det_delay, det_threshold = np.nan, np.nan
    else:
        det_delay, det_threshold = compute_detection_delay(
            M_rate, t0_atk, t1_atk,
            base_mean=met_base["mean_rate"],
            base_burstiness=met_base["burstiness"],
            k_sigma=3.0
        )

    # --------------------------- Fase 3: Mitigation ---------------------
    if verbose:
        mit_desc = f"MITIGATION (level={MITIGATION_LEVEL})"
        if scenario == "baseline":
            mit_desc = "CONTROL MITIGATION WINDOW"
        print(f"[{scenario.upper()} | TRIAL {trial_id}] Phase 3/3: {mit_desc}")

    homeo.active = True

    # Pas tegenmaatregelen toe
    apply_rate_limiting(P)
    mitig_poisoning(Sinp)
    mitig_drift(G, S)
    mitig_svce(G)

    net.run(duration_mitig, report=None)

    t0_mit = duration_baseline + duration_attack
    t1_mit = duration_baseline + duration_attack + duration_mitig

    # We kijken alleen naar de staart van de mitigatie voor recovery succes
    rec_start = t1_mit - RECOVERY_EVAL_WINDOW
    if rec_start < t0_mit:
        rec_start = t0_mit

    met_mit = metrics_from_spikes(M_rate, bin_size=stat_bin,
                                  t_start=rec_start, t_stop=t1_mit)
    mit_hetero = compute_heterogeneity(M_sp, rec_start, t1_mit, N)
    mit_slope = compute_rate_slope(M_rate, rec_start, t1_mit)

    # SEIR stats opslaan als we sVCE draaien
    svce_stats = dict(frac_S=np.nan, frac_E=np.nan,
                      frac_I=np.nan, frac_R=np.nan)
    if scenario == "svce":
        states = np.array(G.seir_state[:])
        total = len(states)
        if total > 0:
            svce_stats = dict(
                frac_S=np.sum(states == 0) / total,
                frac_E=np.sum(states == 1) / total,
                frac_I=np.sum(states == 2) / total,
                frac_R=np.sum(states == 3) / total,
            )

    # Delta's berekenen voor makkelijke analyse in CSV later
    base_mean = met_base["mean_rate"]
    atk_mean = met_atk["mean_rate"]
    mit_mean = met_mit["mean_rate"]

    base_ent = met_base["entropy"]
    atk_ent = met_atk["entropy"]
    mit_ent = met_mit["entropy"]

    delta_atk_mean = atk_mean - base_mean
    delta_mit_mean = mit_mean - base_mean
    delta_atk_ent = atk_ent - base_ent
    delta_mit_ent = mit_ent - base_ent

    if base_mean > 1e-9:
        pct_atk_mean = 100.0 * (atk_mean - base_mean) / base_mean
        pct_mit_mean = 100.0 * (mit_mean - base_mean) / base_mean
    else:
        pct_atk_mean = np.nan
        pct_mit_mean = np.nan

    # Alles in een flat dict proppen voor de DataFrame
    res = {
        "trial_id": trial_id,
        "scenario": scenario,
        # baseline
        "base_mean_rate": base_mean,
        "base_burstiness": met_base["burstiness"],
        "base_synchrony": met_base["synchrony"],
        "base_entropy": base_ent,
        "base_heterogeneity": base_hetero,
        "base_slope": base_slope,
        # attack
        "atk_mean_rate": atk_mean,
        "atk_burstiness": met_atk["burstiness"],
        "atk_synchrony": met_atk["synchrony"],
        "atk_entropy": atk_ent,
        "atk_heterogeneity": atk_hetero,
        "atk_slope": atk_slope,
        # mitigation
        "mit_mean_rate": mit_mean,
        "mit_burstiness": met_mit["burstiness"],
        "mit_synchrony": met_mit["synchrony"],
        "mit_entropy": mit_ent,
        "mit_heterogeneity": mit_hetero,
        "mit_slope": mit_slope,
        # delta's
        "delta_atk_mean_rate": delta_atk_mean,
        "delta_atk_entropy": delta_atk_ent,
        "delta_mit_mean_rate": delta_mit_mean,
        "delta_mit_entropy": delta_mit_ent,
        # percentage
        "pct_atk_mean_rate_change": pct_atk_mean,
        "pct_mit_mean_rate_change": pct_mit_mean,
        # sVCE extras
        "svce_frac_S": svce_stats["frac_S"],
        "svce_frac_E": svce_stats["frac_E"],
        "svce_frac_I": svce_stats["frac_I"],
        "svce_frac_R": svce_stats["frac_R"],
        # nieuwe dynamische features
        "atk_early_mean_rate": met_atk_early["mean_rate"],
        "atk_late_mean_rate": met_atk_late["mean_rate"],
        "atk_early_entropy": met_atk_early["entropy"],
        "atk_late_entropy": met_atk_late["entropy"],
        "detection_delay": det_delay,
        "detection_threshold": det_threshold,
        "trial_jitter_factor": trial_jitter_factor,
    }

    if verbose:
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, (float, np.floating)) else x

        print("  Baseline:",
              ", ".join(f"{k}={fmt(v)}" for k, v in met_base.items()))
        print("  Attack  :",
              ", ".join(f"{k}={fmt(v)}" for k, v in met_atk.items()))
        print("  Mitig   :",
              ", ".join(f"{k}={fmt(v)}" for k, v in met_mit.items()))
        print(f"  Heterogeneity: base={fmt(base_hetero)}, "
              f"atk={fmt(atk_hetero)}, mit={fmt(mit_hetero)}")
        print(f"  Slopes: base={fmt(base_slope)}, atk={fmt(atk_slope)}, mit={fmt(mit_slope)}")
        print(f"  Detection delay (s): {fmt(det_delay)} "
              f"(thr={fmt(det_threshold)})")
        if scenario == "svce":
            print("  sVCE fractions:", svce_stats)

    return res


################################################################################
# Multiprocessing worker
################################################################################

def run_trial_worker(args):
    """
    Simpele wrapper voor MP.
    """
    scenario, trial_id, seed_val = args

    # Codegen target uit env plukken, anders crasht MP soms op globals
    from brian2 import prefs as _prefs
    codegen_target = os.environ.get("BRIAN2_CODEGEN_TARGET", "numpy")
    _prefs.codegen.target = codegen_target

    return run_single_trial(
        scenario=scenario,
        seed_val=seed_val,
        trial_id=trial_id,
        verbose=False
    )


################################################################################
# Quick scenario summary
################################################################################

def quick_scenario_summary(df):
    """
    Print even een snelle samenvatting in de console.
    Scheelt weer CSV openen als je gewoon wilt weten of het gewerkt heeft.
    """
    if "scenario" not in df.columns:
        return

    print("\n=== Quick scenario summary (core metrics) ===\n")

    core_cols = [
        "atk_mean_rate",
        "atk_burstiness",
        "atk_entropy",
        "atk_synchrony",
        "delta_atk_mean_rate",
        "delta_atk_entropy",
        "delta_mit_mean_rate",
        "delta_mit_entropy",
        "atk_heterogeneity",
        "mit_heterogeneity",
        "atk_slope",
        "mit_slope",
        "detection_delay",
        "atk_early_mean_rate",
        "atk_late_mean_rate",
        "svce_frac_S",
        "svce_frac_E",
        "svce_frac_I",
        "svce_frac_R",
    ]

    for scen in df["scenario"].unique():
        sub = df[df["scenario"] == scen]
        print(f"--- Scenario: {scen} ---")
        for col in core_cols:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            if len(vals) == 0:
                continue
            print(f"{col}: mean={vals.mean():.3f}, std={vals.std():.3f}")
        print("")


################################################################################
# Main
################################################################################

if __name__ == "__main__":
    import argparse

    # CLI opties definieren
    parser = argparse.ArgumentParser(
        description="Living Crime Scene SNN simulation with attack & mitigation levels (v7)."
    )
    parser.add_argument(
        "-m", "--mitigation-level",
        choices=["low", "medium", "high"],
        default="high",
        help="Strength of mitigation response (default: high)",
    )
    parser.add_argument(
        "-a", "--attack-intensity",
        choices=["low", "medium", "high"],
        default="high",
        help="Intensity of attacks (default: high)",
    )
    parser.add_argument(
        "--codegen",
        choices=["numpy", "cython"],
        default="numpy",
        help="Brian2 codegen target (default: numpy)",
    )
    parser.add_argument(
        "-n", "--trials",
        type=int,
        default=3,
        help="Number of trials per scenario (default: 3)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count)",
    )

    args = parser.parse_args()

    # Settings doorvoeren
    MITIGATION_LEVEL = args.mitigation_level
    configure_attack_intensity(args.attack_intensity)

    # Codegen target configureren + naar env pushen (voor de child-processen)
    CODEGEN_TARGET = args.codegen
    prefs.codegen.target = CODEGEN_TARGET
    os.environ["BRIAN2_CODEGEN_TARGET"] = CODEGEN_TARGET

    print(f"=== Mitigation level set to: {MITIGATION_LEVEL.upper()} ===")
    print(f"=== Attack intensity set to: {ATTACK_INTENSITY.upper()} ===")
    print(f"=== Codegen target          : {CODEGEN_TARGET} ===")
    print("=== Scenario definitions ===")
    print("  baseline        : healthy control; phase 2 & 3 zijn tijd-/controlewindows zonder aanval")
    print("  overstimulation : DDoS-achtige hoge input tijdens attack-fase")
    print("  poisoning       : data poisoning / backdoor in feedforward-paden")
    print("  drift           : langzame supply-chain drift (tau, weights)")
    print("  svce            : SEIR-achtige virale spread over de graaf")
    print("=============================================")

    scenarios = ["baseline", "overstimulation", "poisoning", "drift", "svce"]
    n_trials = args.trials

    # Jobs voorbereiden voor MP
    all_jobs = []
    base_seed = 123
    for scen in scenarios:
        for k in range(n_trials):
            seed_val = base_seed + k
            all_jobs.append((scen, k, seed_val))

    total_trials = len(all_jobs)
    print(f"--- Starting simulation of {total_trials} total trials ---")

    start_time = time.time()
    results = []

    n_jobs = args.jobs or mp.cpu_count()

    # Pool opstarten. imap_unordered is sneller omdat we de volgorde toch niet boeit.
    with mp.Pool(processes=n_jobs) as pool:
        for idx, res in enumerate(pool.imap_unordered(run_trial_worker, all_jobs), start=1):
            results.append(res)

            elapsed_sec = time.time() - start_time
            if idx > 0:
                avg = elapsed_sec / idx
                remaining = total_trials - idx
                eta_sec = int(avg * remaining)
                eta_str = str(timedelta(seconds=eta_sec))
            else:
                eta_str = "Calculating..."

            elapsed_str = str(timedelta(seconds=int(elapsed_sec)))
            print(f">>> GLOBAL PROGRESS (MP): {idx}/{total_trials} | Elapsed={elapsed_str} | ETA={eta_str}")

    df = pd.DataFrame(results)
    out_csv = "living_crime_scene_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nResultaten weggeschreven naar {out_csv}")

    quick_scenario_summary(df)

    print("[DONE] Simulatie afgerond.")