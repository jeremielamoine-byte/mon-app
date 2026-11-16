import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Simulateur de Flotte Électrique", page_icon="⚡", layout="wide"
)

st.title("⚡ Simulateur de Flotte de Camions Électriques")
st.markdown(
    "Modèle d'optimisation glouton avec suivi des ports et analyse des"
    " performances."
)
st.markdown("---")

# --- 2. PARAMÈTRES D'ENTRÉE (DANS LA BARRE LATÉRALE) ---
st.sidebar.header("🎛️ Paramètres de la Station")

target_power_kw = st.sidebar.slider(
    "Puissance Cible Réseau (kW)",
    min_value=50.0,
    max_value=500.0,
    value=200.0,
    step=10.0,
)
max_simultaneous = st.sidebar.slider(
    "Nombre de Bornes Simultanées", min_value=1, max_value=8, value=4
)
# ====================================================
# FLOTTE
# ====================================================

st.subheader("🚚 Configuration de la Flotte")

default_df = pd.DataFrame({
    "Camion": [
        "Camion 01",
        "Camion 02",
        "Camion 03",
        "Camion 04",
        "Camion 05",
        "Camion 06",
        "Camion 07",
        "Camion 08",
        "Camion 09",
        "Camion 10",
    ],
    "Arrivée": [
        6.0,
        6.0,
        10.0,
        12.5,
        13.0,
        15.0,
        16.5,
        18.0,
        20.0,
        22.0,
    ],
    "Départ": [
        10.0,
        12.0,
        15.0,
        16.5,
        18.0,
        19.0,
        21.5,
        22.0,
        29.0,
        30.0,
    ],
    "Besoin kWh": [
        600,
        600,
        250,
        160,
        300,
        200,
        280,
        150,
        350,
        400,
    ]
})

fleet_df = st.data_editor(
    default_df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={

        "Camion": st.column_config.TextColumn(
            "Camion"
        ),

        "Arrivée": st.column_config.NumberColumn(
            "Arrivée",
            min_value=0.0,
            max_value=48.0,
            step=0.25,
            format="%.2f"
        ),

        "Départ": st.column_config.NumberColumn(
            "Départ",
            min_value=0.0,
            max_value=48.0,
            step=0.25,
            format="%.2f"
        ),

        "Besoin kWh": st.column_config.NumberColumn(
            "Besoin kWh",
            min_value=0,
            step=10
        ),
    }
)
c1, c2 = st.columns(2)

nb_camions = len(
    fleet_df.dropna(
        subset=["Camion"]
    )
)
c1.metric(
    "Camions",
    nb_camions
)

c2.metric(
    "Besoin Total",
    f"{fleet_df['Besoin kWh'].sum():,.0f} kWh"
)
vehicles = []

for _, row in fleet_df.iterrows():

    try:

        vehicles.append({
            "id": str(row["Camion"]),
            "arr": float(row["Arrivée"]),
            "dep": float(row["Départ"]),
            "req": float(row["Besoin kWh"]),
        })

    except Exception:
        pass

if not vehicles:
    st.error("⚠️ Aucun véhicule valide.")
    st.stop()


# --- 3. LOGIQUE DE SIMULATION ---
time_steps = [i * 0.25 for i in range(96)]
n_steps = len(time_steps)
n_veh = len(vehicles)

presence = np.zeros((n_veh, n_steps))
energy_req = np.array([v["req"] for v in vehicles])
energy_delivered = np.zeros(n_veh)
cumulative_energy_matrix = np.zeros((n_veh, n_steps))

for v_idx, v in enumerate(vehicles):
    arr = float(v["arr"])
    dep = float(v["dep"])
    for t_idx, t in enumerate(time_steps):
        # présence 24h/24
        if abs(arr - dep) < 0.001:
            presence[v_idx, t_idx] = 1
        # format lendemain exprimé >24h
        elif dep >= 24:
            dep_norm = dep - 24
            if t >= arr or t < dep_norm:
                presence[v_idx, t_idx] = 1
        # même journée
        elif arr < dep:
            if arr <= t < dep:
                presence[v_idx, t_idx] = 1
        # traverse minuit
        else:
            if t >= arr or t < dep:
                presence[v_idx, t_idx] = 1

power_matrix = np.zeros((n_veh, n_steps))
stall_assignment = np.zeros((n_steps, max_simultaneous), dtype=int)
truck_stalls = {v_idx: set() for v_idx in range(n_veh)}

manipulations = 0
prev_stalls = np.zeros(max_simultaneous, dtype=int)

for t_idx, t in enumerate(time_steps):
  present_needing_charge = [
      v_idx
      for v_idx in range(n_veh)
      if presence[v_idx, t_idx] == 1
      and energy_delivered[v_idx] < energy_req[v_idx]
  ]

  if not present_needing_charge:
    prev_stalls = np.zeros(max_simultaneous, dtype=int)
    for v_idx in range(n_veh):
      cumulative_energy_matrix[v_idx, t_idx] = energy_delivered[v_idx]
    continue

  urgency = []
  for v_idx in present_needing_charge:
    rem_energy = energy_req[v_idx] - energy_delivered[v_idx]
    dep = vehicles[v_idx]["dep"]
    time_left = (dep - t) if dep > t else (dep + 24 - t)
    urgency_score = rem_energy / max(0.25, time_left)
    urgency.append((urgency_score, -dep, v_idx))

  urgency.sort(reverse=True)
  top_urgent = {
      v_idx
      for _, _, v_idx in urgency[
          :min(max_simultaneous, len(urgency))
      ]
  }
  current_stalls = np.zeros(
      max_simultaneous,
      dtype=int
  )
  assigned_vehicles = set()
  for stall_i in range(max_simultaneous):
      old_v = prev_stalls[stall_i]
      if old_v != 0:
          v_idx = old_v - 1
          if (
              v_idx in present_needing_charge
              and v_idx in top_urgent
          ):
              current_stalls[stall_i] = old_v
              assigned_vehicles.add(v_idx)
  unassigned_urgent = [
      v_idx
      for _, _, v_idx in urgency
      if v_idx not in assigned_vehicles
  ]
  for stall_i in range(max_simultaneous):
      if (
          current_stalls[stall_i] == 0
          and unassigned_urgent
      ):
          next_v = unassigned_urgent.pop(0)
          current_stalls[stall_i] = (
              next_v + 1
          )
          assigned_vehicles.add(next_v)

  if t_idx > 0:
    for stall_i in range(max_simultaneous):
      if prev_stalls[stall_i] != current_stalls[stall_i]:
        manipulations += 1
  prev_stalls = current_stalls.copy()
  stall_assignment[t_idx] = current_stalls

  for stall_i, v_val in enumerate(current_stalls):
    if v_val != 0:
      truck_stalls[v_val - 1].add(stall_i + 1)

  plugged_indices = [v - 1 for v in current_stalls if v != 0]

  if plugged_indices:
    remaining_power = target_power_kw
    allocations = {v_idx: 0.0 for v_idx in plugged_indices}
    scores = {
        v_idx: max(
            0.1,
            (energy_req[v_idx] - energy_delivered[v_idx])
            / max(
                0.25,
                (
                    vehicles[v_idx]["dep"] - t
                    if vehicles[v_idx]["dep"] > t
                    else vehicles[v_idx]["dep"] + 24 - t
                ),
            ),
        )
        for v_idx in plugged_indices
    }

    while remaining_power >= 50.0:
      assigned_in_round = False
      sorted_plugged = sorted(
          plugged_indices, key=lambda x: scores[x], reverse=True
      )
      for v_idx in sorted_plugged:
        rem_needed_kw = (
            energy_req[v_idx] - energy_delivered[v_idx]
        ) / 0.25 - allocations[v_idx]
        if rem_needed_kw > 0 and remaining_power >= 50.0:
          allocations[v_idx] += 50.0
          remaining_power -= 50.0
          assigned_in_round = True
      if not assigned_in_round:
        break

    for v_idx in plugged_indices:
      actual_power = allocations[v_idx]
      if actual_power > 0:
        power_matrix[v_idx, t_idx] = actual_power
        energy_delivered[v_idx] += actual_power * 0.25

  for v_idx in range(n_veh):
    cumulative_energy_matrix[v_idx, t_idx] = energy_delivered[v_idx]

total_req = energy_req.sum()
total_liv = energy_delivered.sum()
taux_recup_global = (total_liv / total_req) * 100.0 if total_req > 0 else 0
total_manip_time_hours = manipulations * (5.0 / 60.0)
fleet_total_power = power_matrix.sum(axis=0)
peak_power = float(
    fleet_total_power.max()
)
avg_power = float(
    fleet_total_power.mean()
)
active_ports = np.count_nonzero(
    power_matrix,
    axis=0
)
max_ports_used = int(
    active_ports.max()
)
active_periods = active_ports[
    active_ports > 0
]
avg_ports_used = (
    active_periods.mean()
    if len(active_periods) > 0
    else 0
)
occupation_rate = (
    np.count_nonzero(stall_assignment)
    /
    (
        n_steps
        * max_simultaneous
    )
) * 100
peak_time_index = np.argmax(
    fleet_total_power
)
peak_time = time_steps[
    peak_time_index
]

# --- 4. RESTITUTION VISUELLE AVEC ALERTE COULEUR ---
st.subheader("🎯 Indicateurs Clés de Performance")

col1, col2, col3, col4 = st.columns(4)

col5, col6, col7, col8 = st.columns(4)
# Condition couleur pour le taux de récupération (< 100% en rouge, sinon normal/vert)
if taux_recup_global < 100.0:
  with col1:
    st.markdown(
        f"**Taux de Récupération**\n\n<h3"
        f' style="color:#e74c3c;">{taux_recup_global:.1f}% ⚠️</h3>',
        unsafe_allow_html=True,
    )
else:
  col1.metric("Taux de Récupération", f"{taux_recup_global:.1f}%", "Optimal ✨")

col2.metric("Énergie Livrée", f"{total_liv:.1f} kWh", f"Sur {total_req} kWh")
col3.metric("Manipulations", f"{manipulations} actions", "Coût : 5 min/action")
col4.metric("Temps Perdu", f"{total_manip_time_hours:.1f} heures")
col5.metric(
    "Occupation CCS",
    f"{occupation_rate:.1f}%"
)

col6.metric(
    "Ports Max Utilisés",
    str(max_ports_used)
)

col7.metric(
    "Ports Moyens",
    f"{avg_ports_used:.1f}"
)

col8.metric(
    "Pic Réel",
    f"{peak_power:.0f} kW"
)
st.markdown("---")

# --- 5. AFFICHAGE DES GRAPHIQUES ---

st.subheader("🚚 Flotte")
fleet_table = []
for v_idx, v in enumerate(vehicles):
    req = v["req"]
    delivered = energy_delivered[v_idx]
    pct = (
        delivered / req * 100
        if req > 0
        else 0
    )
    fleet_table.append({
        "Camion": v["id"],
        "Arrivée": v["arr"],
        "Départ": v["dep"],
        "Énergie requise": round(req, 1),
        "Énergie livrée": round(delivered, 1),
        "Couverture (%)": round(pct, 1)
    })
st.dataframe(
    fleet_table,
    use_container_width=True
)

st.subheader("📈 1. Suivi Global de la Station")
fig1, ax1 = plt.subplots(figsize=(12, 4))
fleet_total_power = power_matrix.sum(axis=0)
ax1.step(
    time_steps,
    fleet_total_power,
    where="mid",
    label="Puissance Totale Station (kW)",
    color="#2ecc71",
    linewidth=2.5,
)
ax1.axhline(
    y=target_power_kw,
    color="#e74c3c",
    linestyle="--",
    label=f"Plafond Réseau ({target_power_kw} kW)",
)
ax1.set_xlabel("Heure de la journée")
ax1.set_ylabel("Puissance (kW)")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend()
st.pyplot(fig1)
st.subheader("📊 Occupation des Ports")
fig_occ, ax_occ = plt.subplots(
    figsize=(12, 4)
)
ax_occ.step(
    time_steps,
    active_ports,
    where="mid",
    linewidth=2,
    color="#3498db"
)
ax_occ.axhline(
    y=max_simultaneous,
    color="#e74c3c",
    linestyle="--",
    label="Ports installés"
)
ax_occ.set_ylabel("Ports actifs")
ax_occ.set_xlabel("Heure")
ax_occ.grid(True)
ax_occ.legend()
st.pyplot(fig_occ)
st.subheader("🚚 2. Suivi Individuel par Camion")
# Calcul dynamique du nombre de lignes de graphiques nécessaires selon le nombre de camions
n_rows = (n_veh + 2) // 3
fig2, axes2 = plt.subplots(n_rows, 3, figsize=(15, 3.5 * n_rows), sharex=True)
axes2 = np.atleast_1d(axes2).flatten()

for v_idx, v in enumerate(vehicles):
  ax = axes2[v_idx]
  cumul_profile = cumulative_energy_matrix[v_idx]
  req = v["req"]
  ax.plot(
      time_steps,
      cumul_profile,
      color="#2980b9",
      linewidth=2.5,
      label="Énergie Livrée",
  )
  ax.axhline(y=req, color="#e74c3c", linestyle=":", label=f"Objectif ({req})")
  pct = (energy_delivered[v_idx] / req) * 100.0 if req > 0 else 0
  ax.set_title(f"{v['id']} ({pct:.1f}%)", fontsize=10, fontweight="bold")
  ax.grid(True, linestyle=":", alpha=0.6)
  ax.set_xlim(0, 24)

# Nettoyage des sous-graphiques vides restants
for j in range(len(vehicles), len(axes2)):
  fig2.delaxes(axes2[j])

plt.tight_layout()
st.pyplot(fig2)
st.subheader(
    "🔥 Heatmap de Recharge"
)
fig_heat, ax_heat = plt.subplots(
    figsize=(14,6)
)
im = ax_heat.imshow(
    power_matrix,
    aspect="auto",
    cmap="YlOrRd"
)
ax_heat.set_xlabel(
    "Pas de temps (15 min)"
)
ax_heat.set_ylabel(
    "Camions"
)
ax_heat.set_title(
    "Répartition des puissances"
)
plt.colorbar(
    im,
    ax=ax_heat,
    label="Puissance (kW)"
)
st.pyplot(fig_heat)

st.subheader("🔌 3. Occupation des Points de Charge")
fig3, axes3 = plt.subplots(
    max_simultaneous, 1, figsize=(12, 2.0 * max_simultaneous), sharex=True
)
if max_simultaneous == 1:
  axes3 = [axes3]
stall_colors = ["#e67e22", "#9b59b6", "#1abc9c", "#e74c3c", "#3498db"]

for stall_i in range(max_simultaneous):
  ax = axes3[stall_i]
  stall_usage = stall_assignment[:, stall_i]
  color_choice = stall_colors[stall_i % len(stall_colors)]
  ax.step(
      time_steps, stall_usage, where="mid", color=color_choice, linewidth=2.5
  )
  ax.set_yticks(range(0, n_veh + 1))
  ax.set_yticklabels(["Libre"] + [veh["id"] for veh in vehicles], fontsize=8)
  ax.set_title(
      f"Point de Charge #{stall_i + 1}", fontsize=10, fontweight="bold"
  )
  ax.grid(True, linestyle=":", alpha=0.6)
  ax.set_xlim(0, 24)

plt.xlabel("Heure de la journée")
plt.tight_layout()
st.pyplot(fig3)

st.markdown("---")
st.subheader(
    "🏆 Analyse de l'Infrastructure"
)
util_power_ratio = (
    peak_power
    / target_power_kw
    * 100
)
util_ports_ratio = (
    max_ports_used
    / max_simultaneous
    * 100
)
unused_power = (
    target_power_kw
    - peak_power
)
unused_ports = (
    max_simultaneous
    - max_ports_used
)
r1, r2, r3, r4 = st.columns(4)
r1.metric(
    "Pic Réseau Réel",
    f"{peak_power:.0f} kW"
)
r2.metric(
    "Utilisation Réseau",
    f"{util_power_ratio:.1f}%"
)
r3.metric(
    "Simultanéité Max",
    max_ports_used
)
r4.metric(
    "Utilisation CCS",
    f"{util_ports_ratio:.1f}%"
)
r5, r6 = st.columns(2)
r5.metric(
    "Puissance Inutilisée",
    f"{unused_power:.0f} kW"
)
r6.metric(
    "Ports Inutilisés",
    int(unused_ports)
)
st.info(
    f"Pic observé à {peak_time:.2f} h "
    f"avec {peak_power:.0f} kW."
)

