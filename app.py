import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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

st.sidebar.markdown("---")
st.sidebar.header("🚚 Données des Camions")
st.sidebar.markdown(
    "Format par ligne : `Nom, Arrivée, Départ, Requis (kWh)`"
)

# Zone de texte modifiable directement dans le bandeau de gauche
default_vehicles_text = """Camion 01, 6.0, 10.0, 600
Camion 02, 6.0, 12.0, 600
Camion 03, 10.0, 15.0, 250
Camion 04, 12.5, 16.5, 160
Camion 05, 13.0, 18.0, 300
Camion 06, 15.0, 19.0, 200
Camion 07, 16.5, 21.5, 280
Camion 08, 18.0, 22.0, 150
Camion 09, 20.0, 29.0, 350
Camion 10, 22.0, 30.0, 400"""

vehicles_input_str = st.sidebar.text_area(
    "Modifier la flotte", default_vehicles_text, height=220
)

# Analyse du texte de la barre latérale pour en faire une liste de dictionnaires utilisable
vehicles = []
for line in vehicles_input_str.strip().split("\n"):
  parts = [p.strip() for p in line.split(",")]
  if len(parts) == 4:
    try:
      vehicles.append({
          "id": parts[0],
          "arr": float(parts[1]),
          "dep": float(parts[2]),
          "req": float(parts[3]),
      })
    except ValueError:
      continue

if not vehicles:
  st.error(
      "⚠️ Erreur dans le format des camions dans la barre latérale. Vérifiez"
      " vos virgules."
  )
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
  arr, dep = v["arr"], v["dep"]
  for t_idx, t in enumerate(time_steps):
    if arr <= dep:
      if arr <= t < dep or arr <= t + 24 < dep:
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

  current_stalls = np.zeros(max_simultaneous, dtype=int)
  assigned_vehicles = set()

  for stall_i in range(max_simultaneous):
    old_v = prev_stalls[stall_i]
    if old_v != 0:
      v_idx = old_v - 1
      if v_idx in present_needing_charge:
        current_stalls[stall_i] = old_v
        assigned_vehicles.add(v_idx)

  unassigned_urgent = [
      v_idx for _, _, v_idx in urgency if v_idx not in assigned_vehicles
  ]
  for stall_i in range(max_simultaneous):
    if current_stalls[stall_i] == 0 and unassigned_urgent:
      next_v = unassigned_urgent.pop(0)
      current_stalls[stall_i] = next_v + 1
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

average_power = float(
    fleet_total_power.mean()
)

active_ports = np.count_nonzero(
    power_matrix,
    axis=0
)

max_ports_used = int(
    active_ports.max()
)

avg_ports_used = float(
    active_ports.mean()
)

occupation_rate = (
    np.count_nonzero(stall_assignment)
    /
    (n_steps * max_simultaneous)
) * 100

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
    "🏆 Recommandation Infrastructure"
)
r1, r2, r3, r4 = st.columns(4)
r1.metric(
    "Puissance Réseau",
    f"{target_power_kw:.0f} kW"
)
r2.metric(
    "Bornes Installées",
    max_simultaneous
)
r3.metric(
    "Bornes Réellement Utilisées",
    max_ports_used
)
r4.metric(
    "Occupation Moyenne",
    f"{occupation_rate:.1f}%"
)
st.markdown("---")
st.subheader(
    "💰 Estimation Infrastructure"
)
cost_per_ccs = st.sidebar.number_input(
    "Coût par borne CCS (€)",
    value=12000
)
cost_per_kw = st.sidebar.number_input(
    "Coût par kW installé (€)",
    value=350
)
capex_ccs = (
    max_simultaneous
    * cost_per_ccs
)
capex_power = (
    target_power_kw
    * cost_per_kw
)
capex_total = (
    capex_ccs
    + capex_power
)
c1, c2, c3 = st.columns(3)
c1.metric(
    "CAPEX Réseau",
    f"{capex_power:,.0f} €"
)
c2.metric(
    "CAPEX Bornes",
    f"{capex_ccs:,.0f} €"
)
c3.metric(
    "CAPEX Total",
    f"{capex_total:,.0f} €"
)










