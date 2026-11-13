"""
Simulateur de Flotte - Modèle Glouton Stable avec Suivi des Ports & Graphiques
- Priorisation par urgence & départage par heure de départ
- Utilisation maximale des bornes disponibles sans échange de port en cours de charge
- Minimisation des manipulations physiques (coût de 5 min par action)
- Génération des graphiques : Global, par Camion, et par Point de Charge
"""

import numpy as np
import matplotlib.pyplot as plt

def run_pure_python_simulation(target_power_kw=200.0, max_simultaneous=4):
    vehicles = [
        {"id": "Camion 01", "arr": 6.0,  "dep": 10.0, "req": 600},
        {"id": "Camion 02", "arr": 6.0,  "dep": 12.0, "req": 600},
        {"id": "Camion 03", "arr": 10.0, "dep": 15.0, "req": 250},
        {"id": "Camion 04", "arr": 12.5, "dep": 16.5, "req": 160},
        {"id": "Camion 05", "arr": 13.0, "dep": 18.0, "req": 300},
        {"id": "Camion 06", "arr": 15.0, "dep": 19.0, "req": 200},
        {"id": "Camion 07", "arr": 16.5, "dep": 21.5, "req": 280},
        {"id": "Camion 08", "arr": 18.0, "dep": 22.0, "req": 150},
        {"id": "Camion 09", "arr": 20.0, "dep": 29.0, "req": 350},
        {"id": "Camion 10", "arr": 22.0, "dep": 30.0, "req": 400},
    ]
    
    time_steps = [i * 0.25 for i in range(96)] # 96 pas de 15 min sur 24h
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
                if arr <= t < dep or arr <= t + 24 < dep: presence[v_idx, t_idx] = 1

    power_matrix = np.zeros((n_veh, n_steps))
    stall_assignment = np.zeros((n_steps, max_simultaneous), dtype=int)
    truck_stalls = {v_idx: set() for v_idx in range(n_veh)}
    
    manipulations = 0
    prev_stalls = np.zeros(max_simultaneous, dtype=int)

    # Boucle temporelle (quart-horaire)
    for t_idx, t in enumerate(time_steps):
        present_needing_charge = [
            v_idx for v_idx in range(n_veh) 
            if presence[v_idx, t_idx] == 1 and energy_delivered[v_idx] < energy_req[v_idx]
        ]
        
        if not present_needing_charge:
            prev_stalls = np.zeros(max_simultaneous, dtype=int)
            for v_idx in range(n_veh):
                cumulative_energy_matrix[v_idx, t_idx] = energy_delivered[v_idx]
            continue
            
        # Calcul du score d'urgence avec départage par heure de départ (-dep)
        urgency = []
        for v_idx in present_needing_charge:
            rem_energy = energy_req[v_idx] - energy_delivered[v_idx]
            dep = vehicles[v_idx]["dep"]
            time_left = (dep - t) if dep > t else (dep + 24 - t)
            urgency_score = rem_energy / max(0.25, time_left)
            urgency.append((urgency_score, -dep, v_idx))
            
        urgency.sort(reverse=True)
        
        # Attribution des bornes : conservation du port si le camion a encore besoin de charge (pas d'échange inutile)
        current_stalls = np.zeros(max_simultaneous, dtype=int)
        assigned_vehicles = set()
        
        for stall_i in range(max_simultaneous):
            old_v = prev_stalls[stall_i]
            if old_v != 0:
                v_idx = old_v - 1
                if v_idx in present_needing_charge:
                    current_stalls[stall_i] = old_v
                    assigned_vehicles.add(v_idx)
                    
        # Remplissage des bornes restants vides par les camions les plus urgents
        unassigned_urgent = [v_idx for _, _, v_idx in urgency if v_idx not in assigned_vehicles]
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

        # Enregistrement des ports utilisés par chaque camion
        for stall_i, v_val in enumerate(current_stalls):
            if v_val != 0:
                truck_stalls[v_val - 1].add(stall_i + 1)

        plugged_indices = [v - 1 for v in current_stalls if v != 0]

        if plugged_indices:
            remaining_power = target_power_kw
            allocations = {v_idx: 0.0 for v_idx in plugged_indices}
            
            scores = {v_idx: max(0.1, (energy_req[v_idx] - energy_delivered[v_idx]) / max(0.25, (vehicles[v_idx]["dep"] - t if vehicles[v_idx]["dep"] > t else vehicles[v_idx]["dep"] + 24 - t))) for v_idx in plugged_indices}

            while remaining_power >= 50.0:
                assigned_in_round = False
                sorted_plugged = sorted(plugged_indices, key=lambda x: scores[x], reverse=True)
                for v_idx in sorted_plugged:
                    rem_needed_kw = (energy_req[v_idx] - energy_delivered[v_idx]) / 0.25 - allocations[v_idx]
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

    # Bilan chiffré
    total_req = energy_req.sum()
    total_liv = energy_delivered.sum()
    taux_recup_global = (total_liv / total_req) * 100.0
    total_manip_time_hours = manipulations * (5.0 / 60.0)

    print("==================================================")
    print(" BILAN DE LA SIMULATION DE FLOTTE")
    print("==================================================")
    print(f"• Puissance Cible Réseau        : {target_power_kw} kW")
    print(f"• Bornes Simultanées Max        : {max_simultaneous}")
    print(f"• Nombre Total de Manipulations : {manipulations} actions")
    print(f"• Temps Perdu en Manutention    : {total_manip_time_hours:.1f} heures (5 min / action)")
    print(f"• Énergie Totale Requise        : {total_req:.1f} kWh")
    print(f"• Énergie Totale Livrée         : {total_liv:.1f} kWh")
    print(f"• Taux de Récupération Global   : {taux_recup_global:.1f}%")
    print("==================================================")
    for v_idx, v in enumerate(vehicles):
        req = energy_req[v_idx]
        liv = energy_delivered[v_idx]
        pct = (liv / req) * 100.0
        stalls_used = sorted(list(truck_stalls[v_idx]))
        stalls_str = ", ".join([f"Borne #{s}" for s in stalls_used])
        print(f"  - {v['id']} ({stalls_str}) : Requis {req} kWh | Livré {liv:.1f} kWh ({pct:.1f}%)")

    # ==========================================
    # AFFICHAGE DES GRAPHIQUES DE SUIVI
    # ==========================================

    # 1. Suivi Global de la Station
    plt.figure(figsize=(12, 4.5))
    fleet_total_power = power_matrix.sum(axis=0)
    plt.step(time_steps, fleet_total_power, where='mid', label="Puissance Totale Station (kW)", color="#2ecc71", linewidth=2.5)
    plt.axhline(y=target_power_kw, color="#e74c3c", linestyle="--", label=f"Plafond Réseau ({target_power_kw} kW)")
    plt.title("1. Suivi Global de la Station (Puissance Totale en kW)", fontsize=12, fontweight="bold")
    plt.xlabel("Heure de la journée", fontsize=10)
    plt.ylabel("Puissance (kW)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # 2. Suivi Individuel par Camion (kWh Cumulés)
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True)
    axes = axes.flatten()
    
    for v_idx, v in enumerate(vehicles):
        ax = axes[v_idx]
        cumul_profile = cumulative_energy_matrix[v_idx]
        req = v['req']
        
        ax.plot(time_steps, cumul_profile, color="#2980b9", linewidth=2.5, label="Énergie Livrée")
        ax.axhline(y=req, color="#e74c3c", linestyle=":", label=f"Objectif ({req} kWh)")
        
        pct = (energy_delivered[v_idx] / req) * 100.0
        ax.set_title(f"{v['id']} | Atteint: {pct:.1f}%", fontsize=10, fontweight="bold")
        ax.set_ylabel("kWh", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlim(0, 30)
        ax.legend(fontsize=8, loc="upper left")
        
    plt.suptitle("2. Suivi Individuel de la Charge par Camion (kWh Cumulés)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # 3. Suivi par Point de Charge (#1 à #4)
    fig, axes = plt.subplots(max_simultaneous, 1, figsize=(14, 2.2 * max_simultaneous), sharex=True)
    if max_simultaneous == 1:
        axes = [axes]
        
    stall_colors = ["#e67e22", "#9b59b6", "#1abc9c", "#e74c3c"]
    
    for stall_i in range(max_simultaneous):
        ax = axes[stall_i]
        stall_usage = stall_assignment[:, stall_i]
        
        ax.step(time_steps, stall_usage, where='mid', color=stall_colors[stall_i], linewidth=2.5)
        ax.set_yticks(range(0, n_veh + 1))
        ax.set_yticklabels(["Libre"] + [veh['id'] for veh in vehicles], fontsize=8)
        ax.set_title(f"Point de Charge #{stall_i + 1} - Occupation & Rotation", fontsize=10, fontweight="bold")
        ax.set_ylabel("Camion", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlim(0, 30)
        
    plt.xlabel("Heure de la journée", fontsize=10)
    plt.suptitle("3. Suivi des Points de Charge (Occupation des Connecteurs #1 à #4)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_pure_python_simulation()
