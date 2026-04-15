import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.widgets import Button
import networkx as nx
from matplotlib.collections import LineCollection

matplotlib.use('TkAgg')

# Parametri del Sistema
NUM_AGENTS = 7

# 7 è considerato il numero standard. Non scala mai sotto l'1.0.
SCALE_FACTOR = max(1.0, NUM_AGENTS / 7.0)
print('Fattore di scala: ', SCALE_FACTOR)

ALPHA = ((NUM_AGENTS - 3) * 180) / (NUM_AGENTS - 1)
A = [ALPHA]

# Il raggio si scala dinamicamente
CIRC_RADIUS = 1.5 * SCALE_FACTOR
D = 2 * CIRC_RADIUS * np.sin(np.deg2rad(180 / (NUM_AGENTS - 1)))
DESIRED_DISTANCES = np.zeros((NUM_AGENTS, NUM_AGENTS))

dist = [0, float(D)]

DT = 0.05
MAX_STEPS = 2000
MAX_SPEED = 2.5

# Guadagni e Raggi
K_TARGET = 1.5 * SCALE_FACTOR
K_FORM = 1.5 * SCALE_FACTOR
K_REP = 2.0 * SCALE_FACTOR
K_OBS = 5.0 * SCALE_FACTOR
K_CIRC_OUT = 4.0 * SCALE_FACTOR
K_CIRC_IN = 1.5 * SCALE_FACTOR

REP_RADIUS = 0.8
OBS_INFLUENCE = 0.8
HUBER_DELTA = 1.0
DANGER_OBS = 0.5

CIRC_CENTER_IDX = 0
SATELLITE_IDX = list(range(1, NUM_AGENTS))

# I limiti dell'immagine/mappa si scalano dinamicamente
X_MIN, X_MAX = 0.0, 15.0 * SCALE_FACTOR
Y_MIN, Y_MAX = 0.0, 15.0 * SCALE_FACTOR

MIN_START_GOAL_DIST = 10.0 * SCALE_FACTOR
NUM_OBSTACLES = 8

MAX_DIAG_STEPS = int(2 * SCALE_FACTOR)
print('Numero diagonali: ', min(MAX_DIAG_STEPS*2, NUM_AGENTS -4))
TOPOLOGY_UPDATE_INTERVAL = 40
MIN_FRAMES_BETWEEN_BOUNDS = 40  # 20 frame corrispondono a 1 secondo simulato (se DT=0.05 e intervallo=20ms)
last_bound_frame = 0            # Memoria dell'ultimo frame in cui la topologia è stata aggiornata
WAIT_TIME_SECONDS = 0.4  # Quanti secondi vuoi che si fermi su ogni nodo
WAIT_FRAMES = int(WAIT_TIME_SECONDS / DT)  # Converte i secondi in numero di frame (es. 2.0 / 0.05 = 40)
is_waiting = False       # Stato: True se l'agente sta aspettando sul nodo
wait_start_frame = 0     # Memoria del frame in cui è iniziata l'attesa

# Generazione Ostacoli Casuali
def generate_random_obstacles(num_obs):
    obstacles = []
    while len(obstacles) < num_obs:
        # Dimensioni degli ostacoli scalate proporzionalmente
        w = np.random.uniform(1.5 * SCALE_FACTOR, 3.0 * SCALE_FACTOR)
        h = np.random.uniform(1.5 * SCALE_FACTOR, 3.0 * SCALE_FACTOR)

        # Margine di 1.0 moltiplicato per il fattore di scala rispetto ai bordi
        margin = 1.0 * SCALE_FACTOR
        x = np.random.uniform(X_MIN + margin, X_MAX - w - margin)
        y = np.random.uniform(Y_MIN + margin, Y_MAX - h - margin)

        overlap = False
        for (ox, oy, ow, oh) in obstacles:
            if not (x + w < ox or x > ox + ow or y + h < oy or y > oy + oh):
                overlap = True
                break

        if not overlap:
            obstacles.append((x, y, w, h))

    return obstacles

MAX_SPAWN_ATTEMPTS = 1000  # Quanti tentativi prima di rinunciare e resettare

def generate_safe_position(margin):
    attempts = 0
    while attempts < MAX_SPAWN_ATTEMPTS:
        pt = np.random.uniform([X_MIN + margin, Y_MIN + margin], [X_MAX - margin, Y_MAX - margin])
        safe = True
        for (ox, oy, ow, oh) in OBSTACLES:
            if (ox - margin) <= pt[0] <= (ox + ow + margin) and (oy - margin) <= pt[1] <= (oy + oh + margin):
                safe = False
                break
        if safe:
            return pt.tolist()
        attempts += 1

    # Se esce dal ciclo significa che ha fallito
    return None

while True:
    print("Tentativo di generazione mappa...")

    # Generazione ostacoli
    OBSTACLES = generate_random_obstacles(NUM_OBSTACLES)

    # Prova a piazzare la partenza
    START_POS = generate_safe_position(margin= CIRC_RADIUS)
    if START_POS is None:
        print("Mappa troppo intasata per la partenza. Rigenero...")
        continue  # Ricomincia dall'inizio ricreando gli ostacoli

    # Prova a piazzare l'arrivo
    goal_found = False
    goal_attempts = 0

    while goal_attempts < MAX_SPAWN_ATTEMPTS:
        candidate_goal = generate_safe_position(margin= CIRC_RADIUS/2)

        if candidate_goal is None:
            break  # Mappa intasata, esce dal ciclo interno

        dist_to_start = np.linalg.norm(np.array(candidate_goal) - np.array(START_POS))

        if dist_to_start >= MIN_START_GOAL_DIST:
            GOAL_POS = candidate_goal
            goal_found = True
            break  # Esce dal ciclo interno

        goal_attempts += 1

    # Controllo finale
    if goal_found:
        print("Start e Goal generati con successo!")
        break
    else:
        print("  -> Impossibile trovare un traguardo a distanza sicura. Rigenero...")



# Funzioni Geometriche
def get_closest_point_on_rect(px, py, ox, oy, ow, oh):
    cx = np.clip(px, ox, ox + ow)
    cy = np.clip(py, oy, oy + oh)
    return np.array([cx, cy])


def line_intersects_rect(p1, p2, ox, oy, ow, oh):
    dist_pts = np.linalg.norm(np.array(p2) - np.array(p1))
    samples = int(dist_pts / 0.1)
    for i in range(max(samples, 1) + 1):
        t = i / max(samples, 1)
        x = p1[0] * (1 - t) + p2[0] * t
        y = p1[1] * (1 - t) + p2[1] * t
        if (ox - 0.5) <= x <= (ox + ow + 0.5) and (oy - 0.5) <= y <= (oy + oh + 0.5):
            return True
    return False


# Algoritmo PRM
def generate_prm(start, goal, min_samples=50 , max_samples=500, k_neighbors=4):
    print("Generazione PRM in corso...")
    samples = [start, goal]

    # Inizializziamo subito il Grafo e aggiungiamo Partenza (0) e Arrivo (1)
    G = nx.Graph()
    G.add_node(0, pos=start)
    G.add_node(1, pos=goal)

    node_idx = 2  # Partiamo dall'indice 2 per i nuovi campioni

    # Generazione incrementale
    while node_idx < max_samples + 2:
        # 1. Genera un punto sicuro
        pt = np.random.uniform([X_MIN, Y_MIN], [X_MAX, Y_MAX])
        safe = True
        for (ox, oy, ow, oh) in OBSTACLES:
            if (ox - 0.5) <= pt[0] <= (ox + ow + 0.5) and (oy - 0.5) <= pt[1] <= (oy + oh + 0.5):
                safe = False
                break

        if safe:
            samples.append(pt.tolist())
            G.add_node(node_idx, pos=pt.tolist())

            # 2. Calcola le distanze SOLO rispetto ai nodi già esistenti
            distances = []
            for j in range(node_idx):
                d = np.linalg.norm(np.array(samples[node_idx]) - np.array(samples[j]))
                distances.append((j, d))
            distances.sort(key=lambda x: x[1])

            # 3. Collega il nuovo nodo ai suoi K vicini più prossimi
            for j, d in distances[:k_neighbors]:
                collision = any(
                    line_intersects_rect(samples[node_idx], samples[j], ox, oy, ow, oh) for ox, oy, ow, oh in OBSTACLES)
                if not collision:
                    G.add_edge(node_idx, j, weight=d)

            node_idx += 1

            # --- NOVITÀ: Controllo di Chiusura Anticipata ---
            # Se abbiamo superato il minimo richiesto di campioni...
            if len(samples) >= min_samples + 2:
                # ...controlliamo se i nodi 0 (Start) e 1 (Goal) sono finalmente connessi!
                if nx.has_path(G, 0, 1):
                    print(f"  -> Percorso trovato in anticipo al campione {len(samples) - 2}!")
                    break  # Interrompe il ciclo while e passa all'estrazione del percorso

    # 4. Estrazione del percorso
    try:
        path_indices = nx.shortest_path(G, source=0, target=1, weight='weight')
        path = [samples[i] for i in path_indices]
        print(f"PRM Concluso. Nodi della rotta: {len(path)}")

    except nx.NetworkXNoPath:
        print("ERRORE: Raggiunto il limite massimo di campioni senza trovare vie d'uscita!")
        path = [start, goal]

    return path, samples, G


T, ALL_PRM_NODES, PRM_GRAPH = generate_prm(START_POS, GOAL_POS)

t_idx = 1 if len(T) > 1 else 0
TARGET = np.array(T[t_idx], dtype=float)

# Inizializzazione Formazione
safe_center = list(START_POS)
Pos = [safe_center]

# Scala dinamicamente anche l'area di spawn iniziale dei droni
SPAWN_RADIUS = 2.0 * SCALE_FACTOR

for i in range(1, NUM_AGENTS):
    angle = np.random.uniform(0, 2 * math.pi)
    r = SPAWN_RADIUS * math.sqrt(np.random.uniform(0, 1))

    p = [round(safe_center[0] + r * math.cos(angle), 2),
         round(safe_center[1] + r * math.sin(angle), 2)]
    Pos.append(p)

positions = np.array(Pos)
CONNECTIONS = []

# Calcolo angoli
for a in range(NUM_AGENTS - 4):
    angolo = A[a] - ((180 - ALPHA) / 2)
    A.append(angolo)

# Calcolo lunghezze diagonali
for d in range(int((NUM_AGENTS - 1) / 2) - 1):
    last = dist[-1]
    new_dist = math.sqrt(dist[1] * dist[1] + last * last - 2 * dist[1] * last * math.cos(math.radians(A[d])))
    dist.append(round(new_dist, 2))

dist = dist + dist[::-1]
if NUM_AGENTS % 2 != 0:
    dist.pop(int(len(dist) / 2))


def bound(new_positions):
    global CONNECTIONS, DESIRED_DISTANCES
    DESIRED_DISTANCES.fill(0)
    CONNECTIONS.clear()

    center_pos = new_positions[0]
    angles = []
    for i in range(1, NUM_AGENTS):
        diff = new_positions[i] - center_pos
        angle = math.atan2(diff[1], diff[0])
        angles.append((i, angle))

    angles.sort(key=lambda x: x[1])
    sorted_indices = [item[0] for item in angles]
    num_sat = len(sorted_indices)

    for k in range(num_sat):
        sat_A = sorted_indices[k]
        sat_B = sorted_indices[(k + 1) % num_sat]

        DESIRED_DISTANCES[sat_A, sat_B] = dist[1]
        DESIRED_DISTANCES[sat_B, sat_A] = dist[1]

        limit_step = min(num_sat // 2 + 1, 2 + MAX_DIAG_STEPS)

        for step in range(2, limit_step):
            if step < len(dist):
                sat_C = sorted_indices[(k + step) % num_sat]
                DESIRED_DISTANCES[sat_A, sat_C] = dist[step]
                DESIRED_DISTANCES[sat_C, sat_A] = dist[step]

    for i in range(NUM_AGENTS):
        for j in range(i + 1, NUM_AGENTS):
            if DESIRED_DISTANCES[i, j] > 0:
                CONNECTIONS.append((i, j))


bound(positions)


# Funzioni di Controllo (APF)
def calculate_formation_force(pos, all_positions, agent_index):
    force = np.zeros(2)
    for j, other_pos in enumerate(all_positions):
        if j == agent_index: continue
        d_des = DESIRED_DISTANCES[agent_index, j]

        if d_des > 0:
            diff = pos - other_pos
            dist = np.linalg.norm(diff)

            if dist > 0.001:
                error = dist - d_des
                current_k = K_FORM * min(1.0, (D / d_des))

                if abs(error) <= HUBER_DELTA:
                    force_mag = -current_k * error
                else:
                    force_mag = -current_k *  np.sign(error)

                force += force_mag * (diff / dist)
    return force


def calculate_circular_orbit_force(pos, center_pos, radius):
    diff = pos - center_pos
    dist = np.linalg.norm(diff)
    if dist < 0.001: return np.random.rand(2) * 0.1

    error = dist - radius
    current_k = K_CIRC_OUT if error > 0 else K_CIRC_IN  # Scelta potenziale repulsivo

    if abs(error) <= HUBER_DELTA:
        force_mag = -current_k * error
    else:
        force_mag = -current_k *  np.sign(error)

    return force_mag * (diff / dist)


def calculate_repulsive_force(pos, all_positions, agent_index):
    force = np.zeros(2)
    for j, other_pos in enumerate(all_positions):
        if j == agent_index: continue
        diff = pos - other_pos
        dist = np.linalg.norm(diff)
        if REP_RADIUS > dist > 0.001:
            rep_mag = K_REP * (1.0 / dist - 1.0 / REP_RADIUS) * (1.0 / (dist ** 2))
            force += rep_mag * (diff / dist)
    return force


def calculate_obstacle_force(pos, leader_pos):
    force = np.zeros(2)
    for (ox, oy, ow, oh) in OBSTACLES:
        closest_pt = get_closest_point_on_rect(pos[0], pos[1], ox, oy, ow, oh)
        diff = pos - closest_pt
        dist_to_edge = np.linalg.norm(diff)

        if dist_to_edge < 0.001:
            diff = pos - np.array([ox + ow / 2, oy + oh / 2])
            dist_to_edge = 0.001

        if 0 < dist_to_edge < OBS_INFLUENCE:
            # Forza repulsiva normale
            normal = diff / dist_to_edge
            rep_mag = K_OBS * (1.0 / dist_to_edge - 1.0 / OBS_INFLUENCE) * (1.0 / (dist_to_edge ** 2))
            normal_force = rep_mag * normal

            # Forza tangenziale
            tangent_1 = np.array([-normal[1], normal[0]])
            tangent_2 = np.array([normal[1], -normal[0]])

            # Uso tangente che punta maggiormente verso il leader
            dir_to_leader = leader_pos - pos
            if np.dot(tangent_1, dir_to_leader) > 0:
                tangent_force = rep_mag * tangent_1 * 0.8
            else:
                tangent_force = rep_mag * tangent_2 * 0.8

            force += normal_force + (tangent_force/2)
    return force


def limit_speed(velocity, max_speed):
    speed = np.linalg.norm(velocity)
    return (velocity / speed) * max_speed if speed > max_speed else velocity


# Setup Grafico
fig, ax = plt.subplots(figsize=(8, 8))
fig.subplots_adjust(bottom=0.15)

ax.set_xlim(X_MIN - 1 * SCALE_FACTOR, X_MAX + 1 * SCALE_FACTOR)
ax.set_ylim(Y_MIN - 1 * SCALE_FACTOR, Y_MAX + 1 * SCALE_FACTOR)

ax.set_title("Formazione Multi-Agente PRM")
ax.grid(True)

for (ox, oy, ow, oh) in OBSTACLES:
    rect = patches.Rectangle((ox, oy), ow, oh, color='silver', alpha=0.8, zorder=1)
    ax.add_patch(rect)
prm_line, = ax.plot([p[0] for p in T], [p[1] for p in T], 'go--', lw=1, markersize=6, alpha=0.6, zorder=0)
target_plot, = ax.plot(TARGET[0], TARGET[1], 'rX', markersize=12, label="Target")
prm_nodes_scatter = ax.scatter([p[0] for p in ALL_PRM_NODES], [p[1] for p in ALL_PRM_NODES], c='lightgray', s=15,
                               zorder=0)
prm_nodes_scatter.set_visible(False)
scat = ax.scatter(positions[:, 0], positions[:, 1], c='b', s=100, zorder=4, label="Agenti")
MAX_LINKS = NUM_AGENTS * (NUM_AGENTS - 1) // 2
graph_lines = [ax.plot([], [], 'k-', lw=2, zorder=2)[0] for _ in range(MAX_LINKS)]


edge_segments = []
for u, v in PRM_GRAPH.edges():
    p1 = ALL_PRM_NODES[u]
    p2 = ALL_PRM_NODES[v]
    edge_segments.append([p1, p2])

# Creazione della collezione di linee (archi del grafo)
prm_edges_collection = LineCollection(edge_segments, colors='lightgray',
                                      linewidths=0.5, alpha=0.3, zorder=0)
ax.add_collection(prm_edges_collection)
prm_edges_collection.set_visible(False) # Nascosto di default

show_links = False
show_nodes = False

ax_button = plt.axes((0.1, 0.9, 0.25, 0.06))
btn_links = Button(ax_button, 'Toggle Vincoli')

ax_btn_nodes = plt.axes((0.68, 0.9, 0.3, 0.06))
btn_nodes = Button(ax_btn_nodes, 'Toggle Grafo')


def toggle_visibility(event):
    global show_links
    show_links = not show_links


def toggle_nodes(event):
    global show_nodes
    show_nodes = not show_nodes
    prm_nodes_scatter.set_visible(show_nodes)
    prm_edges_collection.set_visible(show_nodes)


btn_links.on_clicked(toggle_visibility)
btn_nodes.on_clicked(toggle_nodes)

# Ciclo di Aggiornamento
def update(frame):
    global positions, TARGET, t_idx, last_bound_frame, is_waiting, wait_start_frame

    centroid = positions[CIRC_CENTER_IDX]

    if t_idx < len(T):

        # --- NOVITÀ: Logica di Attesa ---
        if is_waiting:
            # Controlla se è passato abbastanza tempo (frame) dall'inizio dell'attesa
            if frame - wait_start_frame >= WAIT_FRAMES:
                is_waiting = False  # Fine attesa!
                t_idx += 1  # Passa al nodo successivo

                if t_idx < len(T):
                    TARGET = np.array(T[t_idx], dtype=float)

                    # Aggiorna la topologia alla ripartenza (cooldown)
                    if (frame - last_bound_frame) >= MIN_FRAMES_BETWEEN_BOUNDS:
                        bound(positions)
                        last_bound_frame = frame
        else:
            # Se NON sta aspettando, controlla se ha raggiunto il bersaglio
            dist_agent_to_target = np.linalg.norm(centroid - TARGET)

            if dist_agent_to_target < 0.5:
                # Arrivato al waypoint! Inizia l'attesa invece di passare subito oltre
                is_waiting = True
                wait_start_frame = frame

        # Comportamento a fine percorso
    else:
        # TOPOLOGY_UPDATE_INTERVAL dovrebbe essere impostato a 20 nei tuoi parametri
        if frame % TOPOLOGY_UPDATE_INTERVAL == 0:
            bound(positions)

    target_plot.set_data([TARGET[0]], [TARGET[1]])

    new_positions = np.copy(positions)
    vector_to_target = TARGET - centroid
    dist_to_target = np.linalg.norm(vector_to_target)

    f_target_global = np.zeros(2)
    if dist_to_target > 0.1:
        if dist_to_target <= HUBER_DELTA:
            f_target_global = K_TARGET * vector_to_target
        else:
            f_target_global = K_TARGET * (vector_to_target / dist_to_target)

    agent_colors = []

    for i in range(NUM_AGENTS):
        f_rep = calculate_repulsive_force(positions[i], positions, i)
        f_obs = calculate_obstacle_force(positions[i], positions[0])
        f_form = calculate_formation_force(positions[i], positions, i)

        if i in SATELLITE_IDX:
            f_circ = calculate_circular_orbit_force(positions[i], positions[CIRC_CENTER_IDX], CIRC_RADIUS)
            total_force = f_form + f_circ + f_rep + f_obs
            velocity = limit_speed(total_force, MAX_SPEED * 1.5)

        else:
            f_global = f_target_global
            total_force = f_global + f_form + f_rep + f_obs
            velocity = limit_speed(total_force, MAX_SPEED)

        new_positions[i] += velocity * DT

        if i in SATELLITE_IDX:
            agent_colors.append('cyan')
        else:
            agent_colors.append('blue')

    positions = new_positions

    scat.set_offsets(positions)
    scat.set_color(agent_colors)

    for idx, line in enumerate(graph_lines):
        if show_links and idx < len(CONNECTIONS):
            i, j = CONNECTIONS[idx]
            p_i = positions[i]
            p_j = positions[j]
            line.set_data([p_i[0], p_j[0]], [p_i[1], p_j[1]])
            line.set_color('black')
        else:
            line.set_data([], [])

    return [scat, target_plot, prm_line, prm_nodes_scatter, prm_edges_collection] + graph_lines


if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, update, frames=MAX_STEPS, interval=20, blit=True)
    plt.show()