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
NUM_AGENTS = 14

# 7 è considerato il numero standard. Non scala mai sotto l'1.0.
SCALE_FACTOR = max(1.0, NUM_AGENTS / 7.0)
print('Fattore di scala:', SCALE_FACTOR)

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

REP_RADIUS = 0.6
OBS_INFLUENCE = 0.6
HUBER_DELTA = 1.0
DANGER_OBS = 0.5

CIRC_CENTER_IDX = 0
SATELLITE_IDX = list(range(1, NUM_AGENTS))

# I limiti dell'immagine/mappa si scalano dinamicamente
X_MIN, X_MAX = 0.0, 15.0 * SCALE_FACTOR
Y_MIN, Y_MAX = 0.0, 15.0 * SCALE_FACTOR

MIN_START_GOAL_DIST = 10.0 * SCALE_FACTOR
NUM_OBSTACLES = 9

MAX_DIAG_STEPS = int(2 * SCALE_FACTOR)
print('Numero diagonali:', min(MAX_DIAG_STEPS*2, NUM_AGENTS -4))
TOPOLOGY_UPDATE_INTERVAL = 40   # Frequenza aggiornamento vincoli quando arriva all'ultimo waypoint
MIN_FRAMES_BETWEEN_BOUNDS = 40  # 20 frame corrispondono a 1 secondo simulato (se DT=0.05 e intervallo=20ms)
last_bound_frame = 0            # Memoria dell'ultimo frame in cui la topologia è stata aggiornata
WAIT_TIME_SECONDS = 0.4         # Quanti secondi si ferma su ogni nodo
WAIT_FRAMES = int(WAIT_TIME_SECONDS / DT)  # Converte i secondi in numero di frame (es. 2.0 / 0.05 = 40)
is_waiting = False       # Stato: True se l'agente sta aspettando sul nodo
wait_start_frame = 0     # Memoria del frame in cui è iniziata l'attesa
satellite_paths = {i: [] for i in range(NUM_AGENTS)}
position_history = np.zeros((NUM_AGENTS, 2))
stall_timers = np.zeros(NUM_AGENTS)
prm_timers = np.zeros(NUM_AGENTS) # Timer per evitare blocchi durante il PRM

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
def generate_prm(start, goal, min_samples=100 , max_samples=1000, k_neighbors=5):
    print("Generazione PRM in corso...")
    samples = [start, goal]

    # Inizializziamo subito il Grafo con Partenza (0) e Arrivo (1)
    G = nx.Graph()
    G.add_node(0, pos=start)
    G.add_node(1, pos=goal)

    node_idx = 2  # Parte dall'indice 2

    # Generazione incrementale
    while node_idx < max_samples + 2:
        # Genera un punto sicuro
        pt = np.random.uniform([X_MIN, Y_MIN], [X_MAX, Y_MAX])
        safe = True
        for (ox, oy, ow, oh) in OBSTACLES:
            if (ox - 0.5) <= pt[0] <= (ox + ow + 0.5) and (oy - 0.5) <= pt[1] <= (oy + oh + 0.5):
                safe = False
                break

        if safe:
            samples.append(pt.tolist())
            G.add_node(node_idx, pos=pt.tolist())

            # Calcola le distanze SOLO rispetto ai nodi già esistenti
            distances = []
            for j in range(node_idx):
                d = np.linalg.norm(np.array(samples[node_idx]) - np.array(samples[j]))
                distances.append((j, d))
            distances.sort(key=lambda x: x[1])

            # Collega il nuovo nodo ai suoi K vicini più prossimi
            for j, d in distances[:k_neighbors]:
                collision = any(
                    line_intersects_rect(samples[node_idx], samples[j], ox, oy, ow, oh) for ox, oy, ow, oh in OBSTACLES)
                if not collision:
                    G.add_edge(node_idx, j, weight=d)

            node_idx += 1

            # Se viene superato il minimo richiesto di campioni
            if len(samples) >= min_samples + 2:
                # verifica se i nodi 0 (Start) e 1 (Goal) sono connessi
                if nx.has_path(G, 0, 1):
                    print(f"  -> Percorso trovato in anticipo al campione {len(samples) - 2}!")
                    break  # Interrompe il ciclo while e passa all'estrazione del percorso

    # Estrazione del percorso
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

# Funzione di calcolo vincoli
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
                    force_mag = -current_k * HUBER_DELTA * np.sign(error)

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
        force_mag = -current_k * HUBER_DELTA * np.sign(error)

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


def calculate_escape_path(start_pos, target_pos):
    best_start = None
    min_ds = float('inf')

    best_start_fallback = None
    min_ds_fallback = float('inf')

    for idx in PRM_GRAPH.nodes():
        node_pos = np.array(ALL_PRM_NODES[idx])
        d = np.linalg.norm(start_pos - node_pos)

        # Aggiorna sempre il nodo più vicino in assoluto (Piano B)
        if d < min_ds_fallback:
            min_ds_fallback = d
            best_start_fallback = idx

        # Cerca il nodo più vicino visibile senza attraversare ostacoli
        if d < min_ds and not any(
                line_intersects_rect(start_pos, node_pos, ox, oy, ow, oh) for ox, oy, ow, oh in OBSTACLES):
            min_ds = d
            best_start = idx

    # Se non ha trovato nodi perfettamente visibili, usa il più vicino in assoluto
    if best_start is None:
        best_start = best_start_fallback

    # Fa lo stesso per il bersaglio (il leader)
    best_target = None
    min_dt = float('inf')
    best_target_fallback = None
    min_dt_fallback = float('inf')

    for idx in PRM_GRAPH.nodes():
        node_pos = np.array(ALL_PRM_NODES[idx])
        d = np.linalg.norm(target_pos - node_pos)

        if d < min_dt_fallback:
            min_dt_fallback = d
            best_target_fallback = idx

        if d < min_dt and not any(
                line_intersects_rect(target_pos, node_pos, ox, oy, ow, oh) for ox, oy, ow, oh in OBSTACLES):
            min_dt = d
            best_target = idx

    if best_target is None:
        best_target = best_target_fallback

    # Calcolo percorso
    if best_start is not None and best_target is not None:
        try:
            path_idx = nx.shortest_path(PRM_GRAPH, best_start, best_target, weight='weight')
            path = [np.array(ALL_PRM_NODES[idx]) for idx in path_idx]
            if len(path) > 0 and np.linalg.norm(start_pos - path[0]) < 1.0:
                path.pop(0)
            return path
        except nx.NetworkXNoPath:
            pass
    return []

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
sat_path_lines = [ax.plot([], [], 'c--', lw=1.5, alpha=0.8, zorder=3)[0] for _ in range(NUM_AGENTS)]

edge_segments = []
for u, v in PRM_GRAPH.edges():
    p1 = ALL_PRM_NODES[u]
    p2 = ALL_PRM_NODES[v]
    edge_segments.append([p1, p2])

# Creazione della collezione di linee (archi del grafo)
prm_edges_collection = LineCollection(edge_segments, colors='black',linewidths=0.5, alpha=0.3, zorder=0)
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
    global positions, TARGET, t_idx, last_bound_frame, is_waiting, wait_start_frame, satellite_paths,\
        position_history, stall_timers, prm_timers
    centroid = positions[CIRC_CENTER_IDX]
    # Inizializza la cronologia posizioni al primissimo frame
    if frame == 0:
        position_history = np.copy(positions)

    if t_idx < len(T):

        if is_waiting:
            # Controlla se è passato abbastanza tempo (frame) dall'inizio dell'attesa
            if frame - wait_start_frame >= WAIT_FRAMES:
                is_waiting = False  # Fine attesa
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
                # Arrivato al waypoint Inizia l'attesa invece di passare subito oltre
                is_waiting = True
                wait_start_frame = frame

        # Comportamento a fine percorso
    else:
        # TOPOLOGY_UPDATE_INTERVAL dovrebbe essere impostato a 20 nei parametri
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
            f_target_global = K_TARGET * HUBER_DELTA * (vector_to_target / dist_to_target)

    agent_colors = []

    leader_velocity = np.zeros(2)

    for i in range(NUM_AGENTS):
        f_rep = calculate_repulsive_force(positions[i], positions, i)
        f_obs = calculate_obstacle_force(positions[i], positions[0])
        f_form = calculate_formation_force(positions[i], positions, i)

        if i in SATELLITE_IDX:
            f_circ = calculate_circular_orbit_force(positions[i], positions[CIRC_CENTER_IDX], CIRC_RADIUS)

            # PRM INDIVIDUALE
            if len(satellite_paths[i]) > 0:
                #  Aggiorniamo il timer di percorrenza
                prm_timers[i] += 1

                # Se ci mette più di 3 secondi (60 frame) per toccare un singolo nodo, è bloccato!
                if prm_timers[i] > 60:
                    print(f"[{frame}] Satellite {i} bloccato lungo il PRM! Abortisco la rotta.")
                    satellite_paths[i] = []  # Svuota la rotta
                    prm_timers[i] = 0  # Azzera il timer
                    continue  # Passa al prossimo frame per ricalcolare tutto

                wp = satellite_paths[i][0]
                dist_to_wp = np.linalg.norm(positions[i] - wp)

                if dist_to_wp < 0.5:
                    satellite_paths[i].pop(0)
                    prm_timers[i] = 0  # Nodo raggiunto: azzera il timer!
                    velocity = np.zeros(2)
                else:
                    vec_to_wp = wp - positions[i]
                    if dist_to_wp <= HUBER_DELTA:
                        f_wp = K_TARGET * vec_to_wp
                    else:
                        f_wp = K_TARGET * HUBER_DELTA * (vec_to_wp / dist_to_wp)

                    # Moltiplica f_wp per 2.0 e riduciamo f_rep al 50%
                    # Così il drone "spingerà" via i compagni per salvarsi, senza farsi bloccare da loro
                    total_force = (f_wp * 2.0) + (f_rep * 0.5) + f_obs
                    velocity = limit_speed(total_force, MAX_SPEED * 1.5)

            else:
                total_force = f_form + f_circ + f_rep + f_obs

                # Rilevamento Stallo Posizionale Infallibile
                stall_timers[i] += 1

                # Controlla la situazione ogni 40 frame (circa 2 secondi di simulazione)
                if stall_timers[i] >= 40:
                    # Quanto si è mosso il drone negli ultimi 2 secondi?
                    dist_moved = np.linalg.norm(positions[i] - position_history[i])
                    dist_from_leader = np.linalg.norm(positions[i] - positions[CIRC_CENTER_IDX])

                    # Se ha percorso meno di 1 metro in 2 secondi, ED è fuori dall'anello
                    if dist_moved < 1.0 and dist_from_leader > (CIRC_RADIUS * 1.5):
                        print(f"[{frame}] Satellite {i} bloccato fisicamente! Genero PRM...")
                        new_path = calculate_escape_path(positions[i], positions[CIRC_CENTER_IDX])
                        if new_path:
                            satellite_paths[i] = new_path

                    # Salva la posizione attuale per il prossimo controllo e resettiamo il timer
                    position_history[i] = np.copy(positions[i])
                    stall_timers[i] = 0

                base_velocity = limit_speed(total_force, MAX_SPEED * 1.5)

                # FEEDFORWARD
                velocity = base_velocity + (leader_velocity * 0.60)

                velocity = limit_speed(velocity, MAX_SPEED * 1.8)

        else:
            # Comportamento del leader principale
            f_global = f_target_global
            total_force = f_global + f_form + f_rep + f_obs
            velocity = limit_speed(total_force, MAX_SPEED)

            leader_velocity = np.copy(velocity)

        new_positions[i] += velocity * DT

        if i in SATELLITE_IDX:
            agent_colors.append('cyan')
        else:
            agent_colors.append('blue')

    positions = new_positions

    scat.set_offsets(positions)
    scat.set_color(agent_colors)

    # Aggiunta del percorso di fuga
    for i in range(NUM_AGENTS):
        if i in SATELLITE_IDX and len(satellite_paths[i]) > 0:
            path_coords = [positions[i]] + satellite_paths[i]
            sat_path_lines[i].set_data([p[0] for p in path_coords], [p[1] for p in path_coords])
        else:
            sat_path_lines[i].set_data([], [])

    for idx, line in enumerate(graph_lines):
        if show_links and idx < len(CONNECTIONS):
            i, j = CONNECTIONS[idx]
            p_i = positions[i]
            p_j = positions[j]
            line.set_data([p_i[0], p_j[0]], [p_i[1], p_j[1]])
            line.set_color('black')
        else:
            line.set_data([], [])

    return [scat, target_plot, prm_line, prm_nodes_scatter, prm_edges_collection] + graph_lines + sat_path_lines


if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, update, frames=MAX_STEPS, interval=20, blit=True)
    plt.show()