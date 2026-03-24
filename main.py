import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

matplotlib.use('TkAgg')

# ==========================================
# Parametri del Sistema
# ==========================================
NUM_AGENTS = 7
ALPHA =((NUM_AGENTS-3)*180)/(NUM_AGENTS - 1)
A = [ALPHA]

CIRC_RADIUS = 1.5
D = 2 * CIRC_RADIUS * np.sin(np.deg2rad(180/(NUM_AGENTS - 1)))
DESIRED_DISTANCES = np.zeros((NUM_AGENTS, NUM_AGENTS))

dist = [0, float(D)]

DT = 0.05
MAX_STEPS = 1000
MAX_SPEED = 2.0

# Guadagni e Raggi
K_TARGET = 1.1
K_FORM = 1.5
K_REP = 2.0
K_OBS = 4.0
K_CIRC_OUT = 4.0
K_CIRC_IN = 1.0

REP_RADIUS = 1.0
OBS_INFLUENCE = 0.8
HUBER_DELTA = 1.0
DANGER_OBS = 0.5

# Parametri dell'Orbita Circolare
CIRC_CENTER_IDX = 0
SATELLITE_IDX = list(range(1, NUM_AGENTS))

X_MIN, X_MAX = 0.0, 15.0
Y_MIN, Y_MAX = 0.0, 15.0

OBSTACLES = [
    (5.2, 4.5, 1.0),
    (10.0, 8.0, 1.2),
    (4.0, 11.0, 1.0),
    (9.0, 13.0, 1.0),
    (12.0, 4.0, 1.0)
]

T = [[3,4], [9,1], [13,13], [5,7], [2,6]]
t = 0
def generate_safe_position(x_min, x_max, y_min, y_max, margin=0.5):
    while True:
        pos = np.random.uniform([x_min, y_min], [x_max, y_max])
        safe = True
        for ox, oy, orad in OBSTACLES:
            if np.linalg.norm(pos - np.array([ox, oy])) <= (orad + margin):
                safe = False
                break
        if safe:
            return pos


# ==========================================
# Inizializzazione
# ==========================================
#TARGET = generate_safe_position(X_MIN + 1, X_MAX - 1, Y_MIN + 1, Y_MAX - 1)
TARGET = [1, 1]
TARGET_SPEED = 0.0
theta = np.random.uniform(0, 2 * np.pi)
target_vel = np.array([np.cos(theta), np.sin(theta)]) * TARGET_SPEED

safe_center = list(TARGET)

Pos = []
for index in range(0, NUM_AGENTS):
    x = math.cos(math.radians(index * 360/(NUM_AGENTS - 1 )))
    y = math.sin(math.radians(index * 360/(NUM_AGENTS - 1 )))

    p = [round(safe_center[0] + CIRC_RADIUS /2*x,2),round(safe_center[1] + CIRC_RADIUS/2 *y, 2)]
    Pos.append(p)

P = np.array(Pos)
positions = P

#Distances = np.zeros((NUM_AGENTS,NUM_AGENTS))
CONNECTIONS = []

#Calcolo angoli
for a in range(NUM_AGENTS - 4):
    angolo = A[a] -((180 - ALPHA)/2)
    A.append(angolo)


#Calcolo distanze
for d in range(int((NUM_AGENTS -1)/2) - 1):
    last = dist[-1]
    new_dist = math.sqrt(dist[1]*dist[1] + last*last - 2*dist[1]*last*math.cos(math.radians(A[d])))
    dist.append(round(new_dist, 2))

dist = dist + dist[::-1]
if NUM_AGENTS % 2 != 0:
    dist.pop(int(len(dist)/2))

#Inserimento distanze in matrice di distanze
for v in range(1, NUM_AGENTS):
    for w in range(v, NUM_AGENTS):
        if w != v:
            DESIRED_DISTANCES[v, w] = dist[w - v]
            DESIRED_DISTANCES[w, v] = DESIRED_DISTANCES[v, w]


for i in range(NUM_AGENTS):
    for j in range(i + 1, NUM_AGENTS):
        if DESIRED_DISTANCES[i, j] > 0:
            CONNECTIONS.append((i, j))


def bound(new_positions):
    global CONNECTIONS, DESIRED_DISTANCES

    DESIRED_DISTANCES.fill(0)
    CONNECTIONS.clear()

    # 1. Trova l'angolo di ogni satellite rispetto all'Agente 0 (centro)
    center_pos = new_positions[0]
    angles = []

    for i in range(1, NUM_AGENTS):
        diff = new_positions[i] - center_pos
        # math.atan2 restituisce l'angolo tra -pi e pi
        angle = math.atan2(diff[1], diff[0])
        angles.append((i, angle))

    # 2. Ordina i satelliti in base all'angolo (senso antiorario)
    angles.sort(key=lambda x: x[1])
    sorted_indices = [item[0] for item in angles]

    # 3. Assegna i vincoli
    num_sat = len(sorted_indices)

    for k in range(num_sat):
        sat_A = sorted_indices[k]
        # Il prossimo satellite nell'anello (con % num_sat per chiudere il cerchio)
        sat_B = sorted_indices[(k + 1) % num_sat]

        # Assegna la distanza tra i vicini dell'anello
        DESIRED_DISTANCES[sat_A, sat_B] = dist[1]
        DESIRED_DISTANCES[sat_B, sat_A] = dist[1]

        # --- OPZIONALE: Diagonali interne ---
        # Se vuoi che mantengano rigidamente la forma dell'esagono/ettagono,
        # devi assegnare anche le distanze tra satelliti non adiacenti.
        # Usa il vettore 'dist' che hai calcolato abilmente in inizializzazione!
        for step in range(2, num_sat // 2 + 1):
            if step < len(dist):  # Evita index out of range
                sat_C = sorted_indices[(k + step) % num_sat]
                DESIRED_DISTANCES[sat_A, sat_C] = dist[step]
                DESIRED_DISTANCES[sat_C, sat_A] = dist[step]

    # 4. Aggiorna la lista delle connessioni per la grafica
    for i in range(NUM_AGENTS):
        for j in range(i + 1, NUM_AGENTS):
            if DESIRED_DISTANCES[i, j] > 0:
                CONNECTIONS.append((i, j))


# ==========================================
# Funzioni di Controllo
# ==========================================
def calculate_formation_force(pos, all_positions, agent_index):
    force = np.zeros(2)
    for j, other_pos in enumerate(all_positions):
        if j == agent_index: continue
        d_des = DESIRED_DISTANCES[agent_index, j]
        if d_des > 0:
            diff = pos - other_pos
            dist = np.linalg.norm(diff)
            if dist > D:
                if dist > 0.001:
                    error = dist - d_des

                    if abs(error) <= HUBER_DELTA:
                        force_mag = -(K_FORM/2.0) * error
                    else:
                        force_mag = -(K_FORM/2.0) * HUBER_DELTA * np.sign(error)
                    force += force_mag * (diff / dist)
            else:
                if dist > 0.001:
                    error = dist - d_des

                    if abs(error) <= HUBER_DELTA:
                        force_mag = -K_FORM * error
                    else:
                        force_mag = -K_FORM * HUBER_DELTA * np.sign(error)
                    force += force_mag * (diff / dist)

    return force


def calculate_circular_orbit_force(pos, center_pos, radius):
    diff = pos - center_pos
    dist = np.linalg.norm(diff)

    # Prevenzione singolarità (se il satellite è esattamente al centro)
    if dist < 0.001:
        return np.random.rand(2) * 0.1

    # Calcolo dell'errore:
    # Positivo = l'agente è ESTERNO all'orbita (dist > radius)
    # Negativo = l'agente è INTERNO all'orbita (dist < radius)
    error = dist - radius

    if error > 0:
        current_k = K_CIRC_OUT  # Parete ripida
    else:
        current_k = K_CIRC_IN  # Parete dolce

    # Applichiamo il potenziale Huber-like usando il guadagno scelto
    if abs(error) <= HUBER_DELTA:
        force_mag = -current_k * error
    else:
        force_mag = -current_k * HUBER_DELTA * np.sign(error)

    # Moltiplichiamo l'intensità per il versore direzione
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


def calculate_obstacle_force(pos):
    force = np.zeros(2)
    for ox, oy, orad in OBSTACLES:
        obs_pos = np.array([ox, oy])
        diff = pos - obs_pos
        dist = np.linalg.norm(diff)
        dist_to_edge = dist - orad

        if 0 < dist_to_edge < OBS_INFLUENCE:
            rep_mag = K_OBS * (1.0 / dist_to_edge - 1.0 / OBS_INFLUENCE) * (1.0 / (dist_to_edge ** 2))
            force += rep_mag * (diff / dist)
    return force


def limit_speed(velocity, max_speed):
    speed = np.linalg.norm(velocity)
    if speed > max_speed:
        velocity = (velocity / speed) * max_speed
    return velocity


# ==========================================
# Setup Grafico
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
fig.subplots_adjust(bottom=0.15)
ax.set_xlim(X_MIN - 1, X_MAX + 1)
ax.set_ylim(Y_MIN - 1, Y_MAX + 1)
ax.set_title("Formazione con satelliti")
ax.grid(True)

#ax.plot([X_MIN, X_MAX, X_MAX, X_MIN, X_MIN], [Y_MIN, Y_MIN, Y_MAX, Y_MAX, Y_MIN], 'k--', lw=1)

for ox, oy, orad in OBSTACLES:
    circle = plt.Circle((ox, oy), orad, color='gray', alpha=0.5)
    ax.add_patch(circle)

target_plot, = ax.plot(TARGET[0], TARGET[1], 'rX', markersize=12, label="Target")
scat = ax.scatter(positions[:, 0], positions[:, 1], c='b', s=100, zorder=4, label="Agenti")
# Creiamo un numero massimo sufficiente di linee (es. Tutte le combinazioni possibili)
MAX_LINKS = NUM_AGENTS * (NUM_AGENTS - 1) // 2
graph_lines = [ax.plot([], [], 'k-', lw=2, zorder=2)[0] for _ in range(MAX_LINKS)]
#ax.legend(loc='upper right')

# Definisci una variabile globale per lo stato di visibilità (True = visibili, False = nascosti)
show_links = False

# Crea un "asse" dedicato al pulsante [posizione_X, posizione_Y, larghezza, altezza]
ax_button = plt.axes((0.1, 0.9, 0.25, 0.06))
btn_links = Button(ax_button, 'Toggle Vincoli')

# Funzione che viene chiamata quando premi il pulsante
def toggle_visibility(event):
    global show_links
    show_links = not show_links # Inverte lo stato (da True a False e viceversa)

# Collega il click del mouse alla funzione
btn_links.on_clicked(toggle_visibility)

def update(frame):
    global positions, TARGET, target_vel, t

    if np.linalg.norm(positions[0] - TARGET) < 0.1:
        t += 1
        TARGET = T[t % len(T)]
        bound(positions)
        #TARGET = generate_safe_position(X_MIN + 1, X_MAX - 1, Y_MIN + 1, Y_MAX - 1)

    angle_noise = np.random.uniform(-0.2, 0.2)
    c, s = np.cos(angle_noise), np.sin(angle_noise)
    rot_matrix = np.array([[c, -s], [s, c]])
    target_vel = np.dot(rot_matrix, target_vel)
    TARGET = TARGET + target_vel * DT

    if TARGET[0] <= X_MIN or TARGET[0] >= X_MAX:
        target_vel[0] *= -1
        TARGET[0] = np.clip(TARGET[0], X_MIN, X_MAX)
    if TARGET[1] <= Y_MIN or TARGET[1] >= Y_MAX:
        target_vel[1] *= -1
        TARGET[1] = np.clip(TARGET[1], Y_MIN, Y_MAX)

    for ox, oy, orad in OBSTACLES:
        if np.linalg.norm(TARGET - np.array([ox, oy])) <= orad:
            target_vel *= -1
            TARGET += target_vel * DT * 2

    target_plot.set_data([TARGET[0]], [TARGET[1]])

    # 2. Agenti e Forze

    new_positions = np.copy(positions)

    centroid = positions[0]

    vector_to_target = TARGET - centroid
    dist_to_target = np.linalg.norm(vector_to_target)

    f_target_global = np.zeros(2)
    if dist_to_target > 0.1:
        if dist_to_target <= HUBER_DELTA:
            f_target_global = K_TARGET * vector_to_target
        else:
            f_target_global = K_TARGET * HUBER_DELTA * (vector_to_target / dist_to_target)

    agent_colors = []

    for i in range(NUM_AGENTS):
        # Inizializza a zero
        f_circ = np.zeros(2)
        f_global = np.zeros(2)

        f_rep = calculate_repulsive_force(positions[i], positions, i)
        f_obs = calculate_obstacle_force(positions[i])
        f_form = calculate_formation_force(positions[i], positions, i)

        if i in SATELLITE_IDX:
            # L'Agente Satellite è guidato dalla funzione circolare rispetto all'Agente 1
            f_circ = calculate_circular_orbit_force(positions[i], positions[CIRC_CENTER_IDX], CIRC_RADIUS)
            total_force = f_global + f_form + f_circ + f_rep + f_obs
            velocity = limit_speed(total_force, MAX_SPEED) * 2
            # NOTA: Non riceve la spinta 'f_target_global'.
            # Viene trascinato indirettamente perché segue l'Agente 1 che si sposta!
        else:
            # Gli agenti normali ricevono forza di formazione e spinta verso il target
            f_global = f_target_global
            total_force = f_global + f_form + f_circ + f_rep + f_obs
            velocity = limit_speed(total_force, MAX_SPEED)

        new_positions[i] += velocity * DT

        if i in SATELLITE_IDX:
            agent_colors.append('cyan')
        else:
            agent_colors.append('blue')

    positions = new_positions

    # 3. Grafica
    scat.set_offsets(positions)
    scat.set_color(agent_colors)

    # Sposta l'anello verde in modo che segua costantemente l'Agente 1
    #orbit_patch.center = (positions[CIRC_CENTER_IDX][0], positions[CIRC_CENTER_IDX][1])

    # 3. Aggiornamento delle linee dinamiche
    global show_links
    for idx, line in enumerate(graph_lines):
        # Se c'è una connessione attiva per questo indice, disegnala
        if show_links and idx < len(CONNECTIONS):
            i, j = CONNECTIONS[idx]
            p_i = positions[i]
            p_j = positions[j]
            line.set_data([p_i[0], p_j[0]], [p_i[1], p_j[1]])
            line.set_color('black')
        else:
            # Se la linea non serve in questo frame, nascondila
            line.set_data([], [])

    return [scat, target_plot] + graph_lines


if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, update, frames=MAX_STEPS, interval=20, blit=True)
    plt.show()