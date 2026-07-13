import numpy as np
from IPython.display import HTML
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde
from matplotlib.animation import FuncAnimation


def sample_gaussian(n=500):
    return np.random.randn(n,2)


def sample_moons(n=500):
    x,_ = make_moons(n_samples=n, noise=0.05)
    return x*2


def match_points(source, target):
    d = np.sum((source[:,None,:]-target[None,:,:])**2, axis=-1)
    rows,cols = linear_sum_assignment(d)
    return target[cols]


def particle_positions(t, source, target):
    return (1-t) * source + t * target


def particle_velocities(source, target):
    return target - source


def normalize_field(field):
    magnitude = np.linalg.norm(field, axis=1, keepdims=True)
    magnitude[magnitude == 0] = 1
    return field / magnitude


def transport_paths(source, target):
    distance = np.sum(
        (source[:,None,:] - target[None,:,:])**2,
        axis=-1
    )

    rows,cols = linear_sum_assignment(distance)
    target_ordered = target[cols]
    plt.figure(figsize=(6,6))

    for i in range(len(source)):
        plt.plot(
            [
                source[i,0],
                target_ordered[i,0]
            ],
            [
                source[i,1],
                target_ordered[i,1]
            ],
            linewidth=0.3
        )

    return source, target_ordered


def velocity_field(source, target_ordered, points, t):
  desired_position = ((1-t)*source + t*target_ordered)
  velocity = desired_position - points
  return velocity


def integrate_trajectory(source, target_ordered):
    points = source.copy()
    history = []
    times = np.linspace(0,1,100)
    for t in times:
        velocity = velocity_field(source, target_ordered, points, t)
        points = points + 0.08 * velocity
        history.append(points.copy())
    return history


def integrate_transport(source, target):
    history = []
    points = source.copy()
    steps = 100
    dt = 1/steps
    for step in range(steps):
        t = step/(steps-1)
        desired = particle_positions(t, source, target)
        velocity = desired - points
        points = points + 0.2*velocity
        history.append(points.copy())
    return history, points, steps, dt


def estimate_field(t, source, target, grid):
    positions = particle_positions(t, source, target)
    velocities = particle_velocities(source, target)
    neighbours = NearestNeighbors(
        n_neighbors=25
    )
    neighbours.fit(positions)
    _, indices = neighbours.kneighbors(grid)

    field = np.zeros_like(grid,dtype=float)
    for i in range(len(grid)):
        field[i] = np.mean(
            velocities[indices[i]],
            axis=0
        )
    return field


def plot_field(history, source, target):
    x = np.linspace(-5,5,15)
    y = np.linspace(-5,5,15)
    X,Y = np.meshgrid(x,y)
    grid = np.column_stack([X.flatten(), Y.flatten()])


    fig,(ax1,ax2)=plt.subplots(
        1,
        2,
        figsize=(12,5)
    )
    particle_plot = ax1.scatter(
        history[0][:,0],
        history[0][:,1],
        s=8
    )
    field = normalize_field(estimate_field(0, source, target, grid))
    arrow_plot = ax2.quiver(
        grid[:,0],
        grid[:,1],
        field[:,0],
        field[:,1],
        color=plt.cm.viridis(0.15),
        angles="xy",
        scale_units="xy",
        scale=None
    )

    for ax in [ax1,ax2]:
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_aspect("equal")

    return fig, (ax1, ax2), grid, particle_plot, arrow_plot


def sample_dice(number_of_samples=1000, number_of_dice=1):
    rolls = np.random.randint(1, 7, size=(number_of_samples, number_of_dice))
    samples = np.sum(rolls, axis=1)

    theoretical_mean = number_of_dice * 3.5
    theoretical_variance = number_of_dice * (35 / 12)

    plt.figure(figsize=(10, 5))
    min_possible = number_of_dice
    max_possible = number_of_dice * 6
    bins = np.arange(min_possible - 0.5, max_possible + 1.5, 1)
    plt.hist(samples, bins=bins, density=True, alpha=0.75)

    total_range = max_possible - min_possible
    step = max(1, int(np.ceil(total_range / 9)))
    ticks = range(min_possible, max_possible + 1, step)

    plt.xticks(ticks)
    plt.xlabel("Sum of Dice Value")
    plt.ylabel("Probability Density")
    plt.title(
        f"Distribution of Rolling {number_of_dice} Dice ({number_of_samples} samples)\n"
        f"Theoretical Mean: {theoretical_mean:.2f} | Variance: {theoretical_variance:.2f}"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def kde(grid_size, x_rot, y_rot, xlim, ylim):
    kde_xx, kde_yy = np.mgrid[
        xlim[0]:xlim[1]:complex(grid_size),
        ylim[0]:ylim[1]:complex(grid_size)
    ]
    positions = np.vstack([kde_xx.ravel(), kde_yy.ravel()])
    
    sample_points = np.vstack([x_rot, y_rot])
    kernel = gaussian_kde(sample_points)
    z = np.reshape(kernel(positions), kde_xx.shape)
    
    plt.contourf(
        kde_xx,
        kde_yy,
        z,
        levels=15,
        cmap="Blues",
        alpha=0.7
    )


def animate_random_transport_pairs(
    source,
    target,
    n_samples=100,
    interval=500
):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set explicit axis limits first to define the KDE grid bounds
    xlim = (-4, 4)
    ylim = (-4, 4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # --- Add Target KDE Background ---
    # Generate grid for KDE evaluation
    grid_size = 100
    kde_xx, kde_yy = np.mgrid[
        xlim[0]:xlim[1]:complex(grid_size),
        ylim[0]:ylim[1]:complex(grid_size)
    ]
    positions = np.vstack([kde_xx.ravel(), kde_yy.ravel()])
    
    # Calculate KDE using all available target points for accuracy
    kernel = gaussian_kde(target.T)
    z = np.reshape(kernel(positions), kde_xx.shape)
    
    # Render transparent, matching color contour background
    ax.contourf(
        kde_xx,
        kde_yy,
        z,
        levels=15,
        cmap="Blues",
        alpha=0.25,
        zorder=0
    )
    # ---------------------------------

    # Randomly choose displayed examples
    source_idx = np.random.choice(len(source), n_samples, replace=False)
    target_idx = np.random.choice(len(target), n_samples, replace=False)

    source_points = source[source_idx]
    target_points = target[target_idx]

    # Source distribution
    source_scatter = ax.scatter(
        source_points[:, 0],
        source_points[:, 1],
        facecolors="none",
        edgecolors="black",
        s=50,
        label="Source",
        zorder=1
    )

    # Target distribution
    target_scatter = ax.scatter(
        target_points[:, 0],
        target_points[:, 1],
        facecolors="none",
        edgecolors="blue",
        s=50,
        label="Target",
        zorder=2
    )

    # Selected points
    source_selected = ax.scatter(
        [], [],
        color="red",
        s=80,
        zorder=5
    )

    target_selected = ax.scatter(
        [], [],
        color="red",
        s=80,
        zorder=5
    )

    # Transport line
    line, = ax.plot(
        [], [],
        color="red",
        linewidth=2,
        zorder=4
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left")

    ax.set_title("Random source-target transport pair")
    pairs = np.random.randint(0, n_samples, size=150)

    def update(frame):
        i = pairs[frame]
        x0 = source_points[i]
        x1 = target_points[i]

        source_selected.set_offsets([x0])
        target_selected.set_offsets([x1])

        line.set_data(
            [x0[0], x1[0]],
            [x0[1], x1[1]]
        )

        return (
            source_selected,
            target_selected,
            line
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=150,
        interval=interval,
        blit=True
    )

    plt.close()
    return ani
