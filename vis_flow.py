import numpy as np
import torch
import scipy.stats

import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation



def setup_flow_animation(
    flow,
    n_particles=300,
    n_steps=50,
    xlim=(-3,3),
    ylim=(-3,3),
    grid_size=100
):

    x = torch.randn(
        n_particles,
        2
    )

    time_steps = torch.linspace(
        0,
        1.0,
        n_steps + 1
    )


    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12,6)
    )


    sc = ax1.scatter(
        x[:,0],
        x[:,1],
        s=10
    )

    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax1.set_aspect("equal")

    title1 = ax1.set_title(
        f"Particles at t={time_steps[0]:.2f}"
    )


    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax2.set_aspect("equal")

    title2 = ax2.set_title(
        "Density"
    )


    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], grid_size),
        np.linspace(ylim[0], ylim[1], grid_size)
    )


    grid_points = torch.tensor(
        np.stack(
            [
                xx.flatten(),
                yy.flatten()
            ],
            axis=1
        ),
        dtype=torch.float32
    )


    kde_xx, kde_yy = np.mgrid[
        xlim[0]:xlim[1]:complex(grid_size),
        ylim[0]:ylim[1]:complex(grid_size)
    ]


    positions = np.vstack(
        [
            kde_xx.ravel(),
            kde_yy.ravel()
        ]
    )


    kernel = scipy.stats.gaussian_kde(
        x.numpy().T
    )

    z = np.reshape(
        kernel(positions),
        kde_xx.shape
    )


    ax2.contourf(
        kde_xx,
        kde_yy,
        z,
        levels=15,
        cmap="Blues",
        alpha=0.7
    )


    return {
        "fig": fig,
        "x": x,
        "time_steps": time_steps,
        "sc": sc,
        "title1": title1,
        "title2": title2,
        "ax2": ax2,
        "kde_xx": kde_xx,
        "kde_yy": kde_yy,
        "positions": positions,
        "n_steps": n_steps
    }



def animate_flow(
    flow,
    state
):

    def update(frame):

        x = state["x"]

        time_steps = state["time_steps"]


        x = flow.step(
            x,
            time_steps[frame],
            time_steps[frame+1]
        )

        state["x"] = x


        state["sc"].set_offsets(
            x.detach().numpy()
        )


        state["title1"].set_text(
            f"Particles at t={time_steps[frame+1]:.2f}"
        )


        x_np = x.detach().numpy()


        kernel = scipy.stats.gaussian_kde(
            x_np.T
        )

        z = np.reshape(
            kernel(state["positions"]),
            state["kde_xx"].shape
        )


        ax2 = state["ax2"]

        for c in ax2.collections:
            c.remove()


        ax2.contourf(
            state["kde_xx"],
            state["kde_yy"],
            z,
            levels=15,
            cmap="Blues",
            alpha=0.7
        )


        state["title2"].set_text(
            f"Density at t={time_steps[frame+1]:.2f}"
        )


        return state["sc"],


    animation = FuncAnimation(
        state["fig"],
        update,
        frames=state["n_steps"],
        interval=100,
        blit=False
    )


    plt.close()

    return HTML(
        animation.to_jshtml()
    )


def animate_vector_field(
    flow,
    n_particles=300,
    n_steps=50,
    xlim=(-3,3),
    ylim=(-3,3),
    grid_size=15,
    interval=100
):

    # Initial particles
    x = torch.randn(
        n_particles,
        2
    )


    time_steps = torch.linspace(
        0,
        1.0,
        n_steps + 1
    )


    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12,6)
    )


    # ---------------------
    # Particle subplot
    # ---------------------

    sc = ax1.scatter(
        x[:,0],
        x[:,1],
        s=10,
        color="blue"
    )


    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax1.set_aspect("equal")


    title1 = ax1.set_title(
        f"Particles t={time_steps[0]:.2f}"
    )



    # ---------------------
    # Vector field subplot
    # ---------------------

    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax2.set_aspect("equal")


    title2 = ax2.set_title(
        "Flow Matching Vector Field"
    )


    xx, yy = np.meshgrid(
        np.linspace(
            xlim[0],
            xlim[1],
            grid_size
        ),
        np.linspace(
            ylim[0],
            ylim[1],
            grid_size
        )
    )


    grid_points = torch.tensor(
        np.stack(
            [
                xx.flatten(),
                yy.flatten()
            ],
            axis=1
        ),
        dtype=torch.float32
    )



    def get_vector_field(t):

        with torch.no_grad():

            t_tensor = torch.full(
                (len(grid_points),1),
                t
            )

            velocity = flow(
                grid_points,
                t_tensor
            )


        velocity = velocity.numpy()


        u = velocity[:,0].reshape(
            xx.shape
        )

        v = velocity[:,1].reshape(
            xx.shape
        )


        return u,v



    u,v = get_vector_field(
        time_steps[0].item()
    )


    quiver = ax2.quiver(
        xx,
        yy,
        u,
        v,
        angles="xy",
        scale=None
    )



    def update(frame):

        nonlocal x


        t0 = time_steps[frame]
        t1 = time_steps[frame+1]


        # Move particles

        x = flow.step(
            x,
            t0,
            t1
        )


        sc.set_offsets(
            x.detach().numpy()
        )


        title1.set_text(
            f"Particles t={t1:.2f}"
        )


        # Update vector field

        u,v = get_vector_field(
            t1.item()
        )


        quiver.set_UVC(
            u,
            v
        )


        title2.set_text(
            f"Vector Field t={t1:.2f}"
        )


        return sc, quiver



    ani = FuncAnimation(
        fig,
        update,
        frames=n_steps,
        interval=interval,
        blit=False
    )


    plt.close()


    return HTML(
        ani.to_jshtml()
    )
