import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plot_protein_transformation(
    n=80,
    noise_scale=0.08
):

    # -----------------------
    # Random starting chain
    # -----------------------

    random_steps = np.random.randn(
        n,
        2
    )

    random_steps /= np.linalg.norm(
        random_steps,
        axis=1,
        keepdims=True
    )


    random_chain = np.cumsum(
        random_steps,
        axis=0
    )


    random_chain -= random_chain.mean(
        axis=0
    )


    # -----------------------
    # Protein-like target
    # -----------------------

    t = np.linspace(
        0,
        1,
        n
    )


    x = (
        2.0*np.sin(2*np.pi*t)
        +
        0.6*np.sin(6*np.pi*t)
    )


    y = (
        1.5*np.cos(2*np.pi*t)
        +
        0.3*np.sin(8*np.pi*t)
    )


    protein_2d = np.column_stack(
        [
            x,
            y
        ]
    )


    protein_2d += (
        noise_scale
        *
        np.random.randn(n,2)
    )


    protein_2d -= protein_2d.mean(
        axis=0
    )


    # -----------------------
    # Flow trajectory
    # -----------------------

    frames = 60

    trajectory = []

    for t in np.linspace(0,1,frames):

        structure = (
            (1-t)*random_chain
            +
            t*protein_2d
        )

        trajectory.append(
            structure
        )


    # -----------------------
    # Convert to 3D
    # -----------------------

    z = 0.5*np.sin(
        np.linspace(
            0,
            4*np.pi,
            n
        )
    )


    random_chain_3d = np.column_stack(
        [
            random_chain[:,0],
            random_chain[:,1],
            z
        ]
    )


    protein_3d = np.column_stack(
        [
            protein_2d[:,0],
            protein_2d[:,1],
            z
        ]
    )


    # -----------------------
    # Plotly figure
    # -----------------------

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [
                {"type":"scene"},
                {"type":"scene"}
            ]
        ],

        subplot_titles=[
            "Random Starting Structure",
            "Generated Protein Structure"
        ]
    )


    fig.add_trace(
        go.Scatter3d(
            x=random_chain_3d[:,0],
            y=random_chain_3d[:,1],
            z=random_chain_3d[:,2],

            mode="lines+markers",

            marker=dict(
                size=5
            ),

            line=dict(
                width=6
            )
        ),

        row=1,
        col=1
    )


    fig.add_trace(
        go.Scatter3d(
            x=protein_3d[:,0],
            y=protein_3d[:,1],
            z=protein_3d[:,2],

            mode="lines+markers",

            marker=dict(
                size=5
            ),

            line=dict(
                width=6
            )
        ),

        row=1,
        col=2
    )


    fig.update_layout(
        title="Flow Matching: Structure Transformation",

        height=700,
        width=1200
    )


    fig.update_scenes(
        aspectmode="cube"
    )

    return {
        "figure": fig,
        "trajectory": trajectory,
        "start_structure": random_chain_3d,
        "target_structure": protein_3d
    }


import numpy as np
import plotly.graph_objects as go



def animate_protein_folding_3d(
    trajectory,
    z_amplitude=0.5,
    x_range=(-8,8),
    y_range=(-8,8),
    z_range=(-8,8),
    frame_duration=80
):

    n = trajectory[0].shape[0]


    # -----------------------------
    # Convert 2D trajectory to 3D
    # -----------------------------

    trajectory_3d = []

    z = z_amplitude * np.sin(
        np.linspace(
            0,
            4*np.pi,
            n
        )
    )


    for structure in trajectory:

        structure_3d = np.column_stack(
            [
                structure[:,0],
                structure[:,1],
                z
            ]
        )

        trajectory_3d.append(
            structure_3d
        )


    # -----------------------------
    # Create figure
    # -----------------------------

    fig = go.Figure()


    initial = trajectory_3d[0]


    fig.add_trace(
        go.Scatter3d(
            x=initial[:,0],
            y=initial[:,1],
            z=initial[:,2],

            mode="lines+markers",

            marker=dict(
                size=5
            ),

            line=dict(
                width=6
            )
        )
    )



    # -----------------------------
    # Animation frames
    # -----------------------------

    fig.frames = [

        go.Frame(

            data=[

                go.Scatter3d(

                    x=structure[:,0],
                    y=structure[:,1],
                    z=structure[:,2],

                    mode="lines+markers",

                    marker=dict(
                        size=5
                    ),

                    line=dict(
                        width=6
                    )

                )

            ],

            name=str(i)

        )

        for i, structure in enumerate(trajectory_3d)

    ]



    # -----------------------------
    # Layout
    # -----------------------------

    fig.update_layout(

        title="Flow Matching Protein Folding",

        width=900,
        height=900,

        # preserve camera during animation
        uirevision="constant",


        scene=dict(

            aspectmode="cube",


            xaxis=dict(
                range=x_range,
                visible=False
            ),

            yaxis=dict(
                range=y_range,
                visible=False
            ),

            zaxis=dict(
                range=z_range,
                visible=False
            ),


            camera=dict(

                eye=dict(
                    x=1.6,
                    y=1.6,
                    z=1.2
                )

            )

        ),



        updatemenus=[

            {

                "type":"buttons",

                "x":0.05,
                "y":0.05,


                "buttons":[


                    {

                        "label":"▶ Play",

                        "method":"animate",

                        "args":[

                            None,

                            {

                                "mode":"immediate",

                                "frame":{

                                    "duration":frame_duration,

                                    "redraw":True

                                },

                                "transition":{

                                    "duration":0

                                }

                            }

                        ]

                    },


                    {

                        "label":"⏸ Pause",

                        "method":"animate",

                        "args":[

                            [None],

                            {

                                "mode":"immediate",

                                "frame":{

                                    "duration":0,

                                    "redraw":False

                                },

                                "transition":{

                                    "duration":0

                                }

                            }

                        ]

                    }

                ]

            }

        ],



        sliders=[

            {

                "active":0,


                "currentvalue":{

                    "prefix":"Step: "

                },


                "steps":[


                    {

                        "method":"animate",

                        "label":f"{i/(len(trajectory_3d)-1):.2f}",


                        "args":[

                            [str(i)],

                            {

                                "mode":"immediate",

                                "frame":{

                                    "duration":0,

                                    "redraw":True

                                },

                                "transition":{

                                    "duration":0

                                }

                            }

                        ]

                    }


                    for i in range(len(trajectory_3d))

                ]

            }

        ]

    )


    return {
        "figure": fig,
        "trajectory_3d": trajectory_3d
    }
