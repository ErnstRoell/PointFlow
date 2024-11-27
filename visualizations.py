import torch


import pyvista as pv


def plot_pcs_single(pcs, offset=0):
    print(max_cols)
    max_cols = min(len(pcs), 10)
    pl = pv.Plotter(
        shape=(1, max_cols),
        window_size=[max_cols * 200, 100],
        border=False,
        polygon_smoothing=True,
    )

    for col in range(max_cols):
        points = pcs[col + offset].reshape(-1, 3).detach().cpu().numpy()
        pl.subplot(0, col)
        actor = pl.add_points(
            points,
            style="points",
            emissive=False,
            show_scalar_bar=False,
            render_points_as_spheres=True,
            scalars=points[:, 2],
            point_size=2,
            ambient=0.2,
            diffuse=0.8,
            specular=0.8,
            specular_power=40,
            smooth_shading=True,
        )

    pl.background_color = "w"
    pl.link_views()
    pl.camera_position = "xy"
    pos = pl.camera.position
    pl.camera.position = (pos[0], pos[1] + 3, pos[2])
    pl.camera.position = (6, 0, 0)
    pl.camera.azimuth = 45
    pl.camera.elevation = 30
    # create a top down light
    light = pv.Light(
        position=(0, 0, 0), positional=True, cone_angle=50, exponent=20, intensity=0.2
    )
    pl.add_light(light)
    pl.camera.zoom(1.3)
    pl.show()
