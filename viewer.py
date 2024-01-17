import numpy as np
import open3d as o3d
import math
import pdb

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(-2.0, 0.0)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

if __name__ == '__main__':
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    pcd_walle = o3d.io.read_point_cloud("supplemental/point_cloud.ply")
    mesh_walle = o3d.io.read_triangle_mesh("supplemental/mesh.ply")
    rot_mat = mesh.get_rotation_matrix_from_xyz((math.radians(-90), 0, 0))
    mesh_walle.rotate(rot_mat).translate((-1.5,0,0)).scale(10, (0,0,0))
    pcd_walle.rotate(rot_mat).translate((0,0,0)).scale(10, (0,0,0))
    
    
    o3d.visualization.draw_geometries_with_animation_callback([mesh_walle], rotate_view)
    # o3d.visualization.draw_geometries([mesh_walle], rotate_view)
    # o3d.visualization.draw_geometries_with_animation_callback([pcd_walle, mesh_walle],rotate_view)


