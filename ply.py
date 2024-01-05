import numpy as np
import os
import pdb

class Ply(object):
    """Class to represent a ply in memory, read plys, and write plys.
    """

    def __init__(self, ply_path=None, triangles=None, points=None, normals=None, colors=None):
        """Initialize the in memory ply representation.

        Args:
            ply_path (str, optional): Path to .ply file to read (note only
                supports text mode, not binary mode). Defaults to None.
            triangles (numpy.array [k, 3], optional): each row is a list of point indices used to
                render triangles. Defaults to None.
            points (numpy.array [n, 3], optional): each row represents a 3D point. Defaults to None.
            normals (numpy.array [n, 3], optional): each row represents the normal vector for the
                corresponding 3D point. Defaults to None.
            colors (numpy.array [n, 3], optional): each row represents the color of the
                corresponding 3D point. Defaults to None.
        """
        super().__init__()
        # TODO: If ply path is None, load in triangles, point, normals, colors.
        #       else load ply from file. If ply_path is specified AND other inputs
        #       are specified as well, ignore other inputs.
        # TODO: If normals are not None make sure that there are equal number of points and normals.
        # TODO: If colors are not None make sure that there are equal number of colors and normals.
        
        if ply_path:
            self.triangles = np.empty((0,3), dtype = np.intc)
            self.points = np.empty((0,3), dtype = np.float32)
            self.normals = np.empty((0,3), dtype = np.float32)
            self.colors = np.empty((0,3), dtype = np.ubyte)
            self.read(ply_path)
        else:
            self.points = points
            self.normals = normals
            self.colors = colors
            self.triangles = triangles

            if self.normals is not None:
                assert len(self.points) == len(self.normals), "number of normals does not match number of points..."
            if self.colors is not None:
                assert len(self.normals) == len(self.colors), "number of colors does not match number of normals..."


    def write(self, ply_path):
        """Write mesh, point cloud, or oriented point cloud to ply file.

        Args:
            ply_path (str): Output ply path.
        """
        # TODO: Write header depending on existance of normals, colors, and triangles.
        # TODO: Write points.
        # TODO: Write normals if they exist.
        # TODO: Write colors if they exist.
        # TODO: Write face list if needed.

        fp = open(ply_path, 'w')

        # header
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {len(self.points)}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        if self.normals is not None:
            fp.write("property float nx\n")
            fp.write("property float ny\n")
            fp.write("property float nz\n")
        if self.colors is not None:
            fp.write("property uchar red\n")
            fp.write("property uchar green\n")
            fp.write("property uchar blue\n")
        if self.triangles is not None:
            fp.write(f"element face {len(self.triangles)}\n")
            fp.write("property list uchar int vertex_index\n")
            fp.write("end_header\n")
        else:
            fp.write("end_header\n")
        
        # vertex data
        for idx in range(len(self.points)):
            point = self.points[idx]
            fp.write(f"{point[0]} {point[1]} {point[2]}")
            if self.normals is not None: 
                normal = self.normals[idx]
                fp.write(f" {normal[0]} {normal[1]} {normal[2]}")
                if self.colors is not None:            
                    color = self.colors[idx]
                    fp.write(f" {color[0]} {color[1]} {color[2]}")
            fp.write("\n")

        # triangle data
        if self.triangles is not None:
            face_cnt = len(self.triangles)
            for idx in range(face_cnt):
                face = self.triangles[idx]
                fp.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        fp.close()

    def read(self, ply_path):
        with open(ply_path, 'r') as fp:
            raw_content = [line.strip() for line in fp]
        
        vertex_cnt = int(raw_content[2].split(" ")[-1])
        tmp = raw_content.index("end_header")
        raw_data = raw_content[tmp+1:]
        for data in raw_data:
            data = data.split(" ")
            if len(self.points) == vertex_cnt: 
                self.triangles = np.append(self.triangles, np.array([data[1:]], dtype = np.intc), axis = 0)
            else:
                xyz = data[:3]
                normal = data[3:6]
                color = data[6:]
                self.points = np.append(self.points, np.array([xyz], dtype = np.float32), axis = 0)
                self.normals = np.append(self.normals, np.array([normal], dtype = np.float32), axis = 0)
                self.colors = np.append(self.colors, np.array([color], dtype = np.ubyte), axis = 0)

