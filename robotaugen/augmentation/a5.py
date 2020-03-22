
from __future__ import print_function
import sys
import vtk 

import numpy as np


def get_light():
    light = vtk.vtkLight()
    light.SetFocalPoint(0.21406, 1.5, 0)
    light.SetPosition(0.0, 4.0, 0.0)
    return light

def get_model_3d_renderer():
    filename = "test/data/model.stl"
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(reader.GetOutput())
    else:
        mapper.SetInputConnection(reader.GetOutputPort())
    model_actor = vtk.vtkActor()
    model_actor.SetMapper(mapper)
    obj = reader.GetOutputDataObject(0)
    #points = obj.GetPoints()
    #for point_id in range(points.GetNumberOfPoints()):
     #   print(points.GetPoint(point_id))
    bounds = obj.GetBounds()
    return model_actor, bounds

def get_background(image_path):
    jpeg_reader = vtk.vtkPNGReader()
    jpeg_reader.SetFileName(image_path)
    jpeg_reader.Update()

    image_data = jpeg_reader.GetOutput()
    image_actor = vtk.vtkImageActor()
    image_actor.SetInputData(image_data)

    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    extent = image_data.GetExtent()

    return image_actor, origin, spacing, extent

def set_camera_parameters(camera, origin, extent, spacing):
    xc = origin[0] +  ( 0.5 * ( extent[0] + extent[1] ) * spacing[0] )
    yc = origin[1] +  ( 0.5 * ( extent[2] + extent[3] ) * spacing[1] )
    focal_len = 35
    camera.SetFocalPoint(xc, yc, 0.0)
    camera.SetPosition(xc, yc, focal_len)
    # this is the dimensions of the background image
    image_distance_y = 1520#512
    view_angle = (180 / np.pi) * ( 2.0 * np.arctan2( image_distance_y / 2.0, focal_len ) )
    camera.SetViewAngle( view_angle )
    return camera

def get_renderer_window(scene_renderer, background_renderer):
    render_window = vtk.vtkRenderWindow()
    render_window.SetNumberOfLayers(2)
    render_window.AddRenderer(background_renderer)
    render_window.AddRenderer(scene_renderer)
    render_window.SetSize(1024,1024)
    return render_window

def get_background_renderer():
    background_renderer = vtk.vtkRenderer()
    background_renderer.SetLayer(0)
    background_renderer.InteractiveOff()
    return background_renderer

def get_scene_renderer(light, model_actor, camera):
    scene_renderer = vtk.vtkRenderer()
    scene_renderer.SetLayer(1)
    scene_renderer.AddLight(light)
    scene_renderer.AddActor(model_actor)
    scene_renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
    scene_renderer.SetActiveCamera(camera)
    return scene_renderer

def write_to_file(render_window):
    window_filter = vtk.vtkWindowToImageFilter()
    window_filter.SetInput(render_window)
    window_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName("screenshot.png")
    writer.SetInputConnection(window_filter.GetOutputPort())
    writer.Write()

def get_transformed_edge_points(transform):
    data_filter = vtk.vtkTransformPolyDataFilter()
    points = vtk.vtkPoints()
    edge_points = [
        (-75.0, 105.0, -75.0),
        (-75.0, 105.0, 75.0),
        (75.0, 105.0, 75.0),
        (75.0, 105.0, -75.0),
        (-75.0, 0.0, -75.0),
        (-75.0, 0.0, 75.0),
        (75.0, 0.0, -75.0),
        (75.0, 0.0, 75.0)]
    for edge_point in edge_points:
        points.InsertNextPoint(edge_point[0], edge_point[1], edge_point[2])
    polyPtsVTP = vtk.vtkPolyData()
    polyPtsVTP.SetPoints(points)
    data_filter.SetInputData(polyPtsVTP)
    data_filter.SetTransform(transform)
    data_filter.Update()
    output = data_filter.GetOutput()
    points = []
    for id in range(output.GetNumberOfPoints()):
        p = output.GetPoint(id)
        points.append(p)
    return points
    
def draw(filename, x, y, z, ry):
    image_actor, origin, spacing, extent = get_background(filename)
    background_renderer = get_background_renderer()
    model_actor, bounds = get_model_3d_renderer()
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    transform = vtk.vtkTransform()
    transform.PostMultiply()
    #transform.RotateX(20)
    #transform.RotateY(40)
    #transform.RotateZ(90)
    scale = 0.1
    transform.Scale(scale, scale, scale)
    transform.Translate(x, (scale * (ymin - ymax) / 2) + y , (-500.0) + z)
    model_actor.SetUserTransform(transform)
    
    #points = get_transformed_edge_points(transform)

    light = get_light()
    camera = vtk.vtkCamera()
    scene_renderer = get_scene_renderer(light, model_actor, camera)
    render_window = get_renderer_window(scene_renderer, background_renderer)
    background_renderer.AddActor(image_actor)
    camera = background_renderer.GetActiveCamera()
    camera.ParallelProjectionOff()
    set_camera_parameters(camera, origin, extent, spacing)
    render_window.Render()
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.Start()
    write_to_file(render_window)
    

if __name__ == '__main__':
    draw(filename=sys.argv[1], x=100, y=100, z=100, ry=0)