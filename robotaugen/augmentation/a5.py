
from __future__ import print_function
import sys
import vtk 

import numpy as np

def get_camera():
    camera = vtk.vtkCamera()
    #camera.SetClippingRange(0.5, 10000)
    #camera.SetFocalPoint(0, 0.0, 0.0)
    camera.SetPosition(0.0, 0.0, -75.0)
    #camera.SetViewUp(0.0, 1.0, 0.0)
    return camera

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
    print(origin, spacing, extent)

    return image_actor, origin, spacing, extent

def set_camera_parameters(camera, origin, extent, spacing):

    xc = origin[0] + 0.5*(extent[0] + extent[1]) * spacing[0]
    yc = origin[1] + 0.5*(extent[2] + extent[3]) * spacing[1]
    xd = (extent[1] - extent[0] + 1) * spacing[0]
    yd = (extent[3] - extent[2] + 1) * spacing[1]
    focal_len = camera.GetDistance()
    print(focal_len)
    #camera.SetParallelScale(0.5 * yd)
    l = 1#np.sqrt( (xc ** 2) + (yc ** 2) )
    camera.SetFocalPoint(xc, yc, 0.0) # camera direction
    camera.SetPosition(xc, yc, focal_len)
    image_distance_y = 512 / 2
    
    view_angle = (180 / np.pi) * ( 2.0 * np.arctan2( image_distance_y / 2.0, focal_len ) )
    camera.SetViewAngle( view_angle )
    return camera

def get_renderer_window(scene_renderer, background_renderer):
    render_window = vtk.vtkRenderWindow()
    render_window.SetNumberOfLayers(2)
    render_window.AddRenderer(background_renderer)
    render_window.AddRenderer(scene_renderer)
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

def main(argv):
    image_actor, origin, spacing, extent = get_background(argv[1])
    background_renderer = get_background_renderer()
    model_actor, bounds = get_model_3d_renderer()
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    model_actor.RotateY(45)
    light = get_light()
    camera = get_camera()
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
    main(sys.argv)