
from __future__ import print_function
import sys
import vtk 

def get_camera():
    camera = vtk.vtkCamera()
    camera.SetClippingRange(0.5, 10000)
    camera.SetFocalPoint(0, 0.0, 0.0)
    camera.SetPosition(0.0, 0.0, -40.0)
    camera.SetViewUp(0.0, 1.0, 0.0)
    return camera

def get_light():
    light = vtk.vtkLight()
    light.SetFocalPoint(0.21406, 1.5, 0)
    light.SetPosition(0.0, 4.0, 0.0)
    return light


def get_model_3d():
    filename = "test/data/teapot.stl"
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(reader.GetOutput())
    else:
        mapper.SetInputConnection(reader.GetOutputPort())
    return mapper

def main(argv):
    
    jpeg_reader = vtk.vtkPNGReader()
    jpeg_reader.SetFileName(argv[1])
    jpeg_reader.Update()

    image_data = jpeg_reader.GetOutput()
    image_actor = vtk.vtkImageActor()
    image_actor.SetInputData(image_data)

    background_renderer = vtk.vtkRenderer()

    superquadric_actor = vtk.vtkActor()
    model_3d = get_model_3d()
    superquadric_actor.SetMapper(model_3d)

    scene_renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()

    background_renderer.SetLayer(0)
    background_renderer.InteractiveOff()

    scene_renderer.SetLayer(1)

    render_window.SetNumberOfLayers(2)
    render_window.AddRenderer(background_renderer)
    render_window.AddRenderer(scene_renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    
    light = get_light()
    scene_renderer.AddLight(light)
    scene_renderer.AddActor(superquadric_actor)

    camera = get_camera()
    scene_renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
    scene_renderer.SetActiveCamera(camera)

    background_renderer.AddActor(image_actor)
    render_window.Render()
    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    extent = image_data.GetExtent()

    camera = background_renderer.GetActiveCamera()
    camera.ParallelProjectionOn()

    xc = origin[0] + 0.5*(extent[0] + extent[1]) * spacing[0]
    yc = origin[1] + 0.5*(extent[2] + extent[3]) * spacing[1]
    # xd = (extent[1] - extent[0] + 1) * spacing[0]
    yd = (extent[3] - extent[2] + 1) * spacing[1]
    d = camera.GetDistance()
    #camera.SetParallelScale(0.5 * yd)
    camera.SetFocalPoint(xc, yc, 0.0)
    camera.SetPosition(xc, yc, d)

    render_window.Render()
    render_window_interactor.Start()

    window_filter = vtk.vtkWindowToImageFilter()
    window_filter.SetInput(render_window)
    window_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName("screenshot.png")
    writer.SetInputConnection(window_filter.GetOutputPort())
    writer.Write()


if __name__ == '__main__':
    main(sys.argv)