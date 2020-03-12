import vtk

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

def get_axes(model_3d, renderer):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(model_3d.GetOutputPort())
    bounds = normals.GetOutput().GetBounds()
    normals.Update()
    bounds = normals.GetOutput().GetBounds()
    axes = vtk.vtkCubeAxesActor()
    axes.SetBounds(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
    axes.SetCamera(renderer.GetActiveCamera())
    axes.SetXLabelFormat("%6.1f")
    axes.SetYLabelFormat("%6.1f")
    axes.SetZLabelFormat("%6.1f")
    axes.SetFlyModeToOuterEdges()
    return axes
'''
def get_background():
    reader = vtk.vtkPNGReader()
    reader.SetFileName("test/data/tramuntana.png")

    background_image = vtk.vtkImageShrink3D()
    background_image.SetInputConnection(reader.GetOutputPort())

    geometry = vtk.vtkImageDataGeometryFilter()
    geometry.SetInputConnection(background_image.GetOutputPort())

    warp = vtk.vtkWarpScalar()
    warp.SetInputConnection(geometry.GetOutputPort())
    warp.SetScaleFactor(-0.001)

    merge = vtk.vtkMergeFilter()
    merge.SetGeometryConnection(warp.GetOutputPort())
    merge.SetScalarsConnection(reader.GetOutputPort())
    
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(merge.GetOutputPort())
    mapper.SetScalarRange(0, 255)

    return mapper
'''
def get_background():
    jpeg_reader = vtk.vtkPNGReader()
    jpeg_reader.SetFileName("test/data/tramuntana.png")
    jpeg_reader.Update()

    image_data = jpeg_reader.GetOutput()
    image_actor = vtk.vtkImageActor()
    image_actor.SetInputData(image_data)

    background_renderer = vtk.vtkRenderer()

    scene_renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()

    background_renderer.SetLayer(0)
    background_renderer.InteractiveOff()

    scene_renderer.SetLayer(1)

    render_window.SetNumberOfLayers(2)
    render_window.AddRenderer(background_renderer)
    render_window.AddRenderer(scene_renderer)
    #render_window_interactor = vtkRenderWindowInteractor()
    #render_window_interactor.SetRenderWindow(render_window)

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

    #render_window.Render()
    #render_window_interactor.Start()

    return render_window, scene_renderer


#actor = vtk.vtkActor()
#model_3d = get_model_3d()
#actor.SetMapper(model_3d)

render_window, scene_renderer =  get_background()

#camera = get_camera()
#scene_renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
#scene_renderer.SetActiveCamera(camera)
#light = get_light()
#scene_renderer.AddLight(light)

render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)
#scene_renderer.AddActor(actor)


#irenderer.Initialize()
render_window.Render()
render_window_interactor.Start()

'''
window_filter = vtk.vtkWindowToImageFilter()
window_filter.SetInput(renderer_window)
window_filter.Update()

writer = vtk.vtkPNGWriter()
writer.SetFileName("screenshot.png")
writer.SetInputConnection(window_filter.GetOutputPort())
writer.Write()
'''