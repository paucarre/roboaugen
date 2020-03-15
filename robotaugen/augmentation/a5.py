
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

def get_model_3d_renderer():
    filename = "test/data/teapot.stl"
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(reader.GetOutput())
    else:
        mapper.SetInputConnection(reader.GetOutputPort())
    model_actor = vtk.vtkActor()
    model_actor.SetMapper(mapper)
    return model_actor

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
    xc = origin[0] + 0.5*(extent[0] + extent[1]) * spacing[0]
    yc = origin[1] + 0.5*(extent[2] + extent[3]) * spacing[1]
    xd = (extent[1] - extent[0] + 1) * spacing[0]
    yd = (extent[3] - extent[2] + 1) * spacing[1]
    d = camera.GetDistance()
    camera.SetParallelScale(0.5 * yd)
    camera.SetFocalPoint(xc, yc, 0.0)
    camera.SetPosition(xc, yc, d)
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
'''

/**
 * Convert standard camera intrinsic and extrinsic parameters to a vtkCamera instance for rendering
 * Assume square pixels and 0 skew (for now).
 *
 * focal_len : camera focal length (units pixels)
 * nx,ny : image dimensions in pixels
 * principal_pt: camera principal point,
 *    i.e. the intersection of the principal ray with the image plane (units pixels)
 * camera_rot, camera_trans : rotation, translation matrix mapping world points to camera coordinates
 * depth_min, depth_max : needed to set the clipping range
 *
 **/
 ''
def make_vtk_camera(focal_len,nx, ny,principal_pt, camera_rot,camera_trans, depth_min,  depth_max):

    camera = vtk.vtkCamera


    // convert camera rotation and translation into a 4x4 homogeneous transformation matrix
    vtkSmartPointer<vtkMatrix4x4> camera_RT = make_transform(camera_rot, camera_trans);
    // apply the transform to scene objects
    camera->SetModelTransformMatrix( camera_RT );

    # the camera can stay at the origin because we are transforming the scene objects
    camera.SetPosition(0, 0, 0)
    # look in the +Z direction of the camera coordinate system
    camera.SetFocalPoint(0, 0, 1)
    # the camera Y axis points down
    camera.SetViewUp(0,-1,0)

    # ensure the relevant range of depths are rendered
    camera.SetClippingRange(depth_min, depth_max)

    # convert the principal point to window center (normalized coordinate system) and set it
    double wcx = -2*(principal_pt.x() - double(nx)/2) / nx
    double wcy =  2*(principal_pt.y() - double(ny)/2) / ny
    camera.SetWindowCenter(wcx, wcy)

    # convert the focal length to view angle and set it
    double view_angle = vnl_math::deg_per_rad * (2.0 * std::atan2( ny/2.0, focal_len ))
    std::cout << "view_angle = " << view_angle << std::endl
    camera.SetViewAngle( view_angle )

    return camera


/** 
 * Helper function: Convert rotation and translation into a vtk 4x4 homogeneous transform
 */
vtkSmartPointer<vtkMatrix4x4> make_transform(vgl_rotation_3d<double> const& R,
                                             vgl_vector_3d<double> const& T)
{
  vtkSmartPointer<vtkMatrix4x4> m = vtkSmartPointer<vtkMatrix4x4>::New();
  vnl_matrix_fixed<double,3,3> R_mat = R.as_matrix();
  for (int r=0; r<3; ++r) {
    for (int c=0; c<3; ++c) {
      m->SetElement(r,c,R_mat[r][c]);
    }
  }
  m->SetElement(0,3,T.x());
  m->SetElement(1,3,T.y());
  m->SetElement(2,3,T.z());
  m->SetElement(3,3,1);

  return m;
}
'''

def main(argv):
    image_actor, origin, spacing, extent = get_background(argv[1])
    background_renderer = get_background_renderer()
    model_actor = get_model_3d_renderer()
    model_actor.RotateY(45)
    light = get_light()
    camera = get_camera()
    scene_renderer = get_scene_renderer(light, model_actor, camera)
    render_window = get_renderer_window(scene_renderer, background_renderer)
    background_renderer.AddActor(image_actor)
    camera = background_renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    set_camera_parameters(camera, origin, extent, spacing)
    render_window.Render()
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.Start()
    write_to_file(render_window)
    

if __name__ == '__main__':
    main(sys.argv)