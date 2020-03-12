#!/usr/bin/env python

import vtk
import vtk.test.Testing
from vtk.util.misc import vtkGetDataRoot

renWin = vtk.vtkRenderWindow()
renWin.SetSize(1024, 512)

image = vtk.vtkPNGReader()
image.SetFileName("test/data/tramuntana.png")


backgroundColor = vtk.vtkImageShrink3D()
backgroundColor.SetInputConnection(image.GetOutputPort())
#backgroundColor.SetShrinkFactors(1, 1, 1)

backgroundLuminance = vtk.vtkImageLuminance()
backgroundLuminance.SetInputConnection(backgroundColor.GetOutputPort())

blend = vtk.vtkImageBlend()

blend.AddInputConnection(backgroundColor.GetOutputPort())
blend.SetBlendModeToCompound()
blend.SetOpacity(1, 0.8)

mapper = vtk.vtkImageMapper()
mapper.SetInputConnection(blend.GetOutputPort())
mapper.SetColorWindow(255)
mapper.SetColorLevel(127.5)

actor =  vtk.vtkActor2D()
actor.SetMapper(mapper)

imager = vtk.vtkRenderer()
imager.AddActor2D(actor)
imager.SetViewport(0, 0, 1.0, 1.0)
imager.SetBackground(0.9, 0.3, 0.3)

renWin.AddRenderer(imager)

# render and interact with data

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
renWin.Render()
iren.Start()