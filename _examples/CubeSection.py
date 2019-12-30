#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# This example is a derivative of vtk's ClipCow
# It is a visualization of Prince Rupert's problem

try:
    import tkinter
except ImportError:
    import Tkinter as tkinter
import vtk
from vtk.tk.vtkTkRenderWindowInteractor import \
     vtkTkRenderWindowInteractor
from numpy import array, cross
from vtk.util.colors import peacock, tomato

#cube = vtk.vtkBYUReader()
#cube.SetGeometryFileName("./cube.g")
cube = vtk.vtkCubeSource()
cube.SetCenter(0.5,0.5,0.5)

cubeNormals = vtk.vtkPolyDataNormals()
cubeNormals.SetInputConnection(cube.GetOutputPort())
cube = cubeNormals

def PlaneNormal(a, b, c):
    assert(a >= 0 and a <= 1)
    assert(b >= 0 and b <= 1)
    assert(c >= 0 and c <= 2)
    A = array([a, 0.0, 0.0])
    B = array([0.0, b, 0.0])
    if c < 1:
        C = array([c, 0.0, 1.0])
    else:
        C = array([1.0, c-1.0, 1.0])
    BA = B-A
    CA = C-A
    return cross(BA, CA)

def CuttingPlane(a, b, c):
    N = PlaneNormal(a,b,c)
    plane = vtk.vtkPlane()
    plane.SetOrigin(a, 0, 0)
    plane.SetNormal(*N)
    return plane

def getAxes(Origin, scale = 1):
    axes = vtk.vtkAxes()
    axes.SetOrigin(*Origin)
    axes.SetScaleFactor(scale)

    axesTubes = vtk.vtkTubeFilter()

    axesTubes.SetInputConnection(axes.GetOutputPort())
    axesTubes.SetRadius(0.01)
    axesTubes.SetNumberOfSides(6)

    axesMapper = vtk.vtkPolyDataMapper()
    axesMapper.SetInputConnection(axesTubes.GetOutputPort())

    axesActor = vtk.vtkActor()
    axesActor.SetMapper(axesMapper)

    XText = vtk.vtkVectorText()
    XText.SetText("x")

    XTextMapper = vtk.vtkPolyDataMapper()
    XTextMapper.SetInputConnection(XText.GetOutputPort())

    XActor = vtk.vtkFollower()
    XActor.SetMapper(XTextMapper)
    XActor.SetScale(.1, .1, .1)
    XActor.SetPosition(1, Origin[1], Origin[2])
    XActor.GetProperty().SetColor(0, 0, 0)

    YText = vtk.vtkVectorText()
    YText.SetText("y")

    YTextMapper = vtk.vtkPolyDataMapper()
    YTextMapper.SetInputConnection(YText.GetOutputPort())

    YActor = vtk.vtkFollower()
    YActor.SetMapper(YTextMapper)
    YActor.SetScale(.1, .1, .1)
    YActor.SetPosition(Origin[0], 1, Origin[2])
    YActor.GetProperty().SetColor(0, 0, 0)
    return axesActor, XActor, YActor

axesActor, XActor, YActor = getAxes([-.1, -.1, -.1])


P = (0.75, 0.75, 1.25)

# vtkClipPolyData requires an implicit function to define what it is to
# clip with. Any implicit function, including complex boolean combinations
# can be used. Notice that we can specify the value of the implicit function
# with the SetValue method.
clipper = vtk.vtkClipPolyData()
clipper.SetInputConnection(cubeNormals.GetOutputPort())
clipper.SetClipFunction(CuttingPlane(*P))

clipper.GenerateClipScalarsOn()
clipper.GenerateClippedOutputOn()
clipper.SetValue(0)
clipMapper = vtk.vtkPolyDataMapper()
clipMapper.SetInputConnection(clipper.GetOutputPort())
clipMapper.ScalarVisibilityOff()
backProp = vtk.vtkProperty()
backProp.SetDiffuseColor(tomato)
clipActor = vtk.vtkActor()
clipActor.SetMapper(clipMapper)
clipActor.GetProperty().SetColor(peacock)
clipActor.SetBackfaceProperty(backProp)

# Here we are cutting the cube. Cutting creates lines where the cut
# function intersects the model. (Clipping removes a portion of the
# model but the dimension of the data does not change.)
#
# The reason we are cutting is to generate a closed polygon at the
# boundary of the clipping process. The cutter generates line
# segments, the stripper then puts them together into polylines. We
# then pull a trick and define polygons using the closed line
# segements that the stripper created.
cutEdges = vtk.vtkCutter()
cutEdges.SetInputConnection(cubeNormals.GetOutputPort())
cutEdges.SetCutFunction(CuttingPlane(*P))
cutEdges.GenerateCutScalarsOn()
cutEdges.SetValue(0, 0)
cutStrips = vtk.vtkStripper()
cutStrips.SetInputConnection(cutEdges.GetOutputPort())
cutStrips.Update()
cutPoly = vtk.vtkPolyData()
cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
cutPoly.SetPolys(cutStrips.GetOutput().GetLines())

# Triangle filter is robust enough to ignore the duplicate point at
# the beginning and end of the polygons and triangulate them.
cutTriangles = vtk.vtkTriangleFilter()
cutTriangles.SetInput(cutPoly)
cutMapper = vtk.vtkPolyDataMapper()
cutMapper.SetInput(cutPoly)
cutMapper.SetInputConnection(cutTriangles.GetOutputPort())
cutActor = vtk.vtkActor()
cutActor.SetMapper(cutMapper)
cutActor.GetProperty().SetColor(peacock)

# The clipped part of the cube is rendered wireframe.
restMapper = vtk.vtkPolyDataMapper()
restMapper.SetInput(clipper.GetClippedOutput())
restMapper.ScalarVisibilityOff()
restActor = vtk.vtkActor()
restActor.SetMapper(restMapper)
restActor.GetProperty().SetRepresentationToWireframe()

# Create graphics stuff
renWin = vtk.vtkRenderWindow()
ren = vtk.vtkRenderer()
ren.SetBackground(tomato)
renWin.AddRenderer(ren)

# Add the actors to the renderer, set the background and size
ren.AddActor(clipActor)
ren.AddActor(cutActor)
ren.AddActor(restActor)
ren.AddActor(axesActor)
ren.AddActor(XActor)
ren.AddActor(YActor)
ren.SetBackground(1, 1, 1)
ren.ResetCamera()
camera = ren.GetActiveCamera()
camera.SetFocalPoint(0.489125, 0.481143, 0.445)
camera.SetPosition(-0.870854, -1.51779, 3.14336)
camera.SetParallelScale(1.00818)
camera.SetParallelProjection(1)
camera.SetViewUp(-0.239476, 0.833984, 0.497114)
ren.ResetCameraClippingRange()

renWin.SetSize(400, 400)

def Cut(v):
    Q = (P[0], P[1], v)
    cp = CuttingPlane(*Q)
    pn = PlaneNormal(*Q)
    clipper.SetClipFunction(cp)
    clipper.SetValue(0)
    cutEdges.SetCutFunction(cp)
    cutEdges.SetValue(0, 0)
    cutStrips.Update()
    cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
    cutPoly.SetPolys(cutStrips.GetOutput().GetLines())
    cutMapper.Update()
    renWin.Render()
 
root = tkinter.Tk()
vtkw = vtkTkRenderWindowInteractor(root, rw=renWin, width=800)

def set_cut(sz):
    sz = float(sz)
    # print(ren.GetActiveCamera())
    Cut(sz)

# propagate this GUI setting to the corresponding VTK object.
size_slider = tkinter.Scale(root, from_=0.0,
                            to=2.0, res=0.01,
                            orient='horizontal', label="Clipping Center", 
                            command=set_cut)

size_slider.set(P[2])
vtkw.Initialize()
size_slider.pack(side="top", fill="both")
vtkw.pack(side="top", fill='both', expand=1)


# Define a quit method that exits cleanly.
def quit(obj=root):
    obj.quit()

root.protocol("WM_DELETE_WINDOW", quit)

renWin.Render()
vtkw.Start()

# start the Tkinter event loop.
root.mainloop()

# end of file
