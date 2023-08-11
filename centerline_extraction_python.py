# Based on the 
#  ExtractCenterline.py(https://github.com/vmtk/SlicerExtension-VMTK/blob/master/ExtractCenterline/ExtractCenterline.py) 
#   in vmtk/SlicerExtension-VMTK(https://github.com/vmtk/SlicerExtension-VMTK)

#  Written by Do Kim <donny8@naver.com> 

import os
import vtk
import vmtk
from vmtk import vmtkscripts
from vtk.util import numpy_support as nps
import numpy as np
import nibabel as nib
import time

blankingArrayName = 'Blanking'
radiusArrayName = 'Radius'  
groupIdsArrayName = 'GroupIds'
centerlineIdsArrayName = 'CenterlineIds'
tractIdsArrayName = 'TractIds'
topologyArrayName = 'Topology'
marksArrayName = 'Marks'
lengthArrayName = 'Length'
curvatureArrayName = 'Curvature'
torsionArrayName = 'Torsion'
tortuosityArrayName = 'Tortuosity'
frenetTangentArrayName = 'FrenetTangent'
frenetNormalArrayName = 'FrenetNormal'
frenetBinormalArrayName = 'FrenetBinormal'
curveSamplingDistance = 1.0

SAVE_VTK = True
SAVE_INFO = True
FILE_PATH = r'path\to\nifti\file'
SAVE_PATH = r'path\to\save\intermediate\outputs'

def getClosedSegmentMesh_(volume, label_id, decimation=0.0, smoothing=0.5, generate_normal=True):
    marchingCubes = vtk.vtkMarchingCubes()
    marchingCubes.SetInputData(volume)
    marchingCubes.ComputeGradientsOff()
    marchingCubes.ComputeNormalsOff()       # While computing normals is faster using the flying edges filter,
    marchingCubes.ComputeScalarsOff()

    marchingCubes.SetValue(0, label_id)
    marchingCubes.Update()
    result_poly = marchingCubes.GetOutput()

    if smoothing > 0:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(result_poly)
        smoother.SetNumberOfIterations(20)          # based on VTK documentation ("Ten or twenty iterations is all the is usually necessary")
        
        passBand = pow(10.0, -4.0 * smoothing)
        smoother.SetPassBand(passBand)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        result_poly = smoother.GetOutput()

    if generate_normal:
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(result_poly)
        normals.SplittingOff()
        normals.Update()
        result_poly = normals.GetOutput()

    return result_poly

def save_poly(save, polydata, path):
    if(save):
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(polydata)
        writer.SetFileName(path)
        writer.Write()
    else:
        pass


if __name__ == '__main__':
    start = time.time()

    img = nib.load(FILE_PATH)

    ni = img
    header = ni.header
    npVolume = ni.get_fdata()
    spacing = ni.affine.diagonal()[:3]
    origin = ni.affine[:,3][:3]

    vtk_array = nps.numpy_to_vtk(npVolume.ravel(order='F'), deep=True, array_type=vtk.VTK_DOUBLE)
    imageVol = vtk.vtkImageData()
    imageVol.SetDimensions(npVolume.shape)
    imageVol.SetSpacing(spacing)
    imageVol.SetOrigin(origin)
    imageVol.GetPointData().SetScalars(vtk_array)

    inputSurfacepolyData = getClosedSegmentMesh_(imageVol, label_id=1, decimation=0.0, smoothing=0.5, generate_normal=True)
    save_poly(SAVE_VTK, inputSurfacepolyData, f"./{SAVE_PATH}/00_inputSurfacepolyData.vtk")

    # In case of loading the stl file
    #reader = vtk.vtkSTLReader()
    #reader.SetFileName('./temp/inputSurfacePolyData.stl')
    #reader.Update()
    #inputSurfacepolyData= reader.GetOutput()

    # ============================================================================================================================================================================ #
    # ============================================================================== preprocessing =============================================================================== #
    # ============================================================================================================================================================================ #

    targetNumberOfPoints = 5000.0
    subdivideInputSurface = False

    numberOfInputPoints = inputSurfacepolyData.GetNumberOfPoints()
    reductionFactor = (numberOfInputPoints-targetNumberOfPoints) / numberOfInputPoints
    if reductionFactor > 0.0:
        decimation = vmtkscripts.vmtkSurfaceDecimation()
        decimation.Surface = inputSurfacepolyData
        decimation.TargetReduction = reductionFactor
        decimation.Execute()
        surfacePolyData = decimation.Surface

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surfacePolyData)
    normals.SetAutoOrientNormals(1)
    normals.SetFlipNormals(0)
    normals.SetConsistency(1)
    normals.SplittingOff()
    normals.Update()

    preprocessedPolyData =  normals.GetOutput()
    save_poly(SAVE_VTK, preprocessedPolyData, f"./{SAVE_PATH}/01_preprocessedPolyData.vtk")

    # ============================================================================================================================================================================ #
    # ============================================================================== extractNetwork ============================================================================== #
    # ============================================================================================================================================================================ #


    cleaner = vtk.vtkCleanPolyData() 
    cleaner.SetInputData(preprocessedPolyData)
    triangleFilter = vtk.vtkTriangleFilter() 
    triangleFilter.SetInputConnection(cleaner.GetOutputPort()) 
    triangleFilter.Update()
    simplifiedPolyData = triangleFilter.GetOutput() 

    bounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    simplifiedPolyData.GetBounds(bounds)
    startPosition = [bounds[0], bounds[2], bounds[4]]

    pointLocator = vtk.vtkPointLocator() 
    pointLocator.SetDataSet(simplifiedPolyData) 
    pointLocator.BuildLocator() 
    holePointIndex = pointLocator.FindClosestPoint(startPosition) 

    cellIds = vtk.vtkIdList()
    simplifiedPolyData.BuildCells()
    simplifiedPolyData.BuildLinks(0)
    simplifiedPolyData.GetPointCells(holePointIndex, cellIds)   # Segmentation Fault-2

    removeFirstCell = True
    if removeFirstCell:
        if cellIds.GetNumberOfIds() > 0:
            simplifiedPolyData.DeleteCell(cellIds.GetId(0))
            simplifiedPolyData.RemoveDeletedCells()

    networkExtraction = vmtkscripts.vmtkNetworkExtraction()
    networkExtraction.Surface = simplifiedPolyData
    networkExtraction.AdvancementRatio = 1.05
    networkExtraction.RadiusArrayName = radiusArrayName
    networkExtraction.TopologyArrayName = topologyArrayName
    networkExtraction.MarksArrayName = marksArrayName    
    networkExtraction.Execute()

    centerlineGeometry = vmtkscripts.vmtkCenterlineGeometry()
    centerlineGeometry.Centerlines = networkExtraction.Network
    centerlineGeometry.LengthArrayName = lengthArrayName
    centerlineGeometry.CurvatureArrayName = curvatureArrayName
    centerlineGeometry.TorsionArrayName = torsionArrayName
    centerlineGeometry.TortuosityArrayName = tortuosityArrayName
    centerlineGeometry.FrenetTangentArrayName = frenetTangentArrayName
    centerlineGeometry.FrenetNormalArrayName = frenetNormalArrayName
    centerlineGeometry.FrenetBinormalArrayName = frenetBinormalArrayName
    centerlineGeometry.Execute()
    networkPolyData = centerlineGeometry.Centerlines
    save_poly(SAVE_VTK, networkPolyData, f"./{SAVE_PATH}/02_centerlineNetwork.vtk")

    # ============================================================================================================================================================================ #
    # ======================================================================= getEndPoints (AutoDetection) ======================================================================= #
    # ============================================================================================================================================================================ #

    startPointPosition=None

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(networkPolyData)
    cleaner.Update()
    network = cleaner.GetOutput() 
    network.BuildCells() 
    network.BuildLinks(0) 
    networkPoints = network.GetPoints() 
    radiusArray = network.GetPointData().GetArray(radiusArrayName) 
    

    startPointId = -1
    maxRadius = 0
    minDistance2 = 0

    endpointIds = vtk.vtkIdList()
    for i in range(network.GetNumberOfCells()):
        numberOfCellPoints = network.GetCell(i).GetNumberOfPoints() 
        if numberOfCellPoints < 2:
            continue

        for pointIndex in [0, numberOfCellPoints - 1]:
            pointId = network.GetCell(i).GetPointId(pointIndex)
            pointCells = vtk.vtkIdList()
            network.GetPointCells(pointId, pointCells) 
            if pointCells.GetNumberOfIds() == 1:
                endpointIds.InsertUniqueId(pointId)
                if startPointPosition is not None:
                    # find start point based on position
                    position = networkPoints.GetPoint(pointId)
                    distance2 = vtk.vtkMath.Distance2BetweenPoints(position, startPointPosition)
                    if startPointId < 0 or distance2 < minDistance2:
                        minDistance2 = distance2
                        startPointId = pointId
                else:
                    # find start point based on radius
                    radius = radiusArray.GetValue(pointId)
                    if startPointId < 0 or radius > maxRadius:
                        maxRadius = radius 
                        startPointId = pointId
                        

    endpointPositions = []
    numberOfEndpointIds = endpointIds.GetNumberOfIds() 
    if numberOfEndpointIds != 0:
        # add the largest radius point first
        endpointPositions.append(networkPoints.GetPoint(startPointId))
        # add all the other points
        for pointIdIndex in range(numberOfEndpointIds):
            pointId = endpointIds.GetId(pointIdIndex)
            if pointId == startPointId:
                # already added
                continue
            endpointPositions.append(networkPoints.GetPoint(pointId))

    endPointsControlPointsPos = []
    for position in endpointPositions:
        endPointsControlPointsPos.append(vtk.vtkVector3d(position))

    # ============================================================================================================================================================================ #
    # ============================================================================= extractCenterline ============================================================================ #
    # ============================================================================================================================================================================ #

    surfaceCapper = vmtkscripts.vmtkSurfaceCapper()
    surfaceCapper.Method = 'centerpoint'
    surfaceCapper.Surface = preprocessedPolyData
    surfaceCapper.Interactive = 0
    surfaceCapper.Execute()

    tubePolyData = surfaceCapper.Surface
    save_poly(SAVE_VTK, tubePolyData, f"./{SAVE_PATH}/03_tubePolyData.vtk")

    numberOfControlPoints = len(endPointsControlPointsPos)
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(tubePolyData)
    pointLocator.BuildLocator()

    sourceIdList = []
    targetIdList = []
    pos = [0.0, 0.0, 0.0]

    for controlPointIndex in range(numberOfControlPoints):
        if controlPointIndex == 0:
            isTarget = False
        else:
            isTarget = True            

        pos = endPointsControlPointsPos[controlPointIndex]
        pointId = pointLocator.FindClosestPoint(pos)
        if isTarget:
            targetIdList.append(pointId)
        else:
            sourceIdList.append(pointId)

    centerlineFilter = vmtkscripts.vmtkCenterlines()
    centerlineFilter.Surface = tubePolyData
    centerlineFilter.SeedSelectorName = 'idlist'
    centerlineFilter.SourceIds = sourceIdList 
    centerlineFilter.TargetIds = targetIdList 
    centerlineFilter.RadiusArrayName = radiusArrayName
    centerlineFilter.CostFunction = '1/R'
    centerlineFilter.FlipNormals = False
    centerlineFilter.AppendEndPoints = 0
    centerlineFilter.SimplifyVoronoi = False
    centerlineFilter.Resampling = 0
    centerlineFilter.ResamplingStepLength = curveSamplingDistance
    centerlineFilter.Execute()

    centerlinePolyData = vtk.vtkPolyData()
    centerlinePolyData.DeepCopy(centerlineFilter.Centerlines)
    voronoiDiagramPolyData = vtk.vtkPolyData()
    voronoiDiagramPolyData.DeepCopy(centerlineFilter.VoronoiDiagram)

    save_poly(SAVE_VTK, centerlinePolyData, f"./{SAVE_PATH}/04_centerlinePolyData.vtk")
    save_poly(SAVE_VTK, voronoiDiagramPolyData, f"./{SAVE_PATH}/05_voronoiDiagramPolyData.vtk")


    # ============================================================================================================================================================================ #
    # ======================================================================= createCurveTreeFromCenterline ====================================================================== #
    # ============================================================================================================================================================================ #

    branchExtractor = vmtkscripts.vmtkBranchExtractor()
    branchExtractor.Centerlines = centerlinePolyData
    branchExtractor.BlankingArrayName = blankingArrayName
    branchExtractor.RadiusArrayName = radiusArrayName
    branchExtractor.GroupIdsArrayName = groupIdsArrayName
    branchExtractor.CenterlineIdsArrayName = centerlineIdsArrayName
    branchExtractor.TractIdsArrayName = tractIdsArrayName
    branchExtractor.Execute()
    centerlines = branchExtractor.Centerlines

    mergeCenterlines = vmtkscripts.vmtkCenterlineMerge()
    mergeCenterlines.Centerlines = centerlines
    mergeCenterlines.RadiusArrayName = radiusArrayName
    mergeCenterlines.GroupIdsArrayName = groupIdsArrayName
    mergeCenterlines.CenterlineIdsArrayName = centerlineIdsArrayName
    mergeCenterlines.TractIdsArrayName = tractIdsArrayName
    mergeCenterlines.BlankingArrayName = blankingArrayName
    mergeCenterlines.ResamplingStepLength = curveSamplingDistance
    mergeCenterlines.MergeBlanked = True
    mergeCenterlines.Execute()
    mergedCenterlines = mergeCenterlines.Centerlines
    save_poly(SAVE_VTK, mergedCenterlines, f"./{SAVE_PATH}/06_mergedCenterlines.vtk")

    # Preliminary for the Radius Calculation in each curve
    cell_pt = {}
    for cell in range(mergedCenterlines.GetNumberOfCells()):
        cell_pt[cell] = []
        getCell = mergedCenterlines.GetCell(cell)
        for idx in range(getCell.GetPointIds().GetNumberOfIds()):
            pt = getCell.GetPointIds().GetId(idx)
            cell_pt[cell].append(pt)

    centerlineBranchGeometry = vmtkscripts.vmtkBranchGeometry()
    centerlineBranchGeometry.Centerlines = mergedCenterlines

    centerlineBranchGeometry.RadiusArrayName = radiusArrayName
    centerlineBranchGeometry.GroupIdsArrayName = groupIdsArrayName
    centerlineBranchGeometry.BlankingArrayName = blankingArrayName
    centerlineBranchGeometry.LengthArrayName = lengthArrayName
    centerlineBranchGeometry.CurvatureArrayName = curvatureArrayName
    centerlineBranchGeometry.TorsionArrayName = torsionArrayName
    centerlineBranchGeometry.TortuosityArrayName = tortuosityArrayName
    centerlineBranchGeometry.LineSmoothing = False

    centerlineBranchGeometry.Execute()
    centerlineProperties = centerlineBranchGeometry.GeometryData
    curves = centerlineBranchGeometry.Centerlines

    if(SAVE_INFO):
        import pickle
        from nibabel.affines import apply_affine

        r1 = mergedCenterlines.GetPointData().GetArray('Radius')
        radius_arr = vtk.util.numpy_support.vtk_to_numpy(r1)        
        with open(f'./{SAVE_PATH}/radius.pickle', 'wb') as f: # Referring the cell_idx, get the radius in each curve
            pickle.dump(radius_arr, f, pickle.HIGHEST_PROTOCOL)

        properties_dict = {}
        for columnName in [lengthArrayName, curvatureArrayName, torsionArrayName, tortuosityArrayName]:
            vtk_arr = centerlineProperties.GetPointData().GetArray(columnName)
            properties_dict[columnName] = vtk.util.numpy_support.vtk_to_numpy(vtk_arr)
        with open(f'./{SAVE_PATH}/property_dict.pickle', 'wb') as f:
            pickle.dump(properties_dict, f, pickle.HIGHEST_PROTOCOL)
            

        with open(f'./{SAVE_PATH}/cell_idx.pickle', 'wb') as f:
            pickle.dump(cell_pt, f, pickle.HIGHEST_PROTOCOL)

        vtk_arr = mergedCenterlines.GetPoints().GetData()
        array = vtk.util.numpy_support.vtk_to_numpy(vtk_arr)
        coord_mm = {}
        coord_voxel = {}
        for cell in cell_pt:
            cell_array = array[cell_pt[cell]]
            coord_mm[cell] = cell_array
            coord_voxel[cell] = apply_affine(np.linalg.inv(ni.affine), cell_array)
        
        with open(f'./{SAVE_PATH}/coord_mm.pickle', 'wb') as f:
            pickle.dump(coord_mm, f, pickle.HIGHEST_PROTOCOL)
        with open(f'./{SAVE_PATH}/coord_voxel.pickle', 'wb') as f:
            pickle.dump(coord_voxel, f, pickle.HIGHEST_PROTOCOL)

    end = time.time() - start        
    print(f"\nElapsed Time: {end//3600:0.0f}hr {end%3600//60:0.0f}m {end%3600%60:0.0f}s")