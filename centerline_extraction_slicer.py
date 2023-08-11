# Execute with the command below at the terminal
# "C:\path\to\Slicer\program\Slicer.exe" --python-script C:\path\to\your\code\centerline_extraction_slicer.py
# (refer https://www.slicer.org/wiki/Documentation/Nightly/Developers/Python_scripting)

#  Written by Do Kim <donny8@naver.com> 

import os
import pickle
import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
import utils
import ExtractCenterline_slicer
import vtk.util.numpy_support

SAVE_INFO = True

# Specific to my dataset 
case_id = '00000000'
segmentName = 'Segment_1'
serverPath = r"\path\to\your\data"

if __name__=='__main__':

    for i in range(1):
        img = nib.load(f'{serverPath}\{case_id}\img.nii.gz')
        affine = img.affine
        for segmentationName in ['label_left.nii.gz']:
            targetClass = segmentationName.split('.nii.gz')[0].split('label_')[1] # Specific to my dataset 

            endpointName = f"Endpoints_{case_id}_{targetClass}"
            endpointmodelName = f"Endptmodel_{case_id}_{targetClass}"
            centermodelName = f"Model_{case_id}_{targetClass}"
            voronoimodelName = f"Voronoi_{case_id}_{targetClass}"
            centertableName = f"Properties_{case_id}_{targetClass}"
            centercurveName = f"Curve_{case_id}_{targetClass}"
            

            # Slicer add node
            endpointModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", endpointmodelName)
            centerlineModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", centermodelName)
            voronoiModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", voronoimodelName)
            centerlinePropertiesTableNode =  slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode" ,centertableName)
            centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", centercurveName)

            # Step1: Load Segmentation: From Path to 'vtkMRMLSegmentationNode' type           
            seg_path = f"{serverPath}\{case_id}\{segmentationName}" # Specific to my dataset 
            if not(os.path.exists(seg_path)):
                continue
            segmentationNode = utils.loadSegmentation(seg_path)
            segmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)

            #from vmtk.ExtractCenterline import ExtractCenterline
            extractLogic = ExtractCenterline_slicer.ExtractCenterlineLogic()

            # Step2: SegmentationNode to vtkPolyData
            inputSurfacePolyData = extractLogic.polyDataFromNode(segmentationNode, segmentID)
            print('DEBUG', inputSurfacePolyData)
            #writer = vtk.vtkSTLWriter()
            #writer.SetInputData(inputSurfacePolyData)
            #writer.SetFileName("./inputSurfacePolyData.stl")
            #writer.Write()            


            targetNumberOfPoints = 5000.0
            decimationAggressiveness = 4 # I had to lower this to 3.5 in at least one case to get it to work, 4 is the default in the module
            subdivideInputSurface = False
            preprocessedPolyData = extractLogic.preprocess(inputSurfacePolyData, targetNumberOfPoints, decimationAggressiveness, subdivideInputSurface)

            #writer = vtk.vtkPolyDataWriter()
            #writer.SetInputData(preprocessedPolyData)
            #writer.SetFileName("./preprocessedPolyData.vtk")
            #writer.Write()            

            # Step3: Extract Centerline Network (Approximated Centerline)
            endPointsMarkupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", endpointName)
            networkPolyData = extractLogic.extractNetwork(preprocessedPolyData, endPointsMarkupsNode, computeGeometry=True)  # Voronoi 계산 이미 진행. Radius 값 보유

            # Create Centerline Model
            endpointModelNode.SetAndObserveMesh(networkPolyData)

            # Step4: Get EndPoints ( AutoDetect )
            startPointPosition=None
            endpointPositions = extractLogic.getEndPoints(networkPolyData, startPointPosition) # AutoDetect the endpoints. type: List
            endPointsMarkupsNode.RemoveAllControlPoints()
            for position in endpointPositions:
                endPointsMarkupsNode.AddControlPoint(vtk.vtkVector3d(position))

            # Step5: Extract Centerline, Voronoi
            centerlinePolyData, voronoiDiagramPolyData = extractLogic.extractCenterline(preprocessedPolyData, endPointsMarkupsNode)
            centerlineModelNode.SetAndObserveMesh(centerlinePolyData)          
            voronoiModelNode.SetAndObserveMesh(voronoiDiagramPolyData)  

            # Step6: Extract centerlineCurves
            mergedCenterlines, centerlineProperties, cell_pt = extractLogic.createCurveTreeFromCenterline(centerlinePolyData, centerlineCurveNode, centerlinePropertiesTableNode) 


            # Step7: Extract centerlineCurve info
            if(SAVE_INFO):
                r1 = mergedCenterlines.GetPointData().GetArray('Radius')
                radius_arr = vtk.util.numpy_support.vtk_to_numpy(r1)        
                with open(rf'{serverPath}\{case_id}\centerCurve_{targetClass}_radius.pickle', 'wb') as f:
                    pickle.dump(radius_arr, f, pickle.HIGHEST_PROTOCOL)

                properties_dict = {}
                for columnName in [extractLogic.lengthArrayName, extractLogic.curvatureArrayName, extractLogic.torsionArrayName, extractLogic.tortuosityArrayName]:
                    vtk_arr = centerlineProperties.GetPointData().GetArray(columnName)
                    properties_dict[columnName] = vtk.util.numpy_support.vtk_to_numpy(vtk_arr)
                with open(rf'{serverPath}\{case_id}\centerCurve_{targetClass}_property_dict.pickle', 'wb') as f:
                    pickle.dump(properties_dict, f, pickle.HIGHEST_PROTOCOL)
                    

                with open(rf'{serverPath}\{case_id}\centerCurve_{targetClass}_cell_idx.pickle', 'wb') as f:
                    pickle.dump(cell_pt, f, pickle.HIGHEST_PROTOCOL)

                vtk_arr = mergedCenterlines.GetPoints().GetData()
                array = vtk.util.numpy_support.vtk_to_numpy(vtk_arr)
                coord = {}
                for cell in cell_pt:
                    coord[cell] = apply_affine(np.linalg.inv(affine), array[cell_pt[cell]])
                with open(rf'{serverPath}\{case_id}\centerCurve_{targetClass}_coord.pickle', 'wb') as f:
                    pickle.dump(coord, f, pickle.HIGHEST_PROTOCOL)
