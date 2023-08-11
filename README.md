# Centerline-Extraction
## Info
The codes were implemented for the coronary artery centerline extraction.

Both codes were written based on the [ExtractCenterline](https://github.com/vmtk/SlicerExtension-VMTK/blob/master/ExtractCenterline/ExtractCenterline.py) in [vmtk/SlicerExtension-VMTK](https://github.com/vmtk/SlicerExtension-VMTK)

`centerline_extraction_slicer.py` was to be used __with__ the [slicer](https://www.slicer.org/) python prompt.

`centerline_extracion_python.py` was to be used __without__ the slicer.


## Environment

```
conda create -n vmtk
conda activate vmtk
conda install pip python=3.6.4 numpy=1.11.3
conda install -c vmtk vtk itk vmtk
```

## Error
The `Segmentation Fault` error could happen.

Processing a vtk file created in higher vtk `ex) 9.1.0` led to the error in lower vtk `ex) 8.1.0`

In this case, you could transform your vtk file 9.1.0 in `stl` format and load the stl file ([refer](https://github.com/donny8/Centerline-Extraction/blob/5648b42003a09133744de2ca2b38e444420a1194/centerline_extraction_python.py#L102)). 
