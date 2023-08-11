# Centerline-Extraction
## Intro
Both codes were written based on the [ExtractCenterline](https://github.com/vmtk/SlicerExtension-VMTK/blob/master/ExtractCenterline/ExtractCenterline.py) in [vmtk/SlicerExtension-VMTK](https://github.com/vmtk/SlicerExtension-VMTK)

`centerline_extraction_slicer.py` was to be used with 3Dslicer python prompt.

`centerline_extracion_python.py` was to be used __without__ the slicer program.


## Environment

```
conda create -n vmtk
conda activate vmtk
conda install pip python=3.6.4 numpy=1.11.3
conda install -c vmtk vtk itk vmtk
```

## Error
The `Segmentation Fault` error could happen.

In my case, it was due to the vtk version difference.

Instead of loading the nifti file, processing a vtk file created in higher vtk `ex) 9.1.0` could lead to the error in lower vtk `ex) 8.1.0`

In this case, you could transform your vtk file 9.1.0 in `stl` format and load the stl file ([refer](https://github.com/donny8/Centerline-Extraction/blob/5648b42003a09133744de2ca2b38e444420a1194/centerline_extraction_python.py#L102)). 

