# SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin;%PATH%
# SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\lib64;%PATH%
# SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include;%PATH%
# SET PATH=C:\tools\cuda\bin;%PATH%

$env:PATH="E:\g\bin;E:\CUDA\CUDA100Dev\bin;E:\CUDA\CUDA100Dev\extras\CUPTI\libx64;E:\CUDA\CUDA100Dev\include;$env:PATH"
$env:PYTHONPATH=$(Get-Location)
.\.venv\Scripts\Activate.ps1


