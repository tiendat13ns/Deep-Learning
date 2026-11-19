# Export a .pptx to PDF with PowerPoint (fonts render exactly as on the presenting machine).
# Usage (from ati_final): powershell -ExecutionPolicy Bypass -File slides/export_pdf.ps1 <pptx> <pdf>
param([string]$Pptx, [string]$Pdf)
$Pptx = (Resolve-Path $Pptx).Path
$Pdf = [System.IO.Path]::GetFullPath($Pdf)
$app = New-Object -ComObject PowerPoint.Application
try {
    # Open(FileName, ReadOnly, Untitled, WithWindow)
    $pres = $app.Presentations.Open($Pptx, -1, 0, 0)
    $pres.SaveAs($Pdf, 32)   # ppSaveAsPDF
    Write-Output "slides=$($pres.Slides.Count) pdf=$Pdf"
    $pres.Close()
} finally {
    $app.Quit()
}
