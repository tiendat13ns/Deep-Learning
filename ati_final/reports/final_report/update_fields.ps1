# Update the table of contents and caption fields in Word, save, and export a PDF preview.
# Usage (from ati_final):  powershell -ExecutionPolicy Bypass -File reports/final_report/update_fields.ps1 <docx> <pdf>
param([string]$Docx, [string]$Pdf)
$Docx = (Resolve-Path $Docx).Path
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $false)
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Repaginate()
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    $doc.Save()
    if ($Pdf) { $doc.ExportAsFixedFormat($Pdf, 17) }
    # page numbers of the main body (Section 1 to the end of Section 7)
    $start = $null; $end = $null
    foreach ($p in $doc.Paragraphs) {
        $t = $p.Range.Text.Trim()
        if ($start -eq $null -and $t -eq "1  Introduction" -and $p.Style.NameLocal -eq "Heading 1") { $start = $p.Range.Information(3) }
        if ($t -eq "References" -and $p.Style.NameLocal -eq "Heading 1") { $end = $p.Range.Information(3) }
    }
    Write-Output "body_start_page=$start references_page=$end total_pages=$($doc.ComputeStatistics(2))"
    $doc.Close()
} finally {
    $word.Quit()
}
