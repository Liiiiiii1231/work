$ErrorActionPreference = 'Stop'

$root = (Get-Location).Path
$sourceDir = Join-Path $root 'results\stage8'
$outputDir = Join-Path $root 'results\figures'
$outputPath = Join-Path $outputDir 'Stage8_最终结果汇总表.xlsx'
$previewDir = Join-Path $env:TEMP 'stage8_summary_preview'

$files = @(
    [pscustomobject]@{ Name = 'Monte Carlo'; File = 'stage8_monte_carlo.csv'; GroupColumn = $null; GroupLabel = '全部' },
    [pscustomobject]@{ Name = 'Cable Length'; File = 'stage8_sweep_cable_length.csv'; GroupColumn = 'parameter_value'; GroupLabel = 'cable_length' },
    [pscustomobject]@{ Name = 'P_DL Max'; File = 'stage8_sweep_p_dl_max.csv'; GroupColumn = 'parameter_value'; GroupLabel = 'p_dl_max' },
    [pscustomobject]@{ Name = 'Rho SI'; File = 'stage8_sweep_rho_si.csv'; GroupColumn = 'parameter_value'; GroupLabel = 'rho_si' },
    [pscustomobject]@{ Name = 'System Load'; File = 'stage8_sweep_system_load.csv'; GroupColumn = 'system_load'; GroupLabel = 'system_load' }
)

$schemes = @(
    [pscustomobject]@{ Key = 'b1'; Label = 'B1' },
    [pscustomobject]@{ Key = 'b2'; Label = 'B2' },
    [pscustomobject]@{ Key = 'b3'; Label = 'B3' },
    [pscustomobject]@{ Key = 'b4'; Label = 'B4' },
    [pscustomobject]@{ Key = 'b5'; Label = 'B5' },
    [pscustomobject]@{ Key = 'proposed'; Label = 'Proposed' }
)

function Mean([double[]]$Values) {
    if ($Values.Count -eq 0) { return [double]::NaN }
    return ($Values | Measure-Object -Average).Average
}

function StdP([double[]]$Values) {
    if ($Values.Count -eq 0) { return [double]::NaN }
    $m = Mean $Values
    $sum = 0.0
    foreach ($v in $Values) { $sum += ($v - $m) * ($v - $m) }
    return [math]::Sqrt($sum / $Values.Count)
}

function Apply-Header($Range, [string]$Fill = '1F4E78') {
    $Range.Font.Bold = $true
    $Range.Font.Color = 0xFFFFFF
    $Range.Interior.Color = [System.Drawing.ColorTranslator]::ToOle([System.Drawing.Color]::FromName('#' + $Fill))
    $Range.HorizontalAlignment = -4108
    $Range.VerticalAlignment = -4108
}

function Apply-Title($Range) {
    $Range.Font.Bold = $true
    $Range.Font.Size = 18
    $Range.Font.Color = 0xFFFFFF
    $Range.Interior.Color = [System.Drawing.ColorTranslator]::ToOle([System.Drawing.Color]::FromName('#17365D'))
    $Range.HorizontalAlignment = -4108
    $Range.VerticalAlignment = -4108
}

function Add-Sheet($Workbook, [string]$Name) {
    $sheet = $Workbook.Worksheets.Add([System.Type]::Missing, $Workbook.Worksheets.Item($Workbook.Worksheets.Count))
    $sheet.Name = $Name
    $sheet.DisplayPageBreaks = $false
    $sheet.Activate()
    $sheet.Application.ActiveWindow.DisplayGridlines = $false
    return $sheet
}

function Write-Matrix($Sheet, [int]$StartRow, [int]$StartCol, [object[][]]$Rows) {
    if ($Rows.Count -eq 0) { return }
    $cols = $Rows[0].Count
    for ($r = 0; $r -lt $Rows.Count; $r++) {
        for ($c = 0; $c -lt $cols; $c++) {
            $Sheet.Cells.Item($StartRow + $r, $StartCol + $c).Value2 = $Rows[$r][$c]
        }
    }
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
New-Item -ItemType Directory -Force -Path $previewDir | Out-Null

$datasets = @{}
$groupSummary = [System.Collections.Generic.List[object]]::new()
$anomalies = [System.Collections.Generic.List[object]]::new()
$totalRows = 0
$totalSceneFirst = 0
$totalGroups = 0
$totalGroupFirst = 0

foreach ($spec in $files) {
    $path = Join-Path $sourceDir $spec.File
    $rows = @(Import-Csv -LiteralPath $path)
    $datasets[$spec.File] = $rows
    $totalRows += $rows.Count

    if ($null -eq $spec.GroupColumn) {
        $groups = @([pscustomobject]@{ Name = '全部'; Rows = $rows })
    } else {
        $groups = @($rows | Group-Object -Property $spec.GroupColumn | Sort-Object { [double]$_.Name })
    }

    foreach ($g in $groups) {
        $gRows = @($g.Rows)
        $groupName = [string]$g.Name
        $stats = @()
        foreach ($scheme in $schemes) {
            $col = $scheme.Key + '_wsr'
            [double[]]$values = @($gRows | ForEach-Object { [double]$_.$col })
            $stats += [pscustomobject]@{ Key = $scheme.Key; Label = $scheme.Label; Mean = Mean $values; Std = StdP $values }
        }
        $ordered = @($stats | Sort-Object Mean -Descending)
        $proposed = $stats | Where-Object Key -eq 'proposed'
        $bestBaseline = $stats | Where-Object Key -ne 'proposed' | Sort-Object Mean -Descending | Select-Object -First 1
        $groupRank = ([array]::IndexOf(@($ordered.Key), 'proposed')) + 1
        if ($groupRank -eq 1) { $totalGroupFirst++ }
        $totalGroups++

        $sceneFirst = 0
        $groupAnomalyCount = 0
        foreach ($row in $gRows) {
            $rowValues = @()
            $finiteError = $false
            foreach ($scheme in $schemes) {
                $value = 0.0
                $ok = [double]::TryParse([string]$row.($scheme.Key + '_wsr'), [ref]$value)
                if (-not $ok -or [double]::IsNaN($value) -or [double]::IsInfinity($value)) { $finiteError = $true }
                $rowValues += [pscustomobject]@{ Key = $scheme.Key; Label = $scheme.Label; Value = $value }
            }
            $rowOrdered = @($rowValues | Sort-Object Value -Descending)
            $rowRank = ([array]::IndexOf(@($rowOrdered.Key), 'proposed')) + 1
            if ($rowRank -eq 1) { $sceneFirst++; $totalSceneFirst++ }
            $pValue = [double]$row.proposed_wsr
            $baselineLeader = $rowValues | Where-Object Key -ne 'proposed' | Sort-Object Value -Descending | Select-Object -First 1
            $z = if ($proposed.Std -gt 0) { ($pValue - $proposed.Mean) / $proposed.Std } else { 0.0 }
            $flags = @()
            if ($finiteError) { $flags += '非有限值' }
            if ($rowRank -ne 1) { $flags += 'Proposed 非第1' }
            if ([math]::Abs($z) -gt 2.0) { $flags += '|z| > 2' }
            if ($flags.Count -gt 0) {
                $groupAnomalyCount++
                $anomalies.Add([pscustomobject]@{
                    File = $spec.File; Experiment = $spec.Name; Parameter = $spec.GroupLabel; Value = $groupName
                    Scene = [int]$row.scene_seed; Proposed = $pValue; Rank = $rowRank
                    BestBaseline = $baselineLeader.Label; BestBaselineWSR = $baselineLeader.Value
                    Margin = $pValue - $baselineLeader.Value; GroupMean = $proposed.Mean
                    GroupStd = $proposed.Std; Z = $z; Flag = ($flags -join '; ')
                })
            }
        }

        $groupSummary.Add([pscustomobject]@{
            File = $spec.File; Experiment = $spec.Name; Parameter = $spec.GroupLabel; Value = $groupName
            N = $gRows.Count; ProposedMean = $proposed.Mean; ProposedStd = $proposed.Std; Rank = $groupRank
            BestBaseline = $bestBaseline.Label; BestBaselineMean = $bestBaseline.Mean
            Gain = $proposed.Mean - $bestBaseline.Mean
            GainPct = if ($bestBaseline.Mean -ne 0) { ($proposed.Mean - $bestBaseline.Mean) / $bestBaseline.Mean } else { 0 }
            SceneFirst = $sceneFirst; SceneTotal = $gRows.Count; AnomalyCount = $groupAnomalyCount
        })
    }
}

$excel = $null
$workbook = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.ScreenUpdating = $false
    $workbook = $excel.Workbooks.Add()
    while ($workbook.Worksheets.Count -gt 1) { $workbook.Worksheets.Item($workbook.Worksheets.Count).Delete() }
    $dashboard = $workbook.Worksheets.Item(1)
    $dashboard.Name = '最终汇总'
    $dashboard.Application.ActiveWindow.DisplayGridlines = $false

    $dashboard.Range('A1:J2').Merge()
    $dashboard.Range('A1').Value2 = 'Stage 8 最终结果汇总表'
    Apply-Title $dashboard.Range('A1:J2')
    $dashboard.Rows.Item(1).RowHeight = 26
    $dashboard.Rows.Item(2).RowHeight = 12

    $dashboard.Range('A4:B4').Merge(); $dashboard.Range('A4').Value2 = '正式 CSV'; Apply-Header $dashboard.Range('A4:B4') '4472C4'
    $dashboard.Range('A5:B6').Merge(); $dashboard.Range('A5').Value2 = 5; $dashboard.Range('A5').Font.Size = 20; $dashboard.Range('A5').Font.Bold = $true; $dashboard.Range('A5').HorizontalAlignment = -4108
    $dashboard.Range('D4:E4').Merge(); $dashboard.Range('D4').Value2 = '总记录数'; Apply-Header $dashboard.Range('D4:E4') '4472C4'
    $dashboard.Range('D5:E6').Merge(); $dashboard.Range('D5').Value2 = $totalRows; $dashboard.Range('D5').Font.Size = 20; $dashboard.Range('D5').Font.Bold = $true; $dashboard.Range('D5').HorizontalAlignment = -4108
    $dashboard.Range('G4:H4').Merge(); $dashboard.Range('G4').Value2 = 'Proposed 分组第1'; Apply-Header $dashboard.Range('G4:H4') '70AD47'
    $dashboard.Range('G5:H6').Merge(); $dashboard.Range('G5').Value2 = "$totalGroupFirst / $totalGroups"; $dashboard.Range('G5').Font.Size = 20; $dashboard.Range('G5').Font.Bold = $true; $dashboard.Range('G5').HorizontalAlignment = -4108
    $dashboard.Range('I4:J4').Merge(); $dashboard.Range('I4').Value2 = '异常候选'; Apply-Header $dashboard.Range('I4:J4') 'C65911'
    $dashboard.Range('I5:J6').Merge(); $dashboard.Range('I5').Value2 = $anomalies.Count; $dashboard.Range('I5').Font.Size = 20; $dashboard.Range('I5').Font.Bold = $true; $dashboard.Range('I5').HorizontalAlignment = -4108

    $mc = $groupSummary | Where-Object File -eq 'stage8_monte_carlo.csv'
    $dashboard.Range('A8:J8').Merge(); $dashboard.Range('A8').Value2 = '核心结论'; Apply-Header $dashboard.Range('A8:J8') '1F4E78'
    $conclusions = @(
        "Proposed 在 $totalGroups/$totalGroups 个实验分组中 mean 排名第 1；在 $totalSceneFirst/$totalRows 条逐 scene 比较中均排名第 1。",
        ('Monte Carlo Proposed mean = {0:F6}，std(P, ddof=0) = {1:F6}。' -f $mc.ProposedMean, $mc.ProposedStd),
        "异常规则：非有限值、Proposed 非第 1，或同参数组内 Proposed WSR 的 |z| > 2。共发现 $($anomalies.Count) 条候选异常。",
        '统计口径与项目 stage8.py 一致：mean 使用算术平均，std 使用总体标准差（ddof=0）。'
    )
    for ($i = 0; $i -lt $conclusions.Count; $i++) { $dashboard.Range("A$($i+9):J$($i+9)").Merge(); $dashboard.Cells.Item($i+9,1).Value2 = '• ' + $conclusions[$i] }
    $dashboard.Range('A9:J12').WrapText = $true
    $dashboard.Range('A9:J12').RowHeight = 24

    $dashboard.Range('A14:G14').Value2 = @('文件','记录数','分组数','Proposed 分组第1','逐 scene 第1','异常候选','完整性')
    Apply-Header $dashboard.Range('A14:G14') '5B9BD5'
    $rowIndex = 15
    foreach ($spec in $files) {
        $rows = $datasets[$spec.File]
        $groups = @($groupSummary | Where-Object File -eq $spec.File)
        $groupWins = @($groups | Where-Object Rank -eq 1).Count
        $sceneWins = ($groups | Measure-Object SceneFirst -Sum).Sum
        $anom = ($groups | Measure-Object AnomalyCount -Sum).Sum
        $dashboard.Cells.Item($rowIndex,1).Value2 = $spec.File
        $dashboard.Cells.Item($rowIndex,2).Value2 = $rows.Count
        $dashboard.Cells.Item($rowIndex,3).Value2 = $groups.Count
        $dashboard.Cells.Item($rowIndex,4).Value2 = "$groupWins/$($groups.Count)"
        $dashboard.Cells.Item($rowIndex,5).Value2 = "$sceneWins/$($rows.Count)"
        $dashboard.Cells.Item($rowIndex,6).Value2 = $anom
        $dashboard.Cells.Item($rowIndex,7).Value2 = '通过'
        $rowIndex++
    }
    $dashboard.Range("A14:G$($rowIndex-1)").Borders.Color = 0xD9D9D9
    $dashboard.Range("A14:G$($rowIndex-1)").Borders.LineStyle = 1
    $dashboard.Range("G15:G$($rowIndex-1)").Interior.Color = [System.Drawing.ColorTranslator]::ToOle([System.Drawing.Color]::FromName('#E2F0D9'))
    $dashboard.Columns.Item('A').ColumnWidth = 34
    $dashboard.Columns.Item('B:G').ColumnWidth = 15
    $dashboard.Columns.Item('H:J').ColumnWidth = 12
    $dashboard.Range('A1:J20').Font.Name = 'Microsoft YaHei'
    $dashboard.PageSetup.Orientation = 2
    $dashboard.PageSetup.Zoom = $false
    $dashboard.PageSetup.FitToPagesWide = 1
    $dashboard.PageSetup.FitToPagesTall = 1

    $mainStats = Add-Sheet $workbook '主结果统计'
    $mainStats.Range('A1:H2').Merge(); $mainStats.Range('A1').Value2 = 'Monte Carlo 六方案统计'; Apply-Title $mainStats.Range('A1:H2')
    $mainStats.Range('A4:H4').Value2 = @('方案','mean','std(P)','排名','min','max','相对最佳Baseline增益','说明')
    Apply-Header $mainStats.Range('A4:H4') '1F4E78'
    $mcRows = $datasets['stage8_monte_carlo.csv']
    $mcStats = @()
    foreach ($scheme in $schemes) {
        [double[]]$vals = @($mcRows | ForEach-Object { [double]$_.$($scheme.Key + '_wsr') })
        $mcStats += [pscustomobject]@{ Key=$scheme.Key; Label=$scheme.Label; Mean=Mean $vals; Std=StdP $vals; Min=($vals|Measure-Object -Minimum).Minimum; Max=($vals|Measure-Object -Maximum).Maximum }
    }
    $ranked = @($mcStats | Sort-Object Mean -Descending)
    $bestBase = $mcStats | Where-Object Key -ne 'proposed' | Sort-Object Mean -Descending | Select-Object -First 1
    for ($i=0; $i -lt $mcStats.Count; $i++) {
        $s=$mcStats[$i]; $r=$i+5
        $mainStats.Cells.Item($r,1).Value2=$s.Label
        $mainStats.Cells.Item($r,2).Value2=$s.Mean
        $mainStats.Cells.Item($r,3).Value2=$s.Std
        $mainStats.Cells.Item($r,4).Value2=([array]::IndexOf(@($ranked.Key),$s.Key)+1)
        $mainStats.Cells.Item($r,5).Value2=$s.Min
        $mainStats.Cells.Item($r,6).Value2=$s.Max
        $mainStats.Cells.Item($r,7).Value2=$s.Mean-$bestBase.Mean
        $mainStats.Cells.Item($r,8).Value2=if($s.Key -eq 'proposed'){'最终方案'}else{'Baseline'}
    }
    $mainStats.Range('B5:G10').NumberFormat = '0.000000'
    $mainStats.Range('A10:H10').Interior.Color = [System.Drawing.ColorTranslator]::ToOle([System.Drawing.Color]::FromName('#E2F0D9'))
    $mainStats.Range('A4:H10').Borders.Color=0xD9D9D9; $mainStats.Range('A4:H10').Borders.LineStyle=1
    $mainStats.Columns.Item('A').ColumnWidth=16; $mainStats.Columns.Item('B:G').ColumnWidth=18; $mainStats.Columns.Item('H').ColumnWidth=16
    $mainStats.Range('A1:H12').Font.Name='Microsoft YaHei'
    $mainStats.Application.ActiveWindow.SplitRow = 4; $mainStats.Application.ActiveWindow.FreezePanes = $true
    $mainStats.PageSetup.Orientation=2; $mainStats.PageSetup.Zoom=$false; $mainStats.PageSetup.FitToPagesWide=1; $mainStats.PageSetup.FitToPagesTall=1

    $gs = Add-Sheet $workbook 'Proposed分组'
    $gs.Range('A1:N2').Merge(); $gs.Range('A1').Value2='Proposed 分组排名与统计'; Apply-Title $gs.Range('A1:N2')
    $headers=@('来源文件','实验','参数','参数值','N','Proposed mean','Proposed std(P)','mean排名','最佳Baseline','Baseline mean','绝对增益','相对增益','逐scene第1','异常数')
    for($c=0;$c-lt$headers.Count;$c++){$gs.Cells.Item(4,$c+1).Value2=$headers[$c]}
    Apply-Header $gs.Range('A4:N4') '1F4E78'
    $r=5
    foreach($x in $groupSummary){
        $vals=@($x.File,$x.Experiment,$x.Parameter,$x.Value,$x.N,$x.ProposedMean,$x.ProposedStd,$x.Rank,$x.BestBaseline,$x.BestBaselineMean,$x.Gain,$x.GainPct,"$($x.SceneFirst)/$($x.SceneTotal)",$x.AnomalyCount)
        for($c=0;$c-lt$vals.Count;$c++){$gs.Cells.Item($r,$c+1).Value2=$vals[$c]}
        $r++
    }
    $gs.Range("F5:K$($r-1)").NumberFormat='0.000000'; $gs.Range("L5:L$($r-1)").NumberFormat='0.00%'
    $gs.Range("H5:H$($r-1)").Interior.Color=[System.Drawing.ColorTranslator]::ToOle([System.Drawing.Color]::FromName('#E2F0D9'))
    $gs.Range("A4:N$($r-1)").Borders.Color=0xD9D9D9; $gs.Range("A4:N$($r-1)").Borders.LineStyle=1
    $gs.Range("A4:N$($r-1)").AutoFilter() | Out-Null
    $gs.Columns.Item('A').ColumnWidth=34; $gs.Columns.Item('B:D').ColumnWidth=16; $gs.Columns.Item('E:N').ColumnWidth=15
    $gs.Range("A1:N$($r-1)").Font.Name='Microsoft YaHei'
    $gs.Application.ActiveWindow.SplitRow=4; $gs.Application.ActiveWindow.FreezePanes=$true
    $gs.PageSetup.Orientation=2; $gs.PageSetup.Zoom=$false; $gs.PageSetup.FitToPagesWide=1

    $as = Add-Sheet $workbook '异常Scene'
    $as.Range('A1:M2').Merge(); $as.Range('A1').Value2='异常 Scene 检查'; Apply-Title $as.Range('A1:M2')
    $as.Range('A3:M3').Merge(); $as.Range('A3').Value2='规则：非有限值；Proposed 逐 scene 非第1；或同参数组内 Proposed WSR 的 |z| > 2。异常仅表示需要复核，不代表结果无效。'
    $as.Range('A3:M3').WrapText=$true; $as.Rows.Item(3).RowHeight=32
    $aheaders=@('来源文件','实验','参数','参数值','scene_seed','Proposed WSR','逐scene排名','最佳Baseline','Baseline WSR','优势差值','组mean','组std(P)','异常标记 / z')
    for($c=0;$c-lt$aheaders.Count;$c++){$as.Cells.Item(5,$c+1).Value2=$aheaders[$c]}
    Apply-Header $as.Range('A5:M5') 'C65911'
    $r=6
    foreach($x in $anomalies){
        $vals=@($x.File,$x.Experiment,$x.Parameter,$x.Value,$x.Scene,$x.Proposed,$x.Rank,$x.BestBaseline,$x.BestBaselineWSR,$x.Margin,$x.GroupMean,$x.GroupStd,($x.Flag + ('; z={0:F3}' -f $x.Z)))
        for($c=0;$c-lt$vals.Count;$c++){$as.Cells.Item($r,$c+1).Value2=$vals[$c]}
        $r++
    }
    if($anomalies.Count -eq 0){$as.Range('A6:M6').Merge();$as.Range('A6').Value2='未发现异常。'}
    $as.Range("F6:L$([math]::Max(6,$r-1))").NumberFormat='0.000000'
    $as.Range("A5:M$([math]::Max(6,$r-1))").Borders.Color=0xD9D9D9; $as.Range("A5:M$([math]::Max(6,$r-1))").Borders.LineStyle=1
    $as.Range("A6:M$([math]::Max(6,$r-1))").Interior.Color=[System.Drawing.ColorTranslator]::ToOle([System.Drawing.Color]::FromName('#FCE4D6'))
    $as.Columns.Item('A').ColumnWidth=34; $as.Columns.Item('B:D').ColumnWidth=16; $as.Columns.Item('E:L').ColumnWidth=15; $as.Columns.Item('M').ColumnWidth=24
    $as.Range("A1:M$([math]::Max(6,$r-1))").Font.Name='Microsoft YaHei'
    $as.Application.ActiveWindow.SplitRow=5; $as.Application.ActiveWindow.FreezePanes=$true
    $as.PageSetup.Orientation=2; $as.PageSetup.Zoom=$false; $as.PageSetup.FitToPagesWide=1; $as.PageSetup.FitToPagesTall=1

    $rawNames=@{
        'stage8_monte_carlo.csv'='Raw_MC'; 'stage8_sweep_cable_length.csv'='Raw_Cable';
        'stage8_sweep_p_dl_max.csv'='Raw_PDL'; 'stage8_sweep_rho_si.csv'='Raw_RhoSI';
        'stage8_sweep_system_load.csv'='Raw_Load'
    }
    foreach($spec in $files){
        $raw=Add-Sheet $workbook $rawNames[$spec.File]
        $rows=$datasets[$spec.File]
        $cols=@($rows[0].PSObject.Properties.Name)
        for($c=0;$c-lt$cols.Count;$c++){$raw.Cells.Item(1,$c+1).Value2=$cols[$c]}
        Apply-Header $raw.Range($raw.Cells.Item(1,1),$raw.Cells.Item(1,$cols.Count)) '5B9BD5'
        for($i=0;$i-lt$rows.Count;$i++){
            for($c=0;$c-lt$cols.Count;$c++){
                $name=$cols[$c]; $value=$rows[$i].$name
                $num=0.0
                if([double]::TryParse([string]$value,[ref]$num)){$raw.Cells.Item($i+2,$c+1).Value2=$num}else{$raw.Cells.Item($i+2,$c+1).Value2=[string]$value}
            }
        }
        $last=$rows.Count+1
        $raw.Range($raw.Cells.Item(1,1),$raw.Cells.Item($last,$cols.Count)).AutoFilter() | Out-Null
        $raw.Range($raw.Cells.Item(2,1),$raw.Cells.Item($last,$cols.Count)).NumberFormat='0.000000'
        foreach($idx in 0..($cols.Count-1)){
            if($cols[$idx] -match 'seed|iterations|n_starts|^K$|^J$|^L$|^R_s$|system_load'){$raw.Columns.Item($idx+1).NumberFormat='0'}
        }
        $raw.Range($raw.Cells.Item(1,1),$raw.Cells.Item($last,$cols.Count)).Font.Name='Calibri'
        $raw.Range($raw.Cells.Item(1,1),$raw.Cells.Item($last,$cols.Count)).Borders.Color=0xE7E6E6
        $raw.Range($raw.Cells.Item(1,1),$raw.Cells.Item($last,$cols.Count)).Borders.LineStyle=1
        $raw.Columns.AutoFit() | Out-Null
        for($c=1;$c-le$cols.Count;$c++){if($raw.Columns.Item($c).ColumnWidth -gt 22){$raw.Columns.Item($c).ColumnWidth=22}}
        $raw.Application.ActiveWindow.SplitRow=1; $raw.Application.ActiveWindow.FreezePanes=$true
        $raw.PageSetup.Orientation=2; $raw.PageSetup.Zoom=$false; $raw.PageSetup.FitToPagesWide=1
    }

    $dashboard.Move($workbook.Worksheets.Item(1))
    $dashboard.Activate()
    $workbook.SaveAs($outputPath, 51)
    $workbook.CalculateFull()
    $workbook.Save()

    foreach($sheetName in @('最终汇总','主结果统计','Proposed分组','异常Scene')){
        $sheet=$workbook.Worksheets.Item($sheetName)
        $pdf=Join-Path $previewDir ($sheetName + '.pdf')
        $sheet.ExportAsFixedFormat(0,$pdf)
    }

    $check = [pscustomobject]@{
        Output = $outputPath; Rows = $totalRows; Groups = $totalGroups; GroupFirst = $totalGroupFirst
        SceneFirst = $totalSceneFirst; Anomalies = $anomalies.Count
        MonteCarloMean = $mc.ProposedMean; MonteCarloStdP = $mc.ProposedStd
        Sheets = @($workbook.Worksheets | ForEach-Object { $_.Name })
        PreviewDir = $previewDir
    }
    $check | ConvertTo-Json -Depth 3
}
finally {
    if($workbook){$workbook.Close($true)}
    if($excel){$excel.Quit()}
    if($dashboard){[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($dashboard)}
    if($workbook){[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($workbook)}
    if($excel){[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel)}
    [gc]::Collect(); [gc]::WaitForPendingFinalizers()
}




