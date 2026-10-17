$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$source = 'D:/Vs codeProjects/research work1/toma_isac/paper/V1.1/论文_V1.1_传统方案说明已更新.docx'
$work = 'D:/Vs codeProjects/research work1/toma_isac/tmp/sim_analysis_revision_20261011'
$output = $work + '/论文_V1.1_已同步.docx'
New-Item -ItemType Directory -Path $work -Force | Out-Null
Copy-Item -LiteralPath $source -Destination $output -Force

$zip = [IO.Compression.ZipFile]::Open($output, [IO.Compression.ZipArchiveMode]::Update)
function Read-Part([string]$name) {
    $reader = [IO.StreamReader]::new($zip.GetEntry($name).Open(), [Text.Encoding]::UTF8)
    try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
}
function Write-Part([string]$name, [string]$content) {
    $entry = $zip.GetEntry($name)
    if ($entry) { $entry.Delete() }
    $entry = $zip.CreateEntry($name)
    $writer = [IO.StreamWriter]::new($entry.Open(), [Text.UTF8Encoding]::new($false))
    try { $writer.Write($content) } finally { $writer.Dispose() }
}

$w = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
$doc = [Xml.XmlDocument]::new()
$doc.PreserveWhitespace = $true
$doc.LoadXml((Read-Part 'word/document.xml'))
$ns = [Xml.XmlNamespaceManager]::new($doc.NameTable)
$ns.AddNamespace('w', $w)
$original = @($doc.SelectNodes('//w:body//w:p', $ns))

function Paragraph-Text($p) {
    return (($p.SelectNodes('.//w:t', $ns) | ForEach-Object { $_.InnerText }) -join '')
}
function Set-Paragraph-Text($p, [string]$text, $template) {
    $pPr = $p.SelectSingleNode('w:pPr', $ns)
    $runPr = $template.SelectSingleNode('w:r/w:rPr', $ns)
    foreach ($child in @($p.ChildNodes)) {
        if ($child -ne $pPr) { [void]$p.RemoveChild($child) }
    }
    $run = $doc.CreateElement('w', 'r', $w)
    if ($runPr) { [void]$run.AppendChild($runPr.CloneNode($true)) }
    $t = $doc.CreateElement('w', 't', $w)
    $t.SetAttribute('xml:space', 'preserve')
    $t.InnerText = $text
    [void]$run.AppendChild($t)
    [void]$p.AppendChild($run)
}
function New-Like-Paragraph($template, [string]$text) {
    $p = $doc.CreateElement('w', 'p', $w)
    $pPr = $template.SelectSingleNode('w:pPr', $ns)
    if ($pPr) { [void]$p.AppendChild($pPr.CloneNode($true)) }
    Set-Paragraph-Text $p $text $template
    return $p
}
function Replace-Span([int]$start, [int]$end, [string[]]$texts) {
    $anchor = $original[$start]
    $template = $original[$start]
    Set-Paragraph-Text $anchor $texts[0] $template
    $last = $anchor
    for ($i = 1; $i -lt $texts.Count; $i++) {
        $newP = New-Like-Paragraph $template $texts[$i]
        [void]$last.ParentNode.InsertAfter($newP, $last)
        $last = $newP
    }
    for ($i = $start + 1; $i -le $end; $i++) {
        [void]$original[$i].ParentNode.RemoveChild($original[$i])
    }
}
function Insert-After-Original([int]$index, [string]$text) {
    $template = $original[$index]
    $p = New-Like-Paragraph $template $text
    [void]$template.ParentNode.InsertAfter($p, $template)
    return $p
}

$statistics = '图1的收敛曲线来自场景种子 8001 对应的单个代表性随机场景。图2～图4中的每个数据点均对场景种子 8001～8010 对应的 10 个场景取平均。在每个场景和参数设置下，所有对比方案均采用相同的用户和目标配置。最大下行发射功率与残余 SI 扫描在各扫描点复用同一组场景种子；系统负载实验在每个 \(K=J=L\) 取值下同样采用场景种子 8001～8010，但场景维度随系统负载改变。'
$statisticsP = Insert-After-Original 243 $statistics

$fig1 = @(
    'Fig.1 展示了所提出的 FP-aware RCG-ToMA + FP 算法在一个代表性随机场景中的收敛过程。WSR 在最初几次联合 AO 迭代中快速增加，从约 \(0.55\) bit/s/Hz 在第一次迭代后提升至 \(6.1\) bit/s/Hz 以上。这一显著提升表明，通信–感知资源与 ToMA 几何的联合调整能够有效改善初始系统配置。',
    '随着迭代继续进行，WSR 仍然提升，但增幅逐渐减小。局部放大的后期曲线表明，算法在后续迭代中仍能获得小幅性能增益，同时相邻迭代之间的 WSR 变化逐渐缩小。算法在第 26 次联合 AO 迭代时终止，最终 WSR 约为 \(6.460\) bit/s/Hz。',
    '上述结果说明，在当前仿真设置下，所提出的联合优化方法具有数值收敛性。但是，由于原优化问题是非凸的，观察到的收敛结果不能保证所得解为全局最优解。'
)
Replace-Span 247 247 $fig1

$fig2 = @(
    'Fig.2 给出了不同方案的平均 WSR 随最大下行发射功率 \(P_{\rm D}^{\max}\) 的变化。当 \(P_{\rm D}^{\max}\) 从 \(20\) dBm 增加到 \(40\) dBm 时，采用 FP 资源优化的方案总体表现为 WSR 提升或近似稳定。其中，FPA-UPA + FP 的提升较为明显，而所提方法、MA + FP 和 Fixed-ToMA + FP 的增幅相对较小。',
    '相比之下，RCG-ToMA + Traditional 和 Fixed-ToMA + Traditional 的 WSR 呈下降趋势，而 FPA-UPA + Traditional 基本保持不变。这说明当前实现的传统波束成形与资源分配策略不一定能够有效利用增加的发射功率预算。在所考虑的全双工 ISAC 系统中，额外的发射功率还可能影响上行通信和感知接收端所经历的干扰。因此，图中传统基准方案的表现不应被解释为相应功率约束下的全局最优 WSR。',
    '在整个测试的发射功率范围内，所提出的 FP-aware RCG-ToMA + FP 方法始终获得最高的平均 WSR，MA + FP 位居第二。这表明，当 ToMA 缆绳上的天线阵元耦合移动与通信–感知资源联合优化相结合时，仍能提供有效的空间重构能力。',
    '此外，所提方法优于 Fixed-ToMA + FP，体现了优化 ToMA 几何所带来的性能收益。Fixed-ToMA + FP 和 FPA-UPA + FP 也分别优于对应的传统资源方案。由于每组比较采用相同的天线架构，这些结果支持 FP 通信–感知资源分配的有效性。'
)
Replace-Span 249 250 $fig2

$fig3 = @(
    'Fig.3 给出了不同残余自干扰水平下各方案的平均 WSR。为了清楚展示系统对残余 SI 的敏感性，该实验将 \(10\log_{10}(\rho_{\rm SI})\) 从 \(-40\) dB 扫描到 \(0\) dB；该范围仅用于残余 SI 敏感性实验，正式标称残余 SI 仍为 \(-100\) dB。随着残余 SI 从 \(-40\) dB 增加到 \(0\) dB，大多数方案的平均 WSR 总体下降或逐渐趋于相对稳定。这与主无人机接收端经历的残余干扰增强相一致。根据所采用的信号模型，残余 SI 信道同时进入上行通信和感知链路的干扰协方差矩阵，从而影响两类链路可达到的 SINR。',
    '所提出的 FP-aware RCG-ToMA + FP 方法在全部五个残余 SI 测试点均获得最高的平均 WSR，MA + FP 位居第二。尽管所提方法的 WSR 随残余 SI 增强总体下降，但在最强的测试 SI 水平处出现轻微回升。这一非单调变化应理解为数值优化中观察到的结果，不能视为残余 SI 增强能够从物理上改善系统性能的证据。',
    '与各基准方案相比，所提方法在整个残余 SI 测试范围内保持了明确的性能优势。这些结果表明，在不同残余 SI 条件下，联合调整 ToMA 几何和通信–感知资源是有效的。此外，采用 FP 资源优化的方案总体优于相应的传统资源基准，说明资源优化对于缓解干扰相关的性能下降具有重要作用。'
)
Replace-Span 252 253 $fig3

$fig4note = '需要强调的是，Fig.4 的横轴表示 \(K=J=L\)，而不是 \(K=J=L=R_{\rm S}\)。该实验采用归一化的通信和感知权重，每个活跃下行用户、上行用户和感知目标的权重均为 \(1/(K+J+L)\)。因此，随着系统负载增加而出现的 WSR 下降反映的是归一化加权性能的变化，不能直接解释为所有通信用户和感知目标的未加权总速率下降。'
$fig4P = Insert-After-Original 256 $fig4note

$comments = [Xml.XmlDocument]::new()
$comments.PreserveWhitespace = $true
$comments.LoadXml((Read-Part 'word/comments.xml'))
$existingIds = @($comments.SelectNodes('//w:comment', $ns) | ForEach-Object { [int]$_.GetAttribute('id', $w) })
$nextId = if ($existingIds.Count) { ($existingIds | Measure-Object -Maximum).Maximum + 1 } else { 0 }
function Add-Comment($p, [string]$message) {
    $id = $script:nextId
    $script:nextId++
    $start = $doc.CreateElement('w', 'commentRangeStart', $w)
    $start.SetAttribute('id', $w, [string]$id)
    $pPr = $p.SelectSingleNode('w:pPr', $ns)
    if ($pPr) { [void]$p.InsertAfter($start, $pPr) } else { [void]$p.PrependChild($start) }
    $end = $doc.CreateElement('w', 'commentRangeEnd', $w)
    $end.SetAttribute('id', $w, [string]$id)
    [void]$p.AppendChild($end)
    $run = $doc.CreateElement('w', 'r', $w)
    $ref = $doc.CreateElement('w', 'commentReference', $w)
    $ref.SetAttribute('id', $w, [string]$id)
    [void]$run.AppendChild($ref)
    [void]$p.AppendChild($run)
    $comment = $comments.CreateElement('w', 'comment', $w)
    $comment.SetAttribute('id', $w, [string]$id)
    $comment.SetAttribute('author', $w, 'Codex')
    $comment.SetAttribute('date', $w, '2026-10-11T00:00:00Z')
    $cp = $comments.CreateElement('w', 'p', $w)
    $cr = $comments.CreateElement('w', 'r', $w)
    $ct = $comments.CreateElement('w', 't', $w)
    $ct.InnerText = $message
    [void]$cr.AppendChild($ct)
    [void]$cp.AppendChild($cr)
    [void]$comment.AppendChild($cp)
    [void]$comments.DocumentElement.AppendChild($comment)
}
Add-Comment $statisticsP '依据 Stage 8 配置、运行程序和正式 CSV，补充单场景、十场景平均、场景种子复用及同场景公平比较说明。'
$current = @($doc.SelectNodes('//w:body//w:p', $ns))
Add-Comment ($current | Where-Object { (Paragraph-Text $_).StartsWith('Fig.1 展示了') } | Select-Object -First 1) '按最新收敛图修正为第 26 次迭代终止、最终 WSR 约 6.460，并删除缺乏图像证据的第 13 次迭代归因。'
Add-Comment ($current | Where-Object { (Paragraph-Text $_).StartsWith('Fig.2 给出了') } | Select-Object -First 1) '按最新功率扫描图保留传统方案下降或稳定的真实趋势，并避免解释为理论最优 WSR 随功率下降。'
Add-Comment ($current | Where-Object { (Paragraph-Text $_).StartsWith('Fig.3 给出了') } | Select-Object -First 1) '按最新残余 SI 图明确所提方法在全部测试点均高于 MA + FP，并谨慎说明最强 SI 点的轻微回升。'
Add-Comment $fig4P '已由 Stage 8 各负载 YAML 验证权重均为 1/(K+J+L)，据此补充归一化 WSR 的解释边界。'

Write-Part 'word/document.xml' $doc.OuterXml
Write-Part 'word/comments.xml' $comments.OuterXml
$zip.Dispose()
Write-Output $output
