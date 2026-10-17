$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
Copy-Item -LiteralPath 'D:/Vs codeProjects/research work1/toma_isac/tmp/traditional_revision_20261011/before.docx' -Destination 'D:/Vs codeProjects/research work1/toma_isac/tmp/traditional_revision_20261011/updated.docx'
$z=[IO.Compression.ZipFile]::Open('D:/Vs codeProjects/research work1/toma_isac/tmp/traditional_revision_20261011/updated.docx','Update')
function ReadPart($name){$r=[IO.StreamReader]::new($z.GetEntry($name).Open());try{return $r.ReadToEnd()}finally{$r.Dispose()}}
function WritePart($name,$text){$z.GetEntry($name).Delete();$e=$z.CreateEntry($name);$w=[IO.StreamWriter]::new($e.Open(),[Text.UTF8Encoding]::new($false));$w.Write($text);$w.Dispose()}
$x=[Xml.XmlDocument]::new();$x.PreserveWhitespace=$true;$x.LoadXml((ReadPart 'word/document.xml'))
$n=[Xml.XmlNamespaceManager]::new($x.NameTable);$wn='http://schemas.openxmlformats.org/wordprocessingml/2006/main';$n.AddNamespace('w',$wn)
function PT($p){return (($p.SelectNodes('.//w:t',$n)|ForEach-Object {$_.InnerText}) -join '')}
$ps=@($x.SelectNodes('//w:body//w:p',$n));$old=@($ps|Where-Object {(PT $_).StartsWith('在进行多方案性能比较之前')});$anchor=@($ps|Where-Object {(PT $_) -eq '所提算法的收敛性评估'});if($old.Count -ne 1 -or $anchor.Count -ne 1){throw 'Paragraph match failed'}
$zh=Get-Content -LiteralPath 'D:/Vs codeProjects/research work1/toma_isac/tmp/traditional_revision_20261011/zh.json' -Raw -Encoding UTF8|ConvertFrom-Json
$newPs=@();foreach($s in $zh){$p=$x.CreateElement('w','p',$wn);$pp=$old[0].SelectSingleNode('w:pPr',$n);if($pp){[void]$p.AppendChild($pp.CloneNode($true))};$r=$x.CreateElement('w','r',$wn);$rp=$old[0].SelectSingleNode('w:r/w:rPr',$n);if($rp){[void]$r.AppendChild($rp.CloneNode($true))};$t=$x.CreateElement('w','t',$wn);$t.SetAttribute('xml:space','preserve');$t.InnerText=$s;[void]$r.AppendChild($t);[void]$p.AppendChild($r);[void]$anchor[0].ParentNode.InsertBefore($p,$anchor[0]);$newPs+=$p}
[void]$old[0].ParentNode.RemoveChild($old[0])
$c=[Xml.XmlDocument]::new();$c.PreserveWhitespace=$true;$c.LoadXml((ReadPart 'word/comments.xml'));$id=3
$p=$newPs[0];$st=$x.CreateElement('w','commentRangeStart',$wn);$st.SetAttribute('id',$wn,[string]$id);$pp=$p.SelectSingleNode('w:pPr',$n);if($pp){[void]$p.InsertAfter($st,$pp)}else{[void]$p.PrependChild($st)};$en=$x.CreateElement('w','commentRangeEnd',$wn);$en.SetAttribute('id',$wn,[string]$id);[void]$p.AppendChild($en);$r=$x.CreateElement('w','r',$wn);$ref=$x.CreateElement('w','commentReference',$wn);$ref.SetAttribute('id',$wn,[string]$id);[void]$r.AppendChild($ref);[void]$p.AppendChild($r)
$note=$c.CreateElement('w','comment',$wn);$note.SetAttribute('id',$wn,[string]$id);$note.SetAttribute('author',$wn,'Codex');$note.SetAttribute('date',$wn,'2026-10-11T00:00:00Z');$p2=$c.CreateElement('w','p',$wn);$r2=$c.CreateElement('w','r',$wn);$t2=$c.CreateElement('w','t',$wn);$t2.InnerText='将原 C 小节的对比方案设置集中到 A 小节，并按实际代码补充传统方案的接收合并与资源重建规则。';[void]$r2.AppendChild($t2);[void]$p2.AppendChild($r2);[void]$note.AppendChild($p2);[void]$c.DocumentElement.AppendChild($note)
WritePart 'word/document.xml' $x.OuterXml;WritePart 'word/comments.xml' $c.OuterXml;$z.Dispose()
Write-Output 'DOCX updated'
