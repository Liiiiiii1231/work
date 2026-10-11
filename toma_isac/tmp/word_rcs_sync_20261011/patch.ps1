$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$base='D:/Vs codeProjects/research work1/toma_isac/tmp/word_rcs_sync_20261011'
$inputDoc='D:/Vs codeProjects/research work1/toma_isac/paper/V1.1/论文_V1.1.docx'
$ed=Get-Content -LiteralPath ($base+'/edits.json') -Raw -Encoding UTF8|ConvertFrom-Json
if((Get-FileHash -LiteralPath $inputDoc -Algorithm SHA256).Hash.ToLower() -ne $ed.originalHash){throw 'Input changed'}
$out=$base+'/论文_V1.1_已同步.docx'
Copy-Item -LiteralPath $inputDoc -Destination $out
$z=[IO.Compression.ZipFile]::Open($out,[IO.Compression.ZipArchiveMode]::Update)
function ReadPart($name){$r=[IO.StreamReader]::new($z.GetEntry($name).Open());try{return $r.ReadToEnd()}finally{$r.Dispose()}}
function WritePart($name,$text){$e=$z.GetEntry($name);if($e){$e.Delete()};$e=$z.CreateEntry($name);$w=[IO.StreamWriter]::new($e.Open(),[Text.UTF8Encoding]::new($false));try{$w.Write($text)}finally{$w.Dispose()}}
$x=[Xml.XmlDocument]::new();$x.PreserveWhitespace=$true;$x.LoadXml((ReadPart 'word/document.xml'))
$n=[Xml.XmlNamespaceManager]::new($x.NameTable);$wn='http://schemas.openxmlformats.org/wordprocessingml/2006/main';$n.AddNamespace('w',$wn)
$ps=@($x.SelectNodes('//w:body//w:p',$n))
function TextOf($p){return (($p.SelectNodes('.//w:t',$n)|ForEach-Object {$_.InnerText}) -join '')}
function ReplaceText($p,$old,$new){
 if((TextOf $p) -cne $old){throw 'Text mismatch'}
 $start=0;while($start -lt [Math]::Min($old.Length,$new.Length) -and $old[$start] -ceq $new[$start]){$start++}
 $tail=0;while($tail -lt [Math]::Min($old.Length,$new.Length)-$start -and $old[$old.Length-1-$tail] -ceq $new[$new.Length-1-$tail]){$tail++}
 $end=$old.Length-$tail;$replacement=$new.Substring($start,$new.Length-$start-$tail);$pos=0;$done=$false
 foreach($t in @($p.SelectNodes('.//w:t',$n))){$s=$t.InnerText;$next=$pos+$s.Length
  if($next -gt $start -and $pos -lt $end){$a=[Math]::Max(0,$start-$pos);$b=[Math]::Min($s.Length,$end-$pos);$insert='';if(-not $done){$insert=$replacement;$done=$true};$t.InnerText=$s.Substring(0,$a)+$insert+$s.Substring($b);$t.SetAttribute('xml:space','preserve')}
  $pos=$next
 }
 if((TextOf $p) -cne $new){throw 'Replacement validation failed'}
}
foreach($r in $ed.replacements){ReplaceText $ps[$r.index] $r.old $r.new}
$last=@{}
foreach($i in $ed.inserts){
 $p=$x.CreateElement('w','p',$wn);$template=$ps[$i.template]
 $prop=$template.SelectSingleNode('w:pPr',$n);if($prop){[void]$p.AppendChild($prop.CloneNode($true))}
 if($i.text -eq '参考文献'){$pp=$p.SelectSingleNode('w:pPr',$n);if(-not $pp){$pp=$x.CreateElement('w','pPr',$wn);[void]$p.PrependChild($pp)};$kn=$pp.SelectSingleNode('w:keepNext',$n);if(-not $kn){$kn=$x.CreateElement('w','keepNext',$wn);[void]$pp.PrependChild($kn)};$kn.SetAttribute('val',$wn,'1')}
 $r=$x.CreateElement('w','r',$wn);$rp=$template.SelectSingleNode('w:r/w:rPr',$n);if($rp){[void]$r.AppendChild($rp.CloneNode($true))}
 $t=$x.CreateElement('w','t',$wn);$t.SetAttribute('xml:space','preserve');$t.InnerText=$i.text;[void]$r.AppendChild($t);[void]$p.AppendChild($r)
 $key=[string]$i.after;$anchor=$ps[$i.after];if($last.ContainsKey($key)){$anchor=$last[$key]};[void]$anchor.ParentNode.InsertAfter($p,$anchor);$last[$key]=$p
}
$comments=[Xml.XmlDocument]::new();$cr=$comments.CreateElement('w','comments',$wn);[void]$comments.AppendChild($cr)
$notes=@(@{p=26;t='已同步为 RCS 等效散射幅度定义，并补充双程功率增益与归一化说明；原回波矩阵和传播相位不变。'},@{p=30;t='将已知或估计的目标参数改为有效 RCS，由其确定等效散射幅度。'},@{p=236;t='已补充统一车辆 RCS=10 m²、无额外随机散射相位、双端接收噪声均为 -114 dBm 的正式仿真假设。'})
$id=0
foreach($note in $notes){
 $p=$ps[$note.p];$st=$x.CreateElement('w','commentRangeStart',$wn);$st.SetAttribute('id',$wn,[string]$id);$pp=$p.SelectSingleNode('w:pPr',$n);if($pp){[void]$p.InsertAfter($st,$pp)}else{[void]$p.PrependChild($st)}
 $en=$x.CreateElement('w','commentRangeEnd',$wn);$en.SetAttribute('id',$wn,[string]$id);[void]$p.AppendChild($en)
 $r=$x.CreateElement('w','r',$wn);$ref=$x.CreateElement('w','commentReference',$wn);$ref.SetAttribute('id',$wn,[string]$id);[void]$r.AppendChild($ref);[void]$p.AppendChild($r)
 $c=$comments.CreateElement('w','comment',$wn);$c.SetAttribute('id',$wn,[string]$id);$c.SetAttribute('author',$wn,'Codex');$c.SetAttribute('date',$wn,'2026-10-11T00:00:00Z')
 $p2=$comments.CreateElement('w','p',$wn);$r2=$comments.CreateElement('w','r',$wn);$t2=$comments.CreateElement('w','t',$wn);$t2.InnerText=$note.t;[void]$r2.AppendChild($t2);[void]$p2.AppendChild($r2);[void]$c.AppendChild($p2);[void]$cr.AppendChild($c);$id++
}
$rel=[Xml.XmlDocument]::new();$rel.PreserveWhitespace=$true;$rel.LoadXml((ReadPart 'word/_rels/document.xml.rels'))
$r=$rel.CreateElement('Relationship',$rel.DocumentElement.NamespaceURI);$r.SetAttribute('Id','rIdRcsSyncComments');$r.SetAttribute('Type','http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments');$r.SetAttribute('Target','comments.xml');[void]$rel.DocumentElement.AppendChild($r)
$ct=[Xml.XmlDocument]::new();$ct.PreserveWhitespace=$true;$ct.LoadXml((ReadPart '[Content_Types].xml'))
$o=$ct.CreateElement('Override',$ct.DocumentElement.NamespaceURI);$o.SetAttribute('PartName','/word/comments.xml');$o.SetAttribute('ContentType','application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml');[void]$ct.DocumentElement.AppendChild($o)
WritePart 'word/document.xml' $x.OuterXml
WritePart 'word/comments.xml' $comments.OuterXml
WritePart 'word/_rels/document.xml.rels' $rel.OuterXml
WritePart '[Content_Types].xml' $ct.OuterXml
$z.Dispose()
Write-Output 'DOCX patched: 2 local text edits, 9 new paragraphs, 3 review comments'
