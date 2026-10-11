$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$base='D:/Vs codeProjects/research work1/toma_isac/tmp/word_rcs_sync_20261011'
$ed=Get-Content -LiteralPath ($base+'/edits.json') -Raw -Encoding UTF8|ConvertFrom-Json
function Parts($path){$z=[IO.Compression.ZipFile]::OpenRead($path);$h=@{};try{foreach($e in $z.Entries){$m=[IO.MemoryStream]::new();$s=$e.Open();$s.CopyTo($m);$s.Dispose();$h[$e.FullName]=$m.ToArray();$m.Dispose()}}finally{$z.Dispose()};return $h}
$a=Parts ($base+'/论文_V1.1_修改前.docx');$b=Parts ($base+'/论文_V1.1_已同步.docx')
$changed=@();foreach($key in $b.Keys){if(-not $a.ContainsKey($key) -or [Convert]::ToBase64String($a[$key]) -cne [Convert]::ToBase64String($b[$key])){$changed+=$key};if($key -match '\.(xml|rels)$'){[xml]$test=[Text.Encoding]::UTF8.GetString($b[$key])}}
if(($changed|Where-Object {$_ -notin @('word/document.xml','word/comments.xml','word/_rels/document.xml.rels','[Content_Types].xml')}).Count){throw 'Unexpected package changes'}
[xml]$x=[Text.Encoding]::UTF8.GetString($a['word/document.xml']);[xml]$y=[Text.Encoding]::UTF8.GetString($b['word/document.xml'])
$n=[Xml.XmlNamespaceManager]::new($x.NameTable);$n.AddNamespace('w','http://schemas.openxmlformats.org/wordprocessingml/2006/main')
$op=@($x.SelectNodes('//w:body//w:p',$n));$np=@($y.SelectNodes('//w:body//w:p',$n))
function PT($p){return (($p.SelectNodes('.//w:t',$n)|ForEach-Object {$_.InnerText}) -join '')}
$j=0;$intact=0
for($i=0;$i -lt $op.Count;$i++){$r=@($ed.replacements|Where-Object {$_.index -eq $i});$expected=PT $op[$i];if($r.Count){$expected=$r[0].new};if((PT $np[$j]) -cne $expected){throw ('Paragraph mismatch '+$i)}
if($i -notin @(26,30,236)){if($op[$i].OuterXml -cne $np[$j].OuterXml){throw ('Formatting changed '+$i)};$intact++};$j++
foreach($ins in @($ed.inserts|Where-Object {$_.after -eq $i})){if((PT $np[$j]) -cne $ins.text){throw 'Insertion mismatch'};$j++}}
if($j -ne $np.Count -or $np.Count -ne 259){throw 'Count mismatch'}
if($y.SelectNodes('//w:commentRangeStart',$n).Count -ne 3){throw 'Comments mismatch'}
@{originalParagraphs=$op.Count;finalParagraphs=$np.Count;unchangedOriginalParagraphsWithFormatting=$intact;modifiedOriginalTextParagraphs=2;addedParagraphs=9;comments=3;changedPackageParts=$changed;xmlValid=$true}|ConvertTo-Json
