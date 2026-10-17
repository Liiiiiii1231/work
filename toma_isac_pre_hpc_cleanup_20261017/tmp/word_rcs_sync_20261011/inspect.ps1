Add-Type -AssemblyName System.IO.Compression.FileSystem
$z=[IO.Compression.ZipFile]::OpenRead('D:\Vs codeProjects\research work1\论文_V1.1.docx')
$r=[IO.StreamReader]::new($z.GetEntry('word/document.xml').Open())
$s=$r.ReadToEnd();$r.Dispose();$z.Dispose()
[IO.File]::WriteAllText('D:\Vs codeProjects\research work1\toma_isac\tmp\word_rcs_sync_20261011\document_original.xml',$s)
[xml]$x=$s
$n=[Xml.XmlNamespaceManager]::new($x.NameTable)
$n.AddNamespace('w','http://schemas.openxmlformats.org/wordprocessingml/2006/main')
$n.AddNamespace('m','http://schemas.openxmlformats.org/officeDocument/2006/math')
$ps=$x.SelectNodes('//w:body//w:p',$n)
$a=@();for($i=0;$i -lt $ps.Count;$i++){ $txt=($ps[$i].SelectNodes('.//w:t|.//m:t',$n)|ForEach-Object {$_.InnerText}) -join ''; $a+=@{index=$i;text=$txt} }
$a|ConvertTo-Json -Depth 5|Set-Content -Encoding UTF8 -LiteralPath 'D:\Vs codeProjects\research work1\toma_isac\tmp\word_rcs_sync_20261011\paragraphs.json'
Write-Output $ps.Count