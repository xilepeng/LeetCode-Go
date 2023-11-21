## markdown文件导出为html、doc、epub、pdf格式

1. 安装pandoc `brew install pandoc`
2. 将markdown转成html: `pandoc -f markdown -t html ./test.md`
3. 将markdown转成doc：`pandoc -f markdown -t html ./test.md | pandoc -f html -t docx -o test.docx`
4. 将markdown转成PDF，需要安装latex。只要安装basicTex就可以了，大概100M+,安装完后运行：`pandoc -f markdown_github test.md -o test.pdf --latex-engine=xelatex -V mainfont="SimSun"` 这个表明使用的是GitHub风格markdown语法
5. 将markdown转成epub： `pandoc -f markdown ./test.md -o test.epub`