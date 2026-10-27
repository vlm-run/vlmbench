Convert this document page to Markdown, faithfully and completely, in natural reading order.

- Keep every piece of text printed on the page exactly as written, in its original language, script and accents: headings, body text, lists, captions, footnotes and page numbers.
- Wrap the running page header in `<page_header>...</page_header>` and the running page footer, page number included, in `<page_footer>...</page_footer>`.
- Headings with `#` to `######` by level, lists with `-` or `1.`, and **bold**, *italic*, <u>underline</u>, ~~strikethrough~~, <sup>superscript</sup> and <sub>subscript</sub> where the page uses them.
- Tables as HTML `<table>`: one `<tr>` per line, `<th>` for header cells, `colspan` / `rowspan` where cells merge. A printed cell stays one cell: join its wrapped lines with `<br>`, never split them into rows. Inside HTML write the text literally, with no Markdown escapes.
- Math as LaTeX: inline between `$` and `$`, displayed between `$$` and `$$`. Code as a fenced block naming its language.
- Form checkboxes as ☐ (empty) or ☑ (ticked).
- A chart: its title, then its data as a table with one row per category (axis label or slice) and one column per series (legend entry), every data point included, each value as printed on the chart or else read from its axis. Don't describe the chart in prose.
- Any other picture: only the text printed inside it; don't describe it.

Return only the Markdown, without a code fence.
