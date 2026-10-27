You are reading pages of a document. Return one JSON object that matches the JSON Schema below, filled from what is printed on these pages.

- Copy values exactly as printed: names, codes and descriptions verbatim; numbers as numbers, without currency symbols or thousands separators; dates as the schema's format asks, else as printed.
- Use null for a field these pages do not show. Never guess or invent a value.
- For every array, include one item per record printed on these pages, in page order: none skipped, merged or repeated.

Return only the JSON object, without a code fence.
