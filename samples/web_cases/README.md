# Web Test Cases

These files were downloaded from public sample/document sources for parser,
indexing, OCR, and retrieval smoke tests.

| File | Type | Suggested use | Source |
| --- | --- | --- | --- |
| `docx-formatted-samplelib.docx` | DOCX | Basic Word parsing, headings, formatting | https://samplelib.com/docx/sample-simple.docx |
| `docx-long-samplelib.docx` | DOCX | Larger Word parsing and chunking | https://samplelib.com/docx/sample-long.docx |
| `pdf-plain-1page-samplelib.pdf` | PDF | Small PDF smoke test | https://samplelib.com/pdf/sample1.pdf |
| `pdf-images-samplelib.pdf` | PDF | PDF with richer visual/layout content | https://samplelib.com/pdf/sample-heavy.pdf |
| `pdf-tables-samplelib.pdf` | PDF | Table-oriented PDF parsing | https://samplelib.com/pdf/sample-table.pdf |
| `pdf-scanned-samplelib.pdf` | PDF | Scanned/OCR path | https://samplelib.com/pdf/sample-scanned.pdf |
| `pdf-azure-layout.pdf` | PDF | Layout extraction sample | https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf |
| `pdf-w3c-dummy.pdf` | PDF | Tiny baseline PDF | https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf |
| `pdf-unec-report.pdf` | PDF | Longer report-style PDF | https://unece.org/fileadmin/DAM/cefact/cf_plenary/2017_Plenary/ECE_TRADE_C_CEFACT_2017_18E.pdf |
| `png-clouds-samplelib.png` | PNG | Image parsing / OCR fallback behavior | https://samplelib.com/png/sample-clouds2-400x300.png |
| `jpg-clouds-samplelib.jpg` | JPG | Image parser smoke test | https://samplelib.com/jpeg/sample-clouds-400x300.jpg |
| `xlsx-frictionless-sample-1-sheet.xlsx` | XLSX | Small spreadsheet parsing | https://raw.githubusercontent.com/frictionlessdata/datasets/main/files/excel/sample-1-sheet.xlsx |
| `xlsx-ou-sample.xlsx` | XLSX | Spreadsheet parsing with more rows/cells | https://www.ou.edu/content/dam/cms/docs/sample-excel-file.xlsx |
| `pptx-suu-example.pptx` | PPTX | Basic presentation parsing | https://www.suu.edu/webservices/styleguide/example-files/example.pptx |
| `pptx-documentfoundation-extlst.pptx` | PPTX | Office compatibility edge case | https://wiki.documentfoundation.org/images/4/47/Extlst-test.pptx |

Quick parse command:

```powershell
.\.venv\Scripts\python.exe -m rag_system.parsing --backend auto --document samples\web_cases
```

Or open the local web UI and drag files from this folder:

```powershell
.\.venv\Scripts\python.exe -m rag_system.web
```
