import os
from pypdf import PdfReader

data_path = os.path.join('..', 'data')
sample_pdf_path = os.path.join('data', 'hdfc-life-click-2-invest-v01-policy-bond.pdf')
sample_pdf2_path = os.path.join('data', 'hdfc-life-click-2-wealth-v03-appendix-7-policy-bond.pdf')

print(data_path)
print(sample_pdf_path)

with open(sample_pdf2_path, "rb") as file:
    # Create a PDF reader object
    reader = PdfReader(file)

    # Print the total number of pages
    print(f"Number of pages: {len(reader.pages)}")

    # Extract first 500 words and print
    fivehundred_wds = ""
    for page_num, page in enumerate(reader.pages):
        fivehundred_wds += page.extract_text()[:500]
        if len(fivehundred_wds) >= 500: break

    print(fivehundred_wds)
