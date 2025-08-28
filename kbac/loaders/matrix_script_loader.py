import regex
from tqdm import tqdm
from more_itertools import flatten
from pypdf import PdfReader, PageObject
from typing import Literal, Optional
from pydantic import BaseModel, NonNegativeInt, PositiveInt, StrictStr, Field

class LineItem(BaseModel):
    page_number: NonNegativeInt
    text_type: Literal["location", "description", "character", "dialog", "raw"]
    text: StrictStr = Field(min_length=1)
    margin: PositiveInt

class Document(BaseModel):
    text: StrictStr
    metadata: dict

class MatrixScriptLoader:
    def __init__(
        self,
        source_path: str,
        ignoread_tags: list[str] = [
            "FADE IN:", "CONTINUED", "OMITTED", "THE MATRIX - Rev.", "FADE OUT.",
            "THE END", "(MORE)", "FADE TO BLACK.",
        ],
        location_margins: set[int] = {8, 9},
        description_margins: set[int] = {8, 9},
        dialog_margins: set[int] = {21, 30},
        character_margins: set[int] = {32, 38, 39},
        start_page: int | None = 1,
        end_page: int | None = None,
    ):
        self.source_path = source_path
        self.ignoread_tags = ignoread_tags
        self.location_margins = location_margins
        self.description_margins = description_margins
        self.dialog_margins = dialog_margins
        self.character_margins = character_margins
        self.start_page = start_page
        self.end_page = end_page

    def _get_location_text(self, line_text: str):
        if not line_text.isupper(): return None, None
        match_ = regex.match(r"([AB]?\d+\s{2,})(.*?)(\s{2,}[AB]?\d+$)", line_text)
        if match_ is None: return None, None
        groups = match_.groups()
        margin = len(groups[0])
        if margin not in self.location_margins: return None, None
        return groups[1].strip(), margin

    def _get_character_text(self, line_text: str):
        if not line_text.isupper(): return None, None
        match_ = regex.match(r"(\s{2,})(.*)", line_text)
        if match_ is None: return None, None
        groups = match_.groups()
        margin = len(groups[0])
        if margin not in self.character_margins: return None, None
        character_text = regex.sub(r"\(.*\)", "", groups[1]).strip()
        return character_text, margin

    def _get_non_upper_text(self, line_text: str, margins: set[int]):
        match_ = regex.match(r"(\s{2,})(.*)", line_text)
        if match_ is None: return None, None
        groups = match_.groups()
        margin = len(groups[0])
        if margin not in margins: return None, None
        return groups[1].strip(), margin

    def _get_description_text(self, line_text: str):
        return self._get_non_upper_text(line_text, margins=self.description_margins)

    def _get_dialog_text(self, line_text: str):
        return self._get_non_upper_text(line_text, margins=self.dialog_margins)

    def _parse_page_line(self, line_text: str, page_number: int) -> Optional[LineItem]:
        if any(it in line_text for it in self.ignoread_tags): return None
        
        # El orden de verificación es importante para evitar falsos positivos
        location_text, margin = self._get_location_text(line_text)
        if location_text: return LineItem(page_number=page_number, text_type="location", text=location_text, margin=margin)
        
        character_text, margin = self._get_character_text(line_text)
        if character_text: return LineItem(page_number=page_number, text_type="character", text=character_text, margin=margin)
        
        dialog_text, margin = self._get_dialog_text(line_text)
        if dialog_text: return LineItem(page_number=page_number, text_type="dialog", text=dialog_text, margin=margin)
        
        description_text, margin = self._get_description_text(line_text)
        if description_text: return LineItem(page_number=page_number, text_type="description", text=description_text, margin=margin)
        
        no_matched_text = line_text.strip()
        if len(no_matched_text):
            # Asumimos que el texto no clasificado es parte de una descripción
            return LineItem(page_number=page_number, text_type="description", text=no_matched_text, margin=1) 
        return None

    def parse_page(self, page: PageObject) -> list[LineItem]:
        if self.start_page and page.page_number < self.start_page: return []
        if self.end_page and page.page_number > self.end_page: return []
        
        page_text = page.extract_text(extraction_mode="layout")
        line_items = (self._parse_page_line(line_text, page.page_number) for line_text in page_text.split("\n"))
        return [li for li in line_items if li is not None]

    def load(self) -> list[Document]:
        """
        Parsea el PDF y devuelve una lista de líneas individuales con metadatos.
        La agrupación se delega a un servicio de nivel superior.
        """
        reader = PdfReader(self.source_path)
        parsed_pages = map(self.parse_page, tqdm(reader.pages, total=len(reader.pages), desc="Parsing script lines with Regex"))
        line_items = list(flatten(parsed_pages))
        
        documents = []
        for li in line_items:
            # Simplificamos: cada línea es un documento con su propio metadato.
            metadata = {
                "text_type": li.text_type,
                "page_number": li.page_number
            }
            documents.append(Document(text=li.text, metadata=metadata))
            
        return documents