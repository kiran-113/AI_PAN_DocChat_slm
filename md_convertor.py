import os, time, re
from marker.output import save_output

class Convert2Markdown:
    """
    Convert2Markdown: Converts PDF and HTML files to Markdown.
    """
    def _markdown_remove_images(self, markdown: str) -> str:
        """Remove image references from Markdown text."""
        return re.sub(r'!\[.*?\]\(.*?\)', '', markdown)

    def _remove_images_from_directory(self, directory: str, extension: str = ".jpeg") -> None:
        """Delete all images with a specific extension in a directory."""
        for file in os.listdir(directory):
            if file.endswith(extension):
                os.remove(os.path.join(directory, file))

    def _load_file(self, file_path: str) -> str:
        """Load and return the content of a file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _save_file(self, file_path: str, content: str) -> None:
        """Save content to a file."""
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
            
    def _format_time(self, response_time):
        hours = response_time // 3600
        minutes = (response_time % 3600) // 60
        seconds = response_time % 60

        if hours:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"Time: {int(seconds)}s"

    def pdf_to_markdown(self, marker_converter, input_pdf: str, output_directory: str, remove_images: bool = True) -> None:
        """Convert a PDF file to Markdown using the Marker tool."""
        
        if not marker_converter:
            raise ValueError("marker_converter instance is required.")
        if not input_pdf or not output_directory:
            raise ValueError("Both input PDF path and output directory are required.")
        if not os.path.exists(input_pdf):
            raise FileNotFoundError(f"Input PDF '{input_pdf}' does not exist.")
        
        os.makedirs(output_directory, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(input_pdf))[0]
        markdown_path = os.path.join(output_directory, f"{base_filename}.md")
        
        start_time = time.time()
        
        try:
            rendered = marker_converter(input_pdf)
            save_output(rendered, output_directory, f"{base_filename}")
            
            # if remove_images and os.path.exists(markdown_path):
            if remove_images :
                markdown_content = self._load_file(markdown_path)
                markdown_content = self._markdown_remove_images(markdown_content)
                self._save_file(markdown_path, markdown_content)
                self._remove_images_from_directory(output_directory, extension=".jpeg")
                
            print(f"Markdown saved: '{markdown_path}', Conversion completed in {self._format_time(time.time() - start_time)} seconds")
        except Exception as e:
            print(f"Error during PDF conversion: {e}")
            raise
