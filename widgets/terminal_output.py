"""Terminal output widget for displaying command execution."""
import re
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtCore import Qt


class TerminalOutputWidget(QWidget):
    """Custom widget for displaying terminal-style command output with ANSI color support."""

    # ANSI color codes to CSS colors
    ANSI_COLORS = {
        '30': '#000000',  # Black
        '31': '#ff5555',  # Red
        '32': '#50fa7b',  # Green
        '33': '#f1fa8c',  # Yellow
        '34': '#6272a4',  # Blue
        '35': '#ff79c6',  # Magenta
        '36': '#8be9fd',  # Cyan
        '37': '#d4d4d4',  # White
        '90': '#6272a4',  # Bright Black (Gray)
        '91': '#ff6e6e',  # Bright Red
        '92': '#69ff94',  # Bright Green
        '93': '#ffffa5',  # Bright Yellow
        '94': '#d6acff',  # Bright Blue
        '95': '#ff92df',  # Bright Magenta
        '96': '#a4ffff',  # Bright Cyan
        '97': '#ffffff',  # Bright White
    }

    # ANSI escape sequence pattern
    ANSI_ESCAPE = re.compile(r'\x1b\[([0-9;]*)([a-zA-Z])')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.auto_scroll = True
        self.text_edit = QTextEdit()
        self.current_color = None
        self.is_bold = False
        self.setup_ui()

    def setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Configure text edit
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        # Apply monospace font
        font = QFont("Monospace", 10)
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        self.text_edit.setFont(font)

        # Apply dark theme
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
        """)

        layout.addWidget(self.text_edit)

    def parse_ansi(self, text: str) -> str:
        """Convert ANSI escape codes to HTML."""
        result = []
        last_end = 0
        open_span = False

        for match in self.ANSI_ESCAPE.finditer(text):
            # Add text before this escape sequence
            before_text = text[last_end:match.start()]
            if before_text:
                result.append(self._escape_html(before_text))

            params = match.group(1)
            command = match.group(2)

            # Only handle SGR (Select Graphic Rendition) commands - 'm'
            if command == 'm':
                if open_span:
                    result.append('</span>')
                    open_span = False

                codes = params.split(';') if params else ['0']

                for code in codes:
                    if code == '0' or code == '':
                        # Reset
                        self.current_color = None
                        self.is_bold = False
                    elif code == '1':
                        # Bold
                        self.is_bold = True
                    elif code in self.ANSI_COLORS:
                        self.current_color = self.ANSI_COLORS[code]

                # Open new span if we have styling
                if self.current_color or self.is_bold:
                    styles = []
                    if self.current_color:
                        styles.append(f'color:{self.current_color}')
                    if self.is_bold:
                        styles.append('font-weight:bold')
                    result.append(f'<span style="{";".join(styles)}">')
                    open_span = True

            # Skip other escape sequences (cursor movement, clear, etc.)
            last_end = match.end()

        # Add remaining text
        remaining = text[last_end:]
        if remaining:
            result.append(self._escape_html(remaining))

        if open_span:
            result.append('</span>')

        return ''.join(result)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace(' ', '&nbsp;')
                .replace('\n', '<br>'))

    def append_command(self, command: str):
        """Append command with $ prefix."""
        html = f'<span style="color:#50fa7b">$</span> {self._escape_html(command)}<br>'
        self._append_html(html)

    def append_output(self, text: str):
        """Append stdout/stderr output with ANSI color support."""
        # Reset color state for each output block
        self.current_color = None
        self.is_bold = False

        html = self.parse_ansi(text)
        self._append_html(html)

    def append_error(self, text: str):
        """Append error text with red color."""
        html = f'<span style="color:#ff5555">{self._escape_html(text)}</span><br>'
        self._append_html(html)

    def _append_html(self, html: str):
        """Append HTML to the text edit."""
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html)
        self.text_edit.setTextCursor(cursor)

        if self.auto_scroll:
            self.scroll_to_bottom()

    def clear(self):
        """Clear all output."""
        self.text_edit.clear()
        self.current_color = None
        self.is_bold = False

    def copy_to_clipboard(self):
        """Copy all text to clipboard."""
        from PyQt6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())

    def scroll_to_bottom(self):
        """Auto-scroll to latest output."""
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_auto_scroll(self, enabled: bool):
        """Enable or disable auto-scroll."""
        self.auto_scroll = enabled
