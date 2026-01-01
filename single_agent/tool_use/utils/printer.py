"""
Colored terminal printer for framework-agnostic progress and result display.

This module provides a unified interface for printing progress, results, and
status messages with colors. Can be used by any framework implementation.
"""
import sys
from typing import Optional, Dict, Any, List
from enum import Enum


class Color:
    """ANSI color codes for terminal output."""
    # Reset
    RESET = '\033[0m'
    
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'


class Status(Enum):
    """Status types for messages."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    PROGRESS = "progress"


class Printer:
    """
    Framework-agnostic colored terminal printer.
    
    Provides methods for printing progress, results, and status messages
    with colors. Can be used by any framework implementation.
    """
    
    def __init__(self, use_colors: bool = True, verbose: bool = True):
        """
        Initialize printer.
        
        Args:
            use_colors: Whether to use colors (default: True)
            verbose: Whether to print verbose messages (default: True)
        """
        self.use_colors = use_colors and sys.stdout.isatty()
        self.verbose = verbose
    
    def _colorize(self, text: str, color: str, style: Optional[str] = None) -> str:
        """Apply color and style to text."""
        if not self.use_colors:
            return text
        
        parts = [color]
        if style:
            parts.append(style)
        parts.append(text)
        parts.append(Color.RESET)
        return ''.join(parts)
    
    def print_header(self, title: str, width: int = 80):
        """Print a section header."""
        line = "=" * width
        colored_title = self._colorize(title, Color.BOLD + Color.CYAN)
        colored_line = self._colorize(line, Color.CYAN)
        print(f"\n{colored_line}")
        print(colored_title)
        print(f"{colored_line}\n")
    
    def print_step(self, step_num: int, total_steps: int, message: str, 
                   sub_message: Optional[str] = None):
        """
        Print a step message.
        
        Args:
            step_num: Current step number (1-indexed)
            total_steps: Total number of steps
            message: Main step message
            sub_message: Optional sub-message (indented)
        """
        step_text = f"[{step_num}/{total_steps}]"
        colored_step = self._colorize(step_text, Color.BOLD + Color.BLUE)
        colored_message = self._colorize(message, Color.WHITE)
        
        print(f"\n{colored_step} {colored_message}")
        
        if sub_message:
            colored_sub = self._colorize(f"  {sub_message}", Color.DIM + Color.WHITE)
            print(colored_sub)
    
    def print_info(self, message: str, indent: int = 0):
        """Print an info message."""
        prefix = "  " * indent
        colored_message = self._colorize(f"{prefix}{message}", Color.CYAN)
        print(colored_message)
    
    def print_success(self, message: str, indent: int = 0):
        """Print a success message."""
        prefix = "  " * indent
        icon = self._colorize("✓", Color.BRIGHT_GREEN)
        colored_message = self._colorize(f"{prefix}{message}", Color.GREEN)
        print(f"{icon} {colored_message}")
    
    def print_error(self, message: str, indent: int = 0):
        """Print an error message."""
        prefix = "  " * indent
        icon = self._colorize("✗", Color.BRIGHT_RED)
        colored_message = self._colorize(f"{prefix}{message}", Color.RED)
        print(f"{icon} {colored_message}")
    
    def print_warning(self, message: str, indent: int = 0):
        """Print a warning message."""
        prefix = "  " * indent
        icon = self._colorize("⚠", Color.BRIGHT_YELLOW)
        colored_message = self._colorize(f"{prefix}{message}", Color.YELLOW)
        print(f"{icon} {colored_message}")
    
    def print_query_header(self, query_num: int, total_queries: int, 
                          query_id: str, query_text: str, 
                          num_gold_apis: int = 0):
        """Print query information header."""
        header = f"--- Query {query_num}/{total_queries}: {query_id} ---"
        colored_header = self._colorize(header, Color.BOLD + Color.MAGENTA)
        print(f"\n{colored_header}")
        
        # Truncate query text if too long
        display_text = query_text[:100] + "..." if len(query_text) > 100 else query_text
        colored_query = self._colorize(f"Query: {display_text}", Color.WHITE)
        print(colored_query)
        
        if num_gold_apis > 0:
            colored_apis = self._colorize(f"Gold APIs: {num_gold_apis}", Color.DIM + Color.WHITE)
            print(colored_apis)
    
    def print_progress(self, message: str, indent: int = 2):
        """Print a progress message."""
        if not self.verbose:
            return
        
        prefix = "  " * indent
        icon = self._colorize("→", Color.BLUE)
        colored_message = self._colorize(message, Color.WHITE)
        print(f"{prefix}{icon} {colored_message}")
    
    def print_result(self, status: str, sopr_score: float, api_score: float, 
                    total_time: float, answer_status: Optional[str] = None):
        """
        Print a query result.
        
        Args:
            status: Status icon ("✓", "?", "✗")
            sopr_score: SoPR score (0.0-1.0)
            api_score: API call score (0.0-1.0)
            total_time: Total time in seconds
            answer_status: Answer status string (e.g., "Solved", "Unsure")
        """
        # Determine status color
        if sopr_score == 1.0:
            status_color = Color.BRIGHT_GREEN
            status_icon = "✓"
        elif sopr_score == 0.5:
            status_color = Color.BRIGHT_YELLOW
            status_icon = "?"
        else:
            status_color = Color.BRIGHT_RED
            status_icon = "✗"
        
        colored_icon = self._colorize(status_icon, status_color)
        
        # Format answer status
        if answer_status:
            status_text = f"{answer_status:8s}"
        else:
            status_text = "        "
        
        # Format scores with colors
        sopr_color = Color.GREEN if sopr_score >= 0.5 else Color.RED
        api_color = Color.GREEN if api_score >= 0.5 else Color.RED
        
        colored_sopr = self._colorize(f"{sopr_score:.2f}", sopr_color)
        colored_api = self._colorize(f"{api_score:.2f}", api_color)
        colored_time = self._colorize(f"{total_time:.2f}s", Color.DIM + Color.WHITE)
        
        print(f"  {colored_icon} {status_text} | "
              f"SoPR: {colored_sopr} | "
              f"API: {colored_api} | "
              f"Time: {colored_time}")
    
    def print_summary(self, summary: Dict[str, Any], output_file: Optional[str] = None):
        """
        Print benchmark summary.
        
        Args:
            summary: Summary dictionary with statistics
            output_file: Optional output file path
        """
        self.print_header("Benchmark Results Summary")
        
        # Agent and test set
        agent_name = summary.get('agent_name', 'Unknown')
        test_set = summary.get('test_set', 'Unknown')
        self.print_info(f"Agent:                    {agent_name}")
        self.print_info(f"Test Set:                 {test_set}")
        
        # Query statistics
        total_queries = summary.get('total_queries', 0)
        solved_count = summary.get('solved_count', 0)
        unsure_count = summary.get('unsure_count', 0)
        unsolved_count = total_queries - solved_count - unsure_count
        finish_count = summary.get('finish_count', 0)
        
        solved_pct = (solved_count / total_queries * 100) if total_queries > 0 else 0.0
        unsure_pct = (unsure_count / total_queries * 100) if total_queries > 0 else 0.0
        unsolved_pct = (unsolved_count / total_queries * 100) if total_queries > 0 else 0.0
        finish_pct = (finish_count / total_queries * 100) if total_queries > 0 else 0.0
        
        self.print_info(f"Total Queries:            {total_queries}")
        
        # Solved (green)
        solved_text = f"Solved:                   {solved_count} ({solved_pct:.1f}%)"
        if solved_count > 0:
            print(f"  {self._colorize('✓', Color.BRIGHT_GREEN)} {self._colorize(solved_text, Color.GREEN)}")
        else:
            self.print_info(solved_text)
        
        # Unsure (yellow)
        unsure_text = f"Unsure:                   {unsure_count} ({unsure_pct:.1f}%)"
        if unsure_count > 0:
            print(f"  {self._colorize('?', Color.BRIGHT_YELLOW)} {self._colorize(unsure_text, Color.YELLOW)}")
        else:
            self.print_info(unsure_text)
        
        # Unsolved (red)
        unsolved_text = f"Unsolved:                 {unsolved_count} ({unsolved_pct:.1f}%)"
        if unsolved_count > 0:
            print(f"  {self._colorize('✗', Color.BRIGHT_RED)} {self._colorize(unsolved_text, Color.RED)}")
        else:
            self.print_info(unsolved_text)
        
        # Finish call
        finish_text = f"Has Finish Call:          {finish_count} ({finish_pct:.1f}%)"
        finish_icon = "✓" if finish_count == total_queries else "✗"
        finish_color = Color.GREEN if finish_count == total_queries else Color.YELLOW
        print(f"  {self._colorize(finish_icon, finish_color)} {self._colorize(finish_text, Color.WHITE)}")
        
        # Scores
        avg_sopr = summary.get('average_sopr_score', 0.0)
        avg_api = summary.get('average_api_call_score', 0.0)
        
        sopr_color = Color.GREEN if avg_sopr >= 0.5 else Color.RED
        api_color = Color.GREEN if avg_api >= 0.5 else Color.RED
        
        self.print_info(f"Average SoPR Score:       {self._colorize(f'{avg_sopr:.3f}', sopr_color)}")
        self.print_info(f"Average API Call Score:   {self._colorize(f'{avg_api:.3f}', api_color)}")
        
        # Timing
        avg_gold_time = summary.get('average_gold_answer_time', 0.0)
        avg_system_time = summary.get('average_system_answer_time', 0.0)
        avg_eval_time = summary.get('average_evaluation_time', 0.0)
        overall_time = summary.get('overall_time', 0.0)
        
        self.print_info(f"Average Gold Answer Time:  {avg_gold_time:.3f}s")
        self.print_info(f"Average System Answer Time: {avg_system_time:.3f}s")
        self.print_info(f"Average Evaluation Time:   {avg_eval_time:.3f}s")
        self.print_info(f"Overall Time:             {overall_time:.2f}s")
        
        # Output file
        if output_file:
            file_text = f"Results saved to: {output_file}"
            colored_file = self._colorize(file_text, Color.BRIGHT_CYAN)
            print(f"\n{colored_file}")
        
        # Footer
        line = "=" * 80
        colored_line = self._colorize(line, Color.CYAN)
        print(f"\n{colored_line}")


# Global printer instance (can be customized per framework)
_default_printer = Printer(use_colors=True, verbose=True)


def get_printer(use_colors: bool = True, verbose: bool = True) -> Printer:
    """
    Get a printer instance.
    
    Args:
        use_colors: Whether to use colors
        verbose: Whether to print verbose messages
        
    Returns:
        Printer instance
    """
    return Printer(use_colors=use_colors, verbose=verbose)


# Convenience functions for quick access
def print_header(title: str, width: int = 80):
    """Print a section header."""
    _default_printer.print_header(title, width)


def print_step(step_num: int, total_steps: int, message: str, 
               sub_message: Optional[str] = None):
    """Print a step message."""
    _default_printer.print_step(step_num, total_steps, message, sub_message)


def print_info(message: str, indent: int = 0):
    """Print an info message."""
    _default_printer.print_info(message, indent)


def print_success(message: str, indent: int = 0):
    """Print a success message."""
    _default_printer.print_success(message, indent)


def print_error(message: str, indent: int = 0):
    """Print an error message."""
    _default_printer.print_error(message, indent)


def print_warning(message: str, indent: int = 0):
    """Print a warning message."""
    _default_printer.print_warning(message, indent)


def print_query_header(query_num: int, total_queries: int, query_id: str, 
                      query_text: str, num_gold_apis: int = 0):
    """Print query information header."""
    _default_printer.print_query_header(query_num, total_queries, query_id, 
                                       query_text, num_gold_apis)


def print_progress(message: str, indent: int = 2):
    """Print a progress message."""
    _default_printer.print_progress(message, indent)


def print_result(status: str, sopr_score: float, api_score: float, 
                total_time: float, answer_status: Optional[str] = None):
    """Print a query result."""
    _default_printer.print_result(status, sopr_score, api_score, total_time, 
                                 answer_status)


def print_summary(summary: Dict[str, Any], output_file: Optional[str] = None):
    """Print benchmark summary."""
    _default_printer.print_summary(summary, output_file)

