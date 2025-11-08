"""
Progress tracking utilities for consistent reporting.
"""
from typing import Dict


class ProgressTracker:
    """Track and display progress during abstract fetching."""
    
    def __init__(self, total: int, phase_name: str = ""):
        self.total = total
        self.current = 0
        self.phase_name = phase_name
        self.source_stats = {}
    
    def start_phase(self, phase_name: str):
        """Start a new phase."""
        self.phase_name = phase_name
        print("\n" + "=" * 80)
        print(phase_name)
        print("=" * 80)
    
    def update(self, title: str, success: bool = False, source: str = None):
        """Update progress for a single document."""
        self.current += 1
        print(f"[{self.current}/{self.total}] {title[:60]}...")
        
        if success and source:
            self.source_stats[source] = self.source_stats.get(source, 0) + 1
    
    def log_attempt(self, source: str, success: bool):
        """Log an attempt to fetch from a source."""
        symbol = "✓" if success else "✗"
        print(f"  Trying {source}...", end=" ", flush=True)
        print(symbol)
    
    def log_success(self, source: str, identifier: str):
        """Log a successful fetch."""
        print(f"  ✓ Success! Found via {source} (ID: {identifier})")
    
    def log_failure(self, message: str = "Not found"):
        """Log a failure."""
        print(f"  ✗ {message}")
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total processed: {self.total}")
        print(f"Successfully retrieved: {sum(self.source_stats.values())}")
        print(f"Failed: {self.total - sum(self.source_stats.values())}")
        
        if self.source_stats:
            success_rate = sum(self.source_stats.values()) / self.total * 100
            print(f"Success rate: {success_rate:.2f}%")
            
            print("\nAbstracts by source:")
            for source, count in sorted(self.source_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {source}: {count}")
        
        print("=" * 80)
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            'total': self.total,
            'processed': self.current,
            'sources': self.source_stats.copy()
        }

