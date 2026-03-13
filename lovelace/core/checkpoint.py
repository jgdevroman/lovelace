"""Checkpoint management for pipeline resumability."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class PipelineCheckpoint:
    """Manages checkpoints for resumable pipeline execution."""
    
    STEPS = [
        "analysis",
        "clustering", 
        "gateway",
        "documentation",
        "shared_service",
        # Domain services added dynamically
    ]
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, step: str, data: dict) -> Path:
        """Save checkpoint data for a step."""
        checkpoint_file = self.checkpoint_dir / f"{step}.json"
        
        # Add metadata
        data["_checkpoint"] = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved checkpoint: {step}")
        return checkpoint_file
    
    def load(self, step: str) -> Optional[dict]:
        """Load checkpoint data for a step."""
        checkpoint_file = self.checkpoint_dir / f"{step}.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded checkpoint: {step}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {step}: {e}")
            return None
    
    def exists(self, step: str) -> bool:
        """Check if a checkpoint exists for a step."""
        return (self.checkpoint_dir / f"{step}.json").exists()
    
    def get_completed_steps(self) -> List[str]:
        """Get list of completed steps."""
        completed = []
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            step = checkpoint_file.stem
            completed.append(step)
        return completed
    
    def get_resume_point(self) -> Optional[str]:
        """Determine which step to resume from."""
        completed = set(self.get_completed_steps())
        
        # Check standard steps in order
        for i, step in enumerate(self.STEPS):
            if step not in completed:
                return step if i == 0 else self.STEPS[i - 1]
        
        return None
    
    def clear(self):
        """Clear all checkpoints."""
        for f in self.checkpoint_dir.glob("*.json"):
            f.unlink()
        logger.info("Cleared all checkpoints")
    
    
    def request_approval(self, service_name: str, service_data: dict):
        """Save a pending approval checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"approval_{service_name}.json"
        data = {
            "service_name": service_name,
            "status": "pending",
            "requested_at": datetime.now().isoformat(),
            "service_data": service_data
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Requested approval for {service_name}")
    
    def is_pending_approval(self, service_name: str) -> bool:
        """Check if a service is pending approval."""
        checkpoint_file = self.checkpoint_dir / f"approval_{service_name}.json"
        if not checkpoint_file.exists():
            return False
            
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("status") == "pending"
    
    def approve_service(self, service_name: str):
        """Mark a service as approved."""
        checkpoint_file = self.checkpoint_dir / f"approval_{service_name}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["status"] = "approved"
            data["approved_at"] = datetime.now().isoformat()
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Approved service {service_name}")

    def get_service_checkpoints(self) -> Dict[str, dict]:
        """Get checkpoints for completed services."""
        services = {}
        for checkpoint_file in self.checkpoint_dir.glob("service_*.json"):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                service_name = checkpoint_file.stem.replace("service_", "")
                services[service_name] = data
            except Exception:
                pass
        return services
