"""
Agent Reporter - Communicates with Master Node

This module handles communication between the agent (running on VMs) 
and the master node (central management server).

Features:
- Registers agent with master node
- Sends periodic heartbeats with system metrics
- Polls for commands from master
- Reports command execution results
- Handles workload change notifications
"""

import requests
import time
import socket
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import psutil
import platform

logger = logging.getLogger(__name__)


class AgentReporter:
    """Handles communication between agent and master node"""
    
    def __init__(self, master_url: str, agent_id: str = None, ):
        """
        Initialize agent reporter
        
        Args:
            master_url: URL of master node API (e.g., http://master:8000)
            agent_id: Unique identifier for this agent (auto-generated if None)
            api_key: API key for authentication
        """
        self.master_url = master_url.rstrip('/')
        self.agent_id = agent_id or self._generate_agent_id()
        # self.api_key = api_key
        self.session = requests.Session()
        
        # Set up authentication header if API key provided
        # if self.api_key:
        #     self.session.headers.update({'X-API-Key': self.api_key})
        
        self.registered = False
        self.last_heartbeat = None
        
        logger.info(f"Agent Reporter initialized: {self.agent_id}")
    
    def _generate_agent_id(self) -> str:
        """Generate unique agent ID based on hostname"""
        hostname = socket.gethostname()
        return f"{hostname}-{int(time.time())}"
    
    def _get_system_info(self) -> Dict:
        """Collect system information"""
        return {
            "hostname": socket.gethostname(),
            "ip_address": self._get_ip_address(),
            "os": platform.system(),
            "os_version": platform.release(),
            "architecture": platform.machine(),
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }
    
    def _get_ip_address(self) -> str:
        """Get primary IP address"""
        try:
            # Connect to external address to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    def register(self, version: str = "1.0.0", metadata: Dict = None) -> bool:
        """
        Register this agent with the master node
        
        Args:
            version: Agent software version
            metadata: Additional metadata about this agent
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            system_info = self._get_system_info()
            
            payload = {
                "agent_id": self.agent_id,
                "hostname": system_info["hostname"],
                "ip_address": system_info["ip_address"],
                "version": version,
                "metadata": {
                    **system_info,
                    **(metadata or {})
                }
            }
            print("Registering agent with payload:", json.dumps(payload, indent=2))
            
            response = self.session.post(
                f"{self.master_url}/api/agents/register",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.registered = True
                logger.info(f"✅ Agent registered successfully: {self.agent_id}")
                print(f"✅ Agent registered successfully: {self.agent_id}")
                
                # Store any configuration returned by master
                if 'config' in data:
                    self.config = data['config']
                
                return True
            else:
                logger.error(f"❌ Registration failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Registration error: {e}")
            return False
    
    def send_heartbeat(self, metrics: Dict, workload_type: str = None, 
                      optimization_score: float = None) -> Dict:
        """
        Send heartbeat with current metrics to master node
        
        Args:
            metrics: Current system metrics (CPU, memory, etc.)
            workload_type: Current workload type being handled
            optimization_score: Current optimization score
            
        Returns:
            Response from master (may contain commands to execute)
        """
        try:
            payload = {
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "status": "healthy",
                "metrics": metrics,
                "workload_type": workload_type,
                "optimization_score": optimization_score
            }
            
            response = self.session.post(
                f"{self.master_url}/api/agents/{self.agent_id}/heartbeat",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                self.last_heartbeat = time.time()
                return response.json()
            else:
                logger.warning(f"⚠️ Heartbeat failed: {response.status_code}")
                return {"status": "error", "has_commands": False}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Heartbeat error: {e}")
            return {"status": "error", "has_commands": False}
    
    def poll_commands(self) -> List[Dict]:
        """
        Poll master node for pending commands
        
        Returns:
            List of commands to execute
        """
        try:
            response = self.session.get(
                f"{self.master_url}/api/agents/{self.agent_id}/commands",
                timeout=10
            )
            
            if response.status_code == 200:
                commands = response.json()
                if commands:
                    logger.info(f"📬 Received {len(commands)} command(s)")
                return commands
            else:
                logger.warning(f"⚠️ Command poll failed: {response.status_code}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Command poll error: {e}")
            return []
    
    def report_command_result(self, command_id: int, status: str, 
                            result: Any = None, error: str = None) -> bool:
        """
        Report command execution result back to master
        
        Args:
            command_id: ID of the command that was executed
            status: Execution status ('success', 'failed', 'error')
            result: Result data (if successful)
            error: Error message (if failed)
            
        Returns:
            True if report was sent successfully
        """
        try:
            payload = {
                "command_id": command_id,
                "status": status,
                "result": result,
                "error": error,
                "executed_at": datetime.now().isoformat()
            }
            
            response = self.session.post(
                f"{self.master_url}/api/agents/{self.agent_id}/command_result",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Command result reported: {command_id}")
                return True
            else:
                logger.warning(f"⚠️ Result report failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Result report error: {e}")
            return False
    
    def report_workload_change(self, previous_workload: str, 
                              current_workload: str, confidence: float = 1.0) -> bool:
        """
        Report workload type change to master
        
        Args:
            previous_workload: Previous workload type
            current_workload: New workload type
            confidence: Confidence level in detection (0-1)
            
        Returns:
            True if report was sent successfully
        """
        try:
            payload = {
                "previous_workload": previous_workload,
                "current_workload": current_workload,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
            response = self.session.post(
                f"{self.master_url}/api/agents/{self.agent_id}/workload_change",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"🔄 Workload change reported: {previous_workload} → {current_workload}")
                return True
            else:
                logger.warning(f"⚠️ Workload change report failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Workload change report error: {e}")
            return False
    
    def report_optimization_progress(self, iteration: int, score: float, 
                                    total_iterations: int = None) -> bool:
        """
        Report optimization progress in real-time
        
        Args:
            iteration: Current iteration number
            score: Current optimization score
            total_iterations: Total planned iterations
            
        Returns:
            True if report was sent successfully
        """
        try:
            payload = {
                "iteration": iteration,
                "score": score,
                "total_iterations": total_iterations,
                "timestamp": datetime.now().isoformat()
            }
            
            response = self.session.post(
                f"{self.master_url}/api/agents/{self.agent_id}/optimization_progress",
                json=payload,
                timeout=5
            )
            
            return response.status_code == 200
                
        except requests.exceptions.RequestException:
            # Don't log errors for progress updates (too noisy)
            return False
    
    def report_error(self, error_type: str, error_message: str, 
                    severity: str = "error") -> bool:
        """
        Report an error or alert to master
        
        Args:
            error_type: Type of error
            error_message: Error details
            severity: Severity level ('info', 'warning', 'error', 'critical')
            
        Returns:
            True if report was sent successfully
        """
        try:
            payload = {
                "error_type": error_type,
                "message": error_message,
                "severity": severity,
                "timestamp": datetime.now().isoformat()
            }
            
            response = self.session.post(
                f"{self.master_url}/api/agents/{self.agent_id}/error",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"🚨 Error reported: {error_type}")
                return True
            else:
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error report failed: {e}")
            return False
    
    def get_current_parameters(self) -> Dict:
        """
        Get current kernel parameters from master
        
        Returns:
            Dictionary of current parameters
        """
        try:
            response = self.session.get(
                f"{self.master_url}/api/agents/{self.agent_id}/parameters",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Parameter fetch error: {e}")
            return {}
    
    def health_check(self) -> bool:
        """
        Check if master node is reachable
        
        Returns:
            True if master is reachable, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.master_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


if __name__ == "__main__":
    # Test agent reporter
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing Agent Reporter...")
    
    # Initialize reporter (will fail if master not running - that's OK for testing)
    reporter = AgentReporter(
        master_url="http://localhost:8000",
        agent_id="test-agent-001"
    )
    
    print(f"✅ Agent ID: {reporter.agent_id}")
    print(f"✅ Master URL: {reporter.master_url}")
    
    # Test metrics collection
    test_metrics = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_io_read": psutil.disk_io_counters().read_bytes if psutil.disk_io_counters() else 0,
        "disk_io_write": psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0,
    }
    
    print(f"\n📊 Test Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Agent Reporter ready!")
    print("💡 Start the master node to test full communication")
