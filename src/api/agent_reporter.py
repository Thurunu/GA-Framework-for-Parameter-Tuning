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
    
    def __init__(self, master_url: str, agent_id: str = None, api_key: str = None):
        """
        Initialize agent reporter
        
        Args:
            master_url: URL of master node API (e.g., http://master:8000)
            agent_id: Unique identifier for this agent (auto-generated if None)
            api_key: API key for authentication
        """
        self.health_status = None
        self.master_url = master_url.rstrip('/')
        self.agent_id = agent_id or self._generate_agent_id()
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set up authentication header if API key provided
        if self.api_key:
            self.session.headers.update({'X-API-Key': self.api_key})
        
        self.registered = False
        self.last_heartbeat = None
        
        logger.info(f"Agent Reporter initialized: {self.agent_id}")

    def _generate_agent_id(self) -> str:
        """Generate unique agent ID based on hostname"""
        hostname = socket.gethostname()
        return f"{hostname}-{int(time.time())}"
    
    def get_system_info(self) -> Dict:
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
    
    def validate_authentication(self) -> bool:
        """Validate agent authentication with master node
        
        Returns:
            True if authentication is valid, False otherwise
        """
        try:
            # Send agent_id as query param, API key via header
            response = self.session.get(
                f"{self.master_url}/api/agents/validate",
                params={"agent_id": self.agent_id},
                headers={"X-API-Key": self.api_key},
                timeout=5
            )
            
            print(f"🔍 Validation request: {response.url}")
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"📦 Response data: {data}")
                self.registered = True
                if data.get('authenticated', False):
                    logger.info(f"✅ Authentication validated: {data.get('message', '')}")
                    return True
                else:
                    logger.warning(f"❌ Authentication failed: {data.get('message', '')}")
                    return False
            else:
                logger.error(f"❌ Validation request failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Validation error: {e}")
            return False
    def register(self, version: str = "1.0.0", metadata: Dict = None) -> bool:
        """
        Register this agent with the master node
        
        Args:
            version: Agent software version
            metadata: Additional metadata about this agent
            
        Returns:
            True if registration successful, False otherwise
        """
        # Note: For first-time registration, we don't validate since the agent doesn't exist yet
        # The master API key is used for registration, then the agent gets its own unique key
        
        try:
            system_info = self.get_system_info()
            
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
            print("🚀 Registering agent with payload:", json.dumps(payload, indent=2))
            print(f"🔑 Using API key: {self.api_key[:20]}...")
            
            # Send registration request with API key in header
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }

            response = self.session.post(
                f"{self.master_url}/api/agents/register",
                json=payload,
                headers=headers,
                timeout=10
            )
            
            print(f"🔍 Registration request URL: {response.url}")
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.registered = True
                
                # IMPORTANT: Save the agent-specific API key returned by the server
                if 'api_key' in data:
                    old_key = self.api_key
                    self.api_key = data['api_key']
                    self.session.headers.update({'X-API-Key': self.api_key})
                    logger.info(f"✅ Received agent-specific API key from server")
                    print(f"✅ Updated API key from master key to agent-specific key")
                
                logger.info(f"✅ Agent registered successfully: {self.agent_id}")
                print(f"✅ Agent registered successfully: {self.agent_id}")
                
                # Store any configuration returned by master
                if 'config' in data:
                    self.config = data['config']
                
                return True
            else:
                logger.error(f"❌ Registration failed: {response.status_code} - {response.text}")
                print(f"❌ Registration failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Registration error: {e}")
            print(f"❌ Registration error: {e}")
            return False
    
    def set_heartbeat(self, metrics: Dict, workload_type: str = None, 
                      optimization_score: float = None) -> Dict:
        
        self.health_status = {
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "status": "healthy",
                "metrics": metrics,
                "workload_type": workload_type,
                "optimization_score": optimization_score
            }

    def send_workload_info(self, metrics: Dict) -> List[str]:
        """
        Send workload metrics to central server"""
        print(f"⛔⛔⛔⛔⛔⛔")

        print("Sending workload metrics to master:", json.dumps(metrics, indent=2))
    
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
      
   
    def post_optimized_parameters(self, workload_type: str, parameters: Dict[str, Any], 
                                  optimization_score: float = None, 
                                  optimization_time: float = None) -> bool:
        """
        Send optimized kernel parameters to master node
        
        Args:
            workload_type: Type of workload the parameters are optimized for
            parameters: Dictionary of parameter names and their optimized values
            optimization_score: Final optimization score achieved
            optimization_time: Time taken for optimization in seconds
            
        Returns:
            True if parameters were sent successfully
        """
        try:
            payload = {
                "agent_id": self.agent_id,
                "workload_type": workload_type,
                "parameters": parameters,
                "optimization_score": optimization_score,
                "optimization_time": optimization_time,
                "timestamp": datetime.now().isoformat(),
                "parameter_count": len(parameters)
            }
            print(f"DEBUG: payload data: {json.dumps(payload, indent=2)}")
            response = self.session.post(
                f"{self.master_url}/api/parameters/optimized_parameters",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Optimized parameters sent to master: {workload_type}")
                return True
            else:
                logger.warning(f"⚠️ Failed to send parameters: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Parameter send error: {e}")
            return False
    
    def post_current_parameters(self) -> Dict:
        """
        Get current kernel parameters from master
        
        Returns:
            Dictionary of current parameters
        """
        try:
            response = self.session.post(
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
    
    # =========================================================================
    # CONFIGURATION SYNCHRONIZATION - NEW FEATURE
    # =========================================================================
    
    def fetch_configuration_updates(self) -> Dict[str, Any]:
        """
        Fetch pending configuration updates from master server
        
        Returns:
            Dictionary containing configuration updates:
            {
                'has_updates': bool,
                'updates': {
                    'add_parameters': [...],
                    'update_parameters': [...],
                    'add_workloads': [...],
                    'update_workloads': [...],
                    'apply_to_system': [...]
                },
                'timestamp': str
            }
        """
        try:
            response = self.session.get(
                f"{self.master_url}/api/agents/{self.agent_id}/config_updates",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('has_updates', False):
                    logger.info(f"📥 Configuration updates available from master")
                return data
            else:
                logger.warning(f"⚠️ Failed to fetch config updates: {response.status_code}")
                return {'has_updates': False, 'updates': {}}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Config update fetch error: {e}")
            return {'has_updates': False, 'updates': {}}
    
    def acknowledge_configuration_update(self, update_id: str, status: str, 
                                        results: Dict = None) -> bool:
        """
        Acknowledge that configuration update has been applied
        
        Args:
            update_id: ID of the configuration update
            status: Status ('success', 'partial', 'failed')
            results: Detailed results of the update operation
            
        Returns:
            True if acknowledgment was sent successfully
        """
        try:
            payload = {
                'update_id': update_id,
                'status': status,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            response = self.session.post(
                f"{self.master_url}/api/agents/{self.agent_id}/config_update_ack",
                json=payload,
                timeout=3  # Reduced timeout to 3 seconds
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Configuration update acknowledged: {update_id}")
                return True
            else:
                logger.warning(f"⚠️ Config update ACK failed: {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ Config update ACK timeout (master server not responding)")
            return False
        except requests.exceptions.ConnectionError:
            logger.warning(f"⚠️ Config update ACK failed (master server not reachable)")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Config update ACK error: {e}")
            return False
    
    def request_configuration_sync(self) -> Dict[str, Any]:
        """
        Request full configuration synchronization from master
        
        Returns:
            Complete configuration from master server
        """
        try:
            response = self.session.get(
                f"{self.master_url}/api/agents/{self.agent_id}/full_config",
                timeout=15
            )
            
            if response.status_code == 200:
                config = response.json()
                logger.info(f"✅ Full configuration synced from master")
                return config
            else:
                logger.warning(f"⚠️ Config sync failed: {response.status_code}")
                return {}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Config sync error: {e}")
            return {}
    
    def report_current_configuration(self, config_data: Dict[str, Any]) -> bool:
        """
        Report current configuration state to master
        
        Args:
            config_data: Current configuration data including:
                - kernel_parameters
                - workload_patterns
                - optimization_profiles
                
        Returns:
            True if report was sent successfully
        """
        try:
            payload = {
                'agent_id': self.agent_id,
                'configuration': config_data,
                'timestamp': datetime.now().isoformat()
            }
            
            response = self.session.post(
                f"{self.master_url}/api/agents/{self.agent_id}/current_config",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Current configuration reported to master")
                return True
            else:
                logger.warning(f"⚠️ Config report failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Config report error: {e}")
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
    
