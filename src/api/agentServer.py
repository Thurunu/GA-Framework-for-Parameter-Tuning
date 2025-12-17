"""
Lightweight Agent API Server - Using Built-in Python HTTP Server
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import psutil
import socket
import platform
import os
import time
import logging
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from PerformanceMonitor import PerformanceMonitor
from agent_reporter import AgentReporter
import getpass
from CentralDataStore import get_data_store
import sys
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

logger = logging.getLogger(__name__)

project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'monitoring'))
sys.path.insert(0, os.path.join(project_root, 'data'))
class AgentMetricsHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for agent metrics.
    Handles GET requests to various endpoints.
    """
    # Class-level variables to share across all handler instances
    reporter = None
    reporter_initialized = False
    
    def __init__(self, *args, **kwargs):
        self.data_store = get_data_store()
        # Initialize reporter once at class level
        if not AgentMetricsHandler.reporter_initialized:
            AgentMetricsHandler._initialize_reporter_class()
        # Call parent class init (this handles the actual request)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def _initialize_reporter_class(cls):
        """Initialize the reporter once for all handler instances"""
        print("🌍 Connecting to centralized management...")
        try:
            master_url = os.getenv("MASTER_URL")
            api_key = os.getenv("API_KEY")
            username = getpass.getuser()  # safer than os.getlogin()
            hostname = socket.gethostname()
            # print(f"MASTER_URL: {master_url}, API_KEY: {api_key}")
            if master_url and api_key:
                cls.reporter = AgentReporter(
                    agent_id=f"{username}@{hostname}",
                    master_url=master_url,
                    api_key=api_key
                )
                cls.reporter.register()
                
                # Get workload information from central data store
                data_store = get_data_store()
                current_workload = data_store.get_current_workload_type()
                active_workloads = data_store.get_active_workloads()
                optimization_status = data_store.get_optimization_status()
                available_workload_types = data_store.get_available_workload_types()
                
                print(f"✅ Connected to {master_url} as {username}@{hostname}")
                print(f"📊 Current Workload Type: {current_workload}")
                print(f"📊 Active Workloads Count: {len(active_workloads)}")
                print(f"📊 Available Workload Types: {available_workload_types}")
                print(f"📊 Optimization Status: {optimization_status}")
                
                # Store reporter in central data store for global access
                data_store.set_agent_reporter(cls.reporter)
                data_store.set_agent_registered(f"{username}@{hostname}", master_url)
                
            cls.reporter_initialized = True
        except Exception as e:
            print(f"❌ Failed to connect to centralized management: {e}")
            cls.reporter = None
            cls.reporter_initialized = True
        # print("Starting Continuous Kernel Optimization System...")
    
    def log_message(self, format, *args):
        """Override to customize logging"""
      
    
    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')  # CORS
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def send_text_response(self, text, status_code=200):
        """Send plain text response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(text.encode())

    def collect_system_info(self):
        """Collect basic system information - reuses AgentReporter's method"""
        if self.reporter:
            # Use the reporter's get_system_info method for consistency
            return self.reporter.get_system_info()
        else:
            # Fallback if reporter is not initialized
            return {
                "hostname": socket.gethostname(),
                "platform": platform.system(),
                "platform_version": platform.release(),
                "architecture": platform.machine(),
                "cpu_count": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            }

    def collect_current_metrics(self):
        """
        Collect current system metrics from central data store or on-demand.
        Called when /metrics or /status is requested.
        """
        # Try to get metrics from central data store first
        metrics = self.data_store.get_current_metrics()
        
        if metrics is None:
            # Fallback: collect metrics on-demand if not available in store
            monitor = PerformanceMonitor()
            metrics = monitor._collect_metrics()
        
        # Convert PerformanceMetrics dataclass to dictionary
        return {
            "timestamp": metrics.timestamp,
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "memory_available_bytes": metrics.memory_available,
            "disk_usage_percent": self._get_disk_usage(),
            "disk_io_read_bytes": metrics.disk_io_read,
            "disk_io_write_bytes": metrics.disk_io_write,
            "network_bytes_sent": metrics.network_bytes_sent,
            "network_bytes_recv": metrics.network_bytes_recv,
            "load_average_1m": metrics.load_average[0] if metrics.load_average else 0,
            "load_average_5m": metrics.load_average[1] if len(metrics.load_average) > 1 else 0,
            "load_average_15m": metrics.load_average[2] if len(metrics.load_average) > 2 else 0,
            "context_switches": metrics.context_switches,
            "interrupts": metrics.interrupts,
            "tcp_connections_active": metrics.tcp_connections
        }

    def _get_disk_usage(self):
        """Get disk usage percentage for root partition"""
        try:
            return psutil.disk_usage('/').percent
        except:
            try:
                return psutil.disk_usage('C:\\').percent
            except:
                return 0

    def format_metrics(self, metrics, system_info):
        """Format metrics in Prometheus text format"""
        lines = []
        labels = f'hostname="{system_info["hostname"]}",platform="{system_info["platform"]}"'

        def add_metric(name, value, help_text, metric_type="gauge"):
            lines.append(f"# HELP agent_{name} {help_text}")
            lines.append(f"# TYPE agent_{name} {metric_type}")
            lines.append(f"agent_{name}{{{labels}}} {value}")
            lines.append("")

        # Add metrics
        add_metric("cpu_percent",
                   metrics["cpu_percent"], "CPU usage percentage")
        add_metric("memory_percent",
                   metrics["memory_percent"], "Memory usage percentage")
        add_metric("memory_available_bytes",
                   metrics["memory_available_bytes"], "Available memory", "gauge")
        add_metric("disk_usage_percent",
                   metrics["disk_usage_percent"], "Disk usage percentage")
        add_metric("load_average_1m",
                   metrics["load_average_1m"], "Load average 1 minute")
        add_metric("tcp_connections",
                   metrics["tcp_connections_active"], "Active TCP connections")

        return "\n".join(lines)

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        try:
            # Root endpoint
            if path == '/':
                response = {
                    "service": "Lightweight Agent Metrics Server",
                    "version": "1.0.0",
                    "implementation": "Python http.server (built-in)",
                    "endpoints": {
                        "GET /health": "Health check",
                        "GET /status": "JSON status and metrics",
                        "GET /metrics": "metrics",
                        "GET /info": "System information",
                        "GET /workloads": "Current workload information and available workload types",
                        "GET /parameters": "Current optimized kernel parameters and optimization status",
                        "GET /data": "All central data store information",
                        "POST /workload/parameters": "Get parameters and ranges for a specific workload type (requires: workload_type)",
                        "POST /config/update": "Apply batch configuration updates from centralized management",
                        "POST /config/validate": "Validate all configurations for referential integrity",
                        "POST /config/current": "Get current configuration state",
                        "POST /config/reload": "Reload all configurations from disk"
                    }
                }
                self.send_json_response(response)

            # Health check
            elif path == '/health':
                response = {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "uptime_seconds": time.time() - psutil.boot_time(),
                    "agent_id": socket.gethostname()
                }
                self.send_json_response(response)

            # System info
            elif path == '/info':
                system_info = self.collect_system_info()
                self.send_json_response(system_info)

            # JSON status (easy to parse)
            elif path == '/status':
                metrics = self.collect_current_metrics()
                system_info = self.collect_system_info()

                response = {
                    "agent_id": socket.gethostname(),
                    "timestamp": datetime.now().isoformat(),
                    "status": "active",
                    "system_info": system_info,
                    "metrics": metrics
                }
                self.send_json_response(response)

            # Get optimization profile data
            elif path == '/optimization-profile':
                profiles = self.data_store.get_optmization_profiles()
                json_ready = {
                    key: {
                        "workload_type": p.workload_type,
                        "parameter_bounds": p.parameter_bounds,
                        "strategy": p.strategy.value,
                        "evaluation_budget": p.evaluation_budget,
                        "time_budget": p.time_budget,
                        "performance_weights": p.performance_weights
                    }
                    for key, p in profiles.items()
                }

                self.send_json_response(json_ready)

            # Current metrics only
            elif path == '/metrics/current':
                metrics = self.collect_current_metrics()
                response = {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics
                }
                self.send_json_response(response)
            
            # Workload information from central data store
            elif path == '/workloads':
                workloads = self.data_store.get_active_workloads()
                current_workload = self.data_store.get_current_workload_type()
                available_workload_types = self.data_store.get_available_workload_types()
                
                # Convert WorkloadInfo objects to dictionaries
                workloads_dict = {}
                for pid, workload in workloads.items():
                    workloads_dict[pid] = {
                        "pid": workload.pid,
                        "name": workload.name,
                        "workload_type": workload.workload_type,
                        "cpu_percent": workload.cpu_percent,
                        "memory_percent": workload.memory_percent
                    }
                
                response = {
                    "current_workload_type": current_workload,
                    "available_workload_types": available_workload_types,
                    "active_workloads": workloads_dict,
                    "total_workloads": len(workloads_dict)
                }
                self.send_json_response(response)
            
            # Get current optimized kernel parameters
            elif path == '/parameters':
                kernel_params = self.data_store.get_kernel_parameters()
                current_workload = self.data_store.get_current_workload_type()
                optimization_status = self.data_store.get_optimization_status()
                
                response = {
                    "current_workload_type": current_workload,
                    "parameters": kernel_params,
                    "parameter_count": len(kernel_params),
                    "optimization_status": {
                        "is_optimizing": optimization_status.get("is_optimizing", False),
                        "current_profile": optimization_status.get("current_profile"),
                        "last_optimization_time": optimization_status.get("last_optimization_time").isoformat() if optimization_status.get("last_optimization_time") else None,
                        "optimization_count": optimization_status.get("optimization_count", 0)
                    },
                    "timestamp": datetime.now().isoformat()
                }
                self.send_json_response(response)

            # Get current configuration (GET) - supports optional reporting via query param
            elif path == '/config/current':
                """
                Return current configuration state via GET.
                Use query param `report_to_master=true` to request reporting to master.
                """
                from ConfigurationManager import get_config_manager

                # Parse query params for reporting flag
                query = parse_qs(parsed_path.query)
                report_flag = query.get('report_to_master', ['false'])[0].lower()
                report_to_master = report_flag in ('1', 'true', 'yes')

                config_mgr = get_config_manager()
                current_config = config_mgr.get_current_configurations()
                print(f"🌍 Setting up current parameters (GET): {json.dumps(current_config, indent=2)}")

                ack_status = None
                if self.reporter and report_to_master:
                    try:
                        sent = self.reporter.report_current_configuration(current_config)
                        ack_status = 'sent' if sent else 'failed'
                        if not sent:
                            logger.warning("Failed to report current configuration to master (non-exception)")
                    except Exception as e:
                        logger.exception("Exception while reporting current configuration to master: %s", e)
                        ack_status = 'error'

                response = {
                    "configuration": current_config,
                    "timestamp": datetime.now().isoformat(),
                    "report_acknowledgement": ack_status
                }

                self.send_json_response(response)
            
            

            # 404 - Not found
            else:
                self.send_json_response(
                    {"error": "Not found", "path": path},
                    status_code=404
                )

        except Exception as e:
            # 500 - Internal server error
            self.send_json_response(
                {"error": str(e), "type": type(e).__name__},
                status_code=500
            )

    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
                request_data = json.loads(body.decode('utf-8'))
            else:
                request_data = {}
            
            # Get workload parameters endpoint
            if path == '/workload/parameters':
                workload_type = request_data.get('workload_type')
                
                if not workload_type:
                    self.send_json_response(
                        {"error": "Missing required field: workload_type"},
                        status_code=400
                    )
                    return
                
                # Get optimization profiles from central data store
                profiles = self.data_store._optimization_profiles
                
                if workload_type not in profiles:
                    available_types = list(profiles.keys())
                    self.send_json_response(
                        {
                            "error": f"Unknown workload type: {workload_type}",
                            "available_workload_types": available_types
                        },
                        status_code=404
                    )
                    return
                
                # Get the profile for the requested workload type
                profile = profiles[workload_type]
                
                # Format parameter bounds for response
                parameters = {}
                for param_name, bounds in profile.parameter_bounds.items():
                    parameters[param_name] = {
                        "min": bounds[0],
                        "max": bounds[1],
                        "range": bounds[1] - bounds[0]
                    }
                
                response = {
                    "workload_type": workload_type,
                    "strategy": profile.strategy.value if hasattr(profile.strategy, 'value') else str(profile.strategy),
                    "evaluation_budget": profile.evaluation_budget,
                    "time_budget": profile.time_budget,
                    "parameters": parameters,
                    "performance_weights": profile.performance_weights,
                    "total_parameters": len(parameters)
                }
                
                self.send_json_response(response)
            
            # NEW: Configuration update endpoint
            elif path == '/config/update':
                """
                Handle configuration updates from central management server
                Request body:
                {
                    "update_id": "unique-update-id",
                    "updates": {
                        "add_parameters": [...],
                        "update_parameters": [...],
                        "add_workloads": [...],
                        "update_workloads": [...],
                        "add_optimization_profiles": [...],
                        "update_optimization_profiles": [...],
                        "delete_optimization_profiles": [...],
                        "apply_to_system": [...]
                    }
                }
                """
                from ConfigurationManager import get_config_manager
                
                update_id = request_data.get('update_id', 'unknown')
                updates = request_data.get('updates', {})
                
                if not updates:
                    self.send_json_response(
                        {"error": "No updates provided"},
                        status_code=400
                    )
                    return
                
                # Get configuration manager
                config_mgr = get_config_manager()
                print(f"🔄 Applying configuration update: {update_id}")
                
                # Apply batch updates
                results = config_mgr.apply_batch_update(updates)
                
                # Report results back to master if reporter is available (non-blocking)
                ack_status = None
                if self.reporter:
                    try:
                        status = 'success' if results['failed'] == 0 else ('partial' if results['successful'] > 0 else 'failed')
                        ack_success = self.reporter.acknowledge_configuration_update(update_id, status, results)
                        ack_status = 'sent' if ack_success else 'failed'
                    except Exception as e:
                        # Don't fail the whole request if acknowledgment fails
                        logger.warning(f"Failed to acknowledge config update to master: {e}")
                        ack_status = 'timeout'
                
                response = {
                    "update_id": update_id,
                    "status": "completed",
                    "results": results,
                    "acknowledgment": ack_status,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.send_json_response(response)
            
            # NEW: Validate configurations endpoint
            elif path == '/config/validate':
                """
                Validate all configurations for referential integrity
                Returns validation results with errors and warnings
                """
                from ConfigurationManager import get_config_manager
                
                config_mgr = get_config_manager()
                validation_results = config_mgr.validate_configurations()
                
                response = {
                    "validation": validation_results,
                    "timestamp": datetime.now().isoformat()
                }
                
                status_code = 200 if validation_results['valid'] else 422
                self.send_json_response(response, status_code=status_code)
            
            # NEW: Get current configuration endpoint
            # elif path == '/config/current':
            #     """
            #     Return current configuration state
            #     """
            #     from ConfigurationManager import get_config_manager
                
            #     config_mgr = get_config_manager()
            #     current_config = config_mgr.get_current_configurations()
            #     print(f"🌍 Setting up current parameters: {json.dumps(current_config, indent=2)}")
            #     # Optionally report to master (safe, non-blocking handling)
            #     ack_status = None
            #     if self.reporter and request_data.get('report_to_master', False):
            #         try:
            #             sent = self.reporter.report_current_configuration(current_config)
            #             ack_status = 'sent' if sent else 'failed'
            #             if not sent:
            #                 logger.warning("Failed to report current configuration to master (non-exception)")
            #         except Exception as e:
            #             # Log exception and return ack status so caller can see failure
            #             logger.exception(f"Exception while reporting current configuration to master: {e}")
            #             ack_status = 'error'

            #     response = {
            #         "configuration": current_config,
            #         "timestamp": datetime.now().isoformat(),
            #         "report_acknowledgement": ack_status
            #     }

            #     self.send_json_response(response)
            
            # NEW: Reload configurations endpoint
            elif path == '/config/reload':
                """
                Reload all configurations from disk
                """
                from ConfigurationManager import get_config_manager
                
                config_mgr = get_config_manager()
                reload_status = config_mgr.reload_configurations()
                
                response = {
                    "reload_status": reload_status,
                    "all_successful": all(reload_status.values()),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.send_json_response(response)
            
            else:
                # 404 - Not found
                self.send_json_response(
                    {"error": "Not found", "path": path},
                    status_code=404
                )
        
        except json.JSONDecodeError as e:
            self.send_json_response(
                {"error": "Invalid JSON in request body", "details": str(e)},
                status_code=400
            )
        except Exception as e:
            # 500 - Internal server error
            self.send_json_response(
                {"error": str(e), "type": type(e).__name__},
                status_code=500
            )
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


def start_server(HOST='0.0.0.0', PORT=9300):
    """
    Args:
        HOST: HOST to bind to (0.0.0.0 = all interfaces)
        PORT: PORT to listen on (9300 is standard)
    """
    # Initialize reporter before starting server
    AgentMetricsHandler._initialize_reporter_class()
    
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, AgentMetricsHandler)

    print(f"""✅ Server started at http://{HOST}:{PORT}

    Press Ctrl+C to stop the server""")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        httpd.shutdown()


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    PORT = 9300
    HOST = '0.0.0.0'

    if len(sys.argv) > 1:
        try:
            PORT = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default 9300")

    if len(sys.argv) > 2:
        HOST = sys.argv[2]

    # Check if psutil is available
    try:
        import psutil
    except ImportError:
        print("⚠️  ERROR: psutil is required but not installed.")
        print("Install it with: pip install psutil")
        sys.exit(1)
    
    # Start the server
    start_server(HOST=HOST, PORT=PORT)
    


