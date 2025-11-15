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
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from PerformanceMonitor import PerformanceMonitor
from agent_reporter import AgentReporter
import getpass
from CentralDataStore import get_data_store


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
            
            if master_url and api_key:
                cls.reporter = AgentReporter(
                    agent_id=f"{username}@{hostname}",
                    master_url=master_url,
                    api_key=api_key
                )
                cls.reporter.register()
                print(f"✅ Connected to {master_url} as {username}@{hostname}")
                
                # Store reporter in central data store for global access
                data_store = get_data_store()
                data_store.set_agent_reporter(cls.reporter)
                data_store.set_agent_registered(f"{username}@{hostname}", master_url)
                
            cls.reporter_initialized = True
        except Exception as e:
            print(f"❌ Failed to connect to centralized management: {e}")
            cls.reporter = None
            cls.reporter_initialized = True
        print("Starting Continuous Kernel Optimization System...")
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {self.address_string()} - {format % args}")

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
                        "/health": "Health check",
                        "/status": "JSON status and metrics",
                        "/metrics": "Prometheus format metrics",
                        "/info": "System information",
                        "/workloads": "Current workload information",
                        "/data": "All central data store information"
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

            # Prometheus format metrics
            elif path == '/metrics':
                metrics = self.collect_current_metrics()
                system_info = self.collect_system_info()
                prometheus_text = self.format_metrics(metrics, system_info)
                self.send_text_response(prometheus_text)

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
                    "active_workloads": workloads_dict,
                    "total_workloads": len(workloads_dict)
                }
                self.send_json_response(response)
            
            # All data from central data store
            elif path == '/data':
                all_data = self.data_store.get_all_data()
                self.send_json_response(all_data)

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
        """Handle POST requests (optional - for future use)"""
        self.send_json_response(
            {"error": "POST not supported",
                "message": "This server only accepts GET requests"},
            status_code=405
        )


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
    


