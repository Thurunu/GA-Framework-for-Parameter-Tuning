class SystemStatus:
    def __init__(self, workload_info, current_params):
        pass
    def get_status(self) -> Dict:
        """Get current status of continuous optimizer"""
        workload_info = self.process_detector.get_current_workload_info()
        current_params = self.kernel_interface.get_current_configuration()
        
        # Determine the active profile based on current workload
        detected_workload = workload_info['dominant_workload']
        active_profile = detected_workload if detected_workload in self.OPTIMIZATION_PROFILES else 'general'
        
        return {
            'running': self.running,
            'optimization_in_progress': self.optimization_in_progress,
            'current_workload': detected_workload,
            'active_profile': active_profile,  # Profile based on detected workload
            'last_optimized_profile': self.current_profile.workload_type if self.current_profile else None,
            'last_optimization': self.last_optimization_time,
            'active_processes': workload_info['active_processes'],
            'current_parameters': current_params,
            'queue_size': self.optimization_queue.qsize()
        }
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """Get current parameter configuration as a simple dict"""
        return {
            name: param.current_value 
            for name, param in self.optimization_parameters.items()
        }
    # Get current workload information
    def get_current_workload_info(self) -> Dict:
        return {
            'dominant_workload': self.dominant_workload,
            'active_processes': len(self.current_processes),
            'workload_history': list(self.workload_history)[-10:],  # Last 10 entries
            'process_details': [
                {
                    'name': proc.name,
                    'workload_type': proc.workload_type,
                    'cpu_percent': proc.cpu_percent,
                    'memory_percent': proc.memory_percent
                }
                for proc in self.current_processes.values()
            ]
        }
