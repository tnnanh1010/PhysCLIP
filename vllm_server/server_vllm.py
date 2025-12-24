import subprocess
import os
import sys
import fire

def launch_vllm_servers(
    model_path,
    gpus_per_instance,
    base_port=8000,
    total_gpus=4,
    max_model_len=None,
    enable_yarn=False
):
    if gpus_per_instance <= 0 or gpus_per_instance > total_gpus:
        raise ValueError("Invalid number of GPUs per instance.")

    num_instances = total_gpus // gpus_per_instance
    processes = []

    for i in range(num_instances):
        gpu_indices = [
            str(j + 0)  
            for j in range(i * gpus_per_instance, (i + 1) * gpus_per_instance)
        ]
        cuda_visible_devices = ",".join(gpu_indices)
        port = base_port + i

        # Command to launch vLLM server
        command = [
            "vllm",
            "serve",
            model_path,
            "--tensor-parallel-size",
            str(gpus_per_instance),
            "--port",
            str(port),
            "--host",
            "0.0.0.0",
            # "--enable-reasoning",  # Fixed: missing comma
            # "--max-model-len",
            # "131072",  # Fixed: changed from 128000 to model's max
            # "32768",
            "--allowed-local-media-path",
            "/bigdisk/minhpvt/quick-test/na/ML/data",
            # "--gpu_memory_utilization",
            # "0.9"
        ]
        if max_model_len is not None:
            command.extend(["--max-model-len", str(max_model_len)])
        
        if enable_yarn:
            command.extend([
                "--rope-scaling",
                '{"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}'
            ])

        print(f"\nLaunching vLLM server instance:")
        print(f"  Port: {port}")
        print(f"  GPUs: {cuda_visible_devices}")
        print(f"  Command: {' '.join(command)}")

        # Set environment with specific GPUs
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices  # Fixed: use calculated GPUs
        env["NCCL_DEBUG"] = "INFO"

        # Create log files for debugging instead of suppressing output
        log_file = f"vllm_server_{port}.log"
        error_file = f"vllm_server_{port}_error.log"
        
        with open(log_file, 'w') as stdout_log, open(error_file, 'w') as stderr_log:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=stdout_log,
                stderr=stderr_log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True
            )
            processes.append((process, port, log_file, error_file))

    print(f"\nLaunched {num_instances} vLLM server instances.")
    print("Log files created:")
    for _, port, log_file, error_file in processes:
        print(f"  Port {port}: {log_file}, {error_file}")
    
    return processes

if __name__ == "__main__":
    fire.Fire(launch_vllm_servers)