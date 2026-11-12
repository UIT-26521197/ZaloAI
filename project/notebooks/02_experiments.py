# %% Chạy một cấu hình thử nghiệm trên cùng tập validation
from src.baseline.run import main

main(["--config", "configs/baseline.yaml", "--output", "outputs/experiments/exp_001",
      "--limit", "10"])
