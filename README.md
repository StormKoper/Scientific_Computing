# Scientific Computing
Tadhg Jones, David Kraakman, Storm Koper

run tests with "uv run pytest set_1"

run benchmarks with "uv run -m set_1.scripts.benchmarks -method SOR -iterations 1000 -N 500 -repeats 5 --warmup_jit"

run experiment B/C with "uv run -m set_1.scripts.run_wave1D -b_type i -steps 10000 -save_every 10 --animate"

run experiment E/F/G with "uv run -m set_1.scripts.run_wave2D -question E/F/G"